"""Aggregate every results/raw/curve_*.csv into results/raw/summary.csv, then build tables and one figure.
Every number here traces back to a raw curve file; nothing is typed in by hand."""
import os, glob, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RAW, TABLES, FIGS, LABELS, WBITS, CHANCE
from bitflip import load_model

cfg = json.load(open(os.path.join(RAW, 'config.json')))
TARGET = cfg['target_acc']
MAXFLIPS = cfg['max_flips']
VORDER = ['int8', 'w4a8', 'w4a4']
VNAME = {'int8': 'int8 (8-bit W)', 'w4a8': 'w4a8 (4-bit W)', 'w4a4': 'w4a4 (4-bit W)'}
PAT = re.compile(r'curve_(.+)_(int8|w4a8|w4a4)_(pbs|rand)_s(\d+)\.csv$')

# ---- rebuild the per-run summary from the curves ----------------------------------------------
_bits = {}
def total_bits(model, variant):
    if (model, variant) not in _bits:
        _bits[(model, variant)] = load_model(model, variant)['total_bits']
    return _bits[(model, variant)]

rows = []
for f in sorted(glob.glob(os.path.join(RAW, 'curve_*.csv'))):
    mobj = PAT.search(os.path.basename(f))
    if not mobj:
        continue
    model, variant, method, seed = mobj.group(1), mobj.group(2), mobj.group(3), int(mobj.group(4))
    c = pd.read_csv(f).sort_values('flip')
    hit = c[c['acc'] <= TARGET]
    reached = int(hit['flip'].iloc[0]) if len(hit) else None
    rows.append(dict(model=model, label=LABELS[model], variant=variant, wbits=WBITS[variant],
                     method=method, seed=seed, total_weight_bits=total_bits(model, variant),
                     base_eval_acc=round(float(c[c.flip == 0]['acc'].iloc[0]), 4),
                     flips_to_target=reached, final_acc=round(float(c['acc'].iloc[-1]), 4),
                     flips_run=int(c['flip'].iloc[-1])))
summ = pd.DataFrame(rows).sort_values(['model', 'variant', 'method', 'seed'])
summ.to_csv(os.path.join(RAW, 'summary.csv'), index=False)
MODELS = list(dict.fromkeys(summ['model']))     # models present, in first-seen order


def agg(g):
    f = g['flips_to_target'].astype(float)
    reached = f.notna()
    bits = g['total_weight_bits'].iloc[0]
    out = dict(n_seeds=len(g), base_acc=round(g['base_eval_acc'].mean(), 4),
               total_weight_bits=int(bits), reached=int(reached.sum()))
    if reached.any():
        out['flips_mean'] = round(f[reached].mean(), 2)
        out['flips_min'] = int(f[reached].min()); out['flips_max'] = int(f[reached].max())
        out['flips_sd'] = round(f[reached].std(ddof=0), 2) if reached.sum() > 1 else 0.0
        out['bits_per_million'] = round(1e6 * f[reached].mean() / bits, 3)
    else:
        for k in ('flips_mean', 'flips_min', 'flips_max', 'flips_sd', 'bits_per_million'):
            out[k] = np.nan
    out['final_acc_mean'] = round(g['final_acc'].mean(), 4)
    return pd.Series(out)


tbl = summ.groupby(['model', 'variant', 'method']).apply(agg, include_groups=False).reset_index()
tbl['label'] = tbl['model'].map(LABELS)
tbl.to_csv(os.path.join(TABLES, 'flips_summary.csv'), index=False)


def md_main():
    lines = [f'Flips to drive 12-class evaluation accuracy to <= {TARGET:.0%} (chance = {CHANCE:.1%}). '
             f'Mean over the seeds that reached it; "reached" counts seeds (of {int(tbl.n_seeds.max())}) that hit '
             f'the target within {MAXFLIPS} flips. "flips/Mbit" = mean flips per million stored weight bits.',
             '',
             '| model | variant | base acc | PBS flips [min,max] (reached) | PBS flips/Mbit | random flips (reached) | random final acc |',
             '|---|---|---|---|---|---|---|']
    for key in MODELS:
        for v in VORDER:
            p = tbl[(tbl.model == key) & (tbl.variant == v) & (tbl.method == 'pbs')]
            r = tbl[(tbl.model == key) & (tbl.variant == v) & (tbl.method == 'rand')]
            if p.empty:
                continue
            p = p.iloc[0]; r = r.iloc[0] if not r.empty else None
            pf = (f"{p.flips_mean:.1f} [{int(p.flips_min)},{int(p.flips_max)}] ({int(p.reached)}/{int(p.n_seeds)})"
                  if p.reached else f"not reached (0/{int(p.n_seeds)})")
            pm = f"{p.bits_per_million:.3f}" if p.reached else "-"
            if r is None:
                rf, rfa = "-", "-"
            else:
                rf = (f"{r.flips_mean:.1f} ({int(r.reached)}/{int(r.n_seeds)})" if r.reached
                      else f"not reached (0/{int(r.n_seeds)})")
                rfa = f"{r.final_acc_mean:.3f}"
            lines.append(f"| {LABELS[key]} | {VNAME[v]} | {p.base_acc:.3f} | {pf} | {pm} | {rf} | {rfa} |")
    return '\n'.join(lines)


open(os.path.join(TABLES, 'flips_summary.md'), 'w').write(md_main() + '\n')

# ---- paired int8-vs-4bit (PBS) ----------------------------------------------------------------
pl = []
for key in MODELS:
    base = tbl[(tbl.model == key) & (tbl.method == 'pbs')]
    i8 = base[base.variant == 'int8']
    if i8.empty or not i8.iloc[0].reached:
        continue
    i8 = i8.iloc[0]
    for v in ('w4a8', 'w4a4'):
        g = base[base.variant == v]
        if g.empty or not g.iloc[0].reached:
            continue
        g = g.iloc[0]
        pl.append(dict(model=LABELS[key], contrast=f'int8 vs {v}',
                       int8_flips=i8.flips_mean, v4_flips=g.flips_mean,
                       int8_flips_per_Mbit=i8.bits_per_million, v4_flips_per_Mbit=g.bits_per_million,
                       int8_bits=i8.total_weight_bits, v4_bits=g.total_weight_bits))
pd.DataFrame(pl).to_csv(os.path.join(TABLES, 'paired_int8_vs_4bit.csv'), index=False)

# ---- figure: accuracy vs flips ----------------------------------------------------------------
fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.3), sharey=True, squeeze=False)
axes = axes[0]
col = {'int8': '#1b6ca8', 'w4a8': '#d1495b', 'w4a4': '#e3a008'}
for ax, key in zip(axes, MODELS):
    for v in VORDER:
        for method, ls, al, lw in (('pbs', '-', 1.0, 1.9), ('rand', '--', 0.55, 1.2)):
            files = sorted(glob.glob(os.path.join(RAW, f'curve_{key}_{v}_{method}_s*.csv')))
            if not files:
                continue
            curves = [pd.read_csv(f) for f in files]
            mx = max(c['flip'].max() for c in curves)
            grid = np.arange(0, mx + 1)
            M = np.vstack([pd.Series(c.set_index('flip')['acc']).reindex(grid).ffill().values for c in curves])
            lbl = f'{v} PBS' if method == 'pbs' else None
            ax.plot(grid, np.nanmean(M, 0), ls, color=col[v], alpha=al, lw=lw, label=lbl)
    ax.axhline(CHANCE, color='grey', lw=0.8, ls=':')
    ax.axhline(TARGET, color='k', lw=0.6, ls='-', alpha=0.3)
    ax.set_title(LABELS[key], fontsize=10); ax.set_xlabel('bit flips')
    ax.set_xlim(0, MAXFLIPS); ax.set_ylim(0, 1)
axes[0].set_ylabel('12-class eval accuracy')
axes[0].legend(fontsize=7, loc='upper right', title='solid = PBS\ndashed = random', title_fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'accuracy_vs_flips.png'), dpi=150, metadata={'Software': None})
print('wrote results/raw/summary.csv, tables, and results/figures/accuracy_vs_flips.png')
print(md_main())
