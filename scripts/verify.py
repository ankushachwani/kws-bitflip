"""Independent recomputation of the headline numbers from the raw per-flip curves.

Reads only results/raw/curve_*.csv (not summary.csv), recomputes each run's baseline accuracy and the first
flip at which eval accuracy reaches the pre-declared target, and checks these against results/raw/summary.csv
and results/tables/flips_summary.csv. Exits non-zero on any mismatch.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from common import RAW, TABLES

cfg = json.load(open(os.path.join(RAW, 'config.json')))
TARGET = cfg['target_acc']
fails = []


def first_at_or_below(c, target):
    hit = c[c['acc'] <= target]
    return int(hit['flip'].iloc[0]) if len(hit) else None


# recompute per-run from curves
rows = []
for f in sorted(glob.glob(os.path.join(RAW, 'curve_*.csv'))):
    c = pd.read_csv(f).sort_values('flip')
    model, variant, method, seed = c.iloc[0][['model', 'variant', 'method', 'seed']]
    rows.append(dict(model=model, variant=variant, method=method, seed=int(seed),
                     base=round(float(c[c.flip == 0]['acc'].iloc[0]), 4),
                     flips_to_target=first_at_or_below(c, TARGET),
                     final=round(float(c['acc'].iloc[-1]), 4)))
rec = pd.DataFrame(rows)

summ = pd.read_csv(os.path.join(RAW, 'summary.csv'))
key = ['model', 'variant', 'method', 'seed']
m = rec.merge(summ, on=key, suffixes=('_rec', ''))
for _, r in m.iterrows():
    tag = f"{r.model}/{r.variant}/{r.method}/s{r.seed}"
    if abs(r.base - r.base_eval_acc) > 1e-6:
        fails.append(f"{tag}: base acc {r.base} != summary {r.base_eval_acc}")
    a = None if pd.isna(r.flips_to_target_rec) else int(r.flips_to_target_rec)
    b = None if pd.isna(r.flips_to_target) else int(r.flips_to_target)
    if a != b:
        fails.append(f"{tag}: flips_to_target {a} != summary {b}")
    if abs(r.final - r.final_acc) > 1e-6:
        fails.append(f"{tag}: final acc {r.final} != summary {r.final_acc}")
if len(m) != len(summ):
    fails.append(f"row count: {len(m)} matched vs {len(summ)} summary rows")

# recompute the group means in flips_summary.csv (PBS, reached seeds only)
tbl = pd.read_csv(os.path.join(TABLES, 'flips_summary.csv'))
for _, t in tbl.iterrows():
    g = rec[(rec.model == t.model) & (rec.variant == t.variant) & (rec.method == t.method)]
    reached = g['flips_to_target'].dropna()
    if len(reached):
        mean = round(reached.mean(), 2)
        if not pd.isna(t.flips_mean) and abs(mean - t.flips_mean) > 1e-6:
            fails.append(f"{t.model}/{t.variant}/{t.method}: flips_mean {mean} != table {t.flips_mean}")
    if int((~g['flips_to_target'].isna()).sum()) != int(t.reached):
        fails.append(f"{t.model}/{t.variant}/{t.method}: reached {int((~g['flips_to_target'].isna()).sum())} != table {t.reached}")

print(f"checked {len(m)} runs against summary, {len(tbl)} groups against flips_summary.csv")
if fails:
    print('MISMATCHES:')
    for x in fails:
        print('  ', x)
    sys.exit(1)
print('OK: all headline numbers recomputed from raw curves match the tables.')
