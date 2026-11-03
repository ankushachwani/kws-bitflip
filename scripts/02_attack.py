"""Run the bit-flip attacks and write raw per-flip curves and a summary.

Pre-declared before any run (do not change to chase a result):
  TARGET_ACC = 0.15   -- "broken" = 12-class evaluation accuracy at or below 15% (chance is 1/12 ~ 8.3%).
  MAX_FLIPS  = 30     -- stop a run after this many flips even if the target is not reached.
  ATTACK_BATCH = 64   -- clips per attack batch, drawn from the attack pool (disjoint from the eval set).
  SEEDS = 0,1,2       -- attack-batch draw (pbs) / flip RNG (rand).

Usage:  python3 02_attack.py [--models s,tc8] [--variants int8,w4a8] [--seeds 3] [--max-flips 30]
"""
import argparse, os, json, time
import numpy as np
from common import CACHE, RAW, LABELS, WBITS
from bitflip import load_model, quant_input, attack_pbs, attack_rand

TARGET_ACC = 0.15
MAX_FLIPS = 30
ATTACK_BATCH = 64

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='s,m,tc8')
ap.add_argument('--variants', default='int8,w4a8')
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--max-flips', type=int, default=MAX_FLIPS)
args = ap.parse_args()

sp = np.load(os.path.join(CACHE, 'split.npz'), allow_pickle=True)
labels, attack_pool, eval_idx = sp['labels'], sp['attack_pool'], sp['eval_idx']
FRONTEND = {'s': 'hello', 'm': 'hello', 'l': 'hello', 'tc8': 'tang', 'tc14_15': 'tang'}

summary = []
log = open(os.path.join(RAW, '02_attack.log'), 'w')


def logp(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


logp('target_acc', TARGET_ACC, 'max_flips', args.max_flips, 'attack_batch', ATTACK_BATCH,
     'eval_n', len(eval_idx), 'attack_pool_n', len(attack_pool))

for key in args.models.split(','):
    kind = FRONTEND[key]
    feat = np.load(os.path.join(CACHE, f'feat_{kind}.npy'))
    ye = labels[eval_idx]
    for variant in args.variants.split(','):
        for seed in range(args.seeds):
            rng = np.random.RandomState(100 + seed)
            abatch = rng.choice(attack_pool, ATTACK_BATCH, replace=False)
            ya = labels[abatch]
            for method in ('pbs', 'rand'):
                t0 = time.time()
                m = load_model(key, variant)            # fresh weights every run
                Xe = quant_input(feat[eval_idx], m['s_in_json'], m['net'], m['hi0'])
                Xa = quant_input(feat[abatch], m['s_in_json'], m['net'], m['hi0'])
                if method == 'pbs':
                    hist, reached = attack_pbs(m, Xa, ya, Xe, ye, TARGET_ACC, args.max_flips)
                else:
                    hist, reached = attack_rand(m, Xe, ye, TARGET_ACC, args.max_flips, seed)
                import pandas as pd
                df = pd.DataFrame(hist)
                df.insert(0, 'model', key); df.insert(1, 'variant', variant)
                df.insert(2, 'method', method); df.insert(3, 'seed', seed)
                df.to_csv(os.path.join(RAW, f'curve_{key}_{variant}_{method}_s{seed}.csv'), index=False)
                base_acc = hist[0]['acc']
                summary.append(dict(model=key, label=LABELS[key], variant=variant, wbits=WBITS[variant],
                                    method=method, seed=seed, total_weight_bits=m['total_bits'],
                                    base_eval_acc=round(base_acc, 4), flips_to_target=reached,
                                    final_acc=round(hist[-1]['acc'], 4), flips_run=hist[-1]['flip']))
                logp('%-8s %-5s %-4s s%d: base_acc %.3f flips_to_target %s final_acc %.3f  (%.0fs)'
                     % (key, variant, method, seed, base_acc, reached, hist[-1]['acc'], time.time() - t0))

# config constants are invocation-independent; 03_tables_figs.py aggregates all curve_*.csv on disk.
json.dump(dict(target_acc=TARGET_ACC, max_flips=args.max_flips, attack_batch=ATTACK_BATCH),
          open(os.path.join(RAW, 'config.json'), 'w'), indent=1)
logp('wrote', len(summary), 'curves; run 03_tables_figs.py to aggregate')
