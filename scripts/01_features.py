"""Build the evaluation and attack data once.

Index the official Speech Commands v0.02 test set (12 classes), compute the MFCC features for the two front
ends the models use (hello = 10 MFCC for the DS-CNN models, tang = 40 MFCC for the TC-ResNet models), and fix a
disjoint split: an attack pool (attack batches are drawn from it) and an evaluation set (all reported accuracy
and the accuracy-vs-flips curves come from it). The split RNG seed is fixed here, before any attack is run.
"""
import os, json
import numpy as np
from common import DATA, CACHE, FOLDER_TO_CLASS, read_wav16k, load_norm
from frontend import features

TEST = os.path.join(DATA, 'test_set')
SPLIT_SEED = 0
ATTACK_POOL_N = 1200      # clips reserved for drawing attack batches
EVAL_N = 1500             # clips used for every reported accuracy / curve (disjoint from the pool)

# ---- index the test set ------------------------------------------------------------------------
rows = []
for folder in sorted(FOLDER_TO_CLASS):
    d = os.path.join(TEST, folder)
    for f in sorted(os.listdir(d)):
        if f.endswith('.wav'):
            rows.append((folder, f, FOLDER_TO_CLASS[folder]))
folders = [r[0] for r in rows]; files = [r[1] for r in rows]
labels = np.array([r[2] for r in rows], np.int64)
print('test clips', len(rows), 'class counts', np.bincount(labels).tolist())

# ---- waves + features per front end ------------------------------------------------------------
waves = np.stack([read_wav16k(os.path.join(TEST, fo, fi)) for fo, fi in zip(folders, files)])
feats = {}
for kind in ('hello', 'tang'):
    mu, sd = load_norm(kind)
    feats[kind] = features(waves, kind, mu, sd).astype(np.float32)
    print('features', kind, feats[kind].shape)
    np.save(os.path.join(CACHE, f'feat_{kind}.npy'), feats[kind])

# ---- disjoint split ----------------------------------------------------------------------------
rng = np.random.RandomState(SPLIT_SEED)
perm = rng.permutation(len(rows))
attack_pool = np.sort(perm[:ATTACK_POOL_N])
eval_idx = np.sort(perm[ATTACK_POOL_N:ATTACK_POOL_N + EVAL_N])
assert len(set(attack_pool) & set(eval_idx)) == 0
np.savez(os.path.join(CACHE, 'split.npz'), labels=labels, attack_pool=attack_pool, eval_idx=eval_idx,
         folders=np.array(folders), files=np.array(files))
json.dump(dict(split_seed=SPLIT_SEED, n_test=len(rows), attack_pool_n=int(len(attack_pool)),
               eval_n=int(len(eval_idx)), eval_class_counts=np.bincount(labels[eval_idx]).tolist(),
               attack_pool_class_counts=np.bincount(labels[attack_pool]).tolist()),
          open(os.path.join(CACHE, 'split_info.json'), 'w'), indent=1)
print('attack pool', len(attack_pool), 'eval', len(eval_idx), '(disjoint)')
