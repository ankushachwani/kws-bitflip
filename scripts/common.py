"""Shared paths, model list and loaders. Everything runs from the repo root.

The keyword-spotting models, the integer simulator (qmodel.run_int), the blob loader and the MFCC
front end are reused from my earlier quantisation project so that the two studies are comparable.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.pylib'))      # torchaudio built for the installed torch
sys.path.insert(0, os.path.join(ROOT, 'vendor'))
import torch
torch.set_num_threads(2)   # keep the laptop usable while this runs

DATA = os.path.join(ROOT, 'data')
ASSETS = os.path.join(ROOT, 'assets')
RAW = os.path.join(ROOT, 'results', 'raw')
TABLES = os.path.join(ROOT, 'results', 'tables')
FIGS = os.path.join(ROOT, 'results', 'figures')
CACHE = os.path.join(ROOT, 'cache')
for d in (RAW, TABLES, FIGS, CACHE):
    os.makedirs(d, exist_ok=True)

MODELS = ['s', 'm', 'l', 'tc8', 'tc14_15']
LABELS = {'s': 'DS-CNN-S', 'm': 'DS-CNN-M', 'l': 'DS-CNN-L', 'tc8': 'TC-ResNet8', 'tc14_15': 'TC-ResNet14-1.5'}
# variant -> (weight bits, activation bits).  "int8" = 8-bit weights, "w4a8"/"w4a4" = 4-bit weights.
VARIANTS = ['int8', 'w4a8', 'w4a4']
WBITS = {'int8': 8, 'w4a8': 4, 'w4a4': 4}

KW = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']   # model outputs 2..11
CLASSES = ['silence', 'unknown'] + KW
FOLDER_TO_CLASS = {'_silence_': 0, '_unknown_': 1, **{w: i + 2 for i, w in enumerate(KW)}}
SILENCE = 0
NCLS = 12
CHANCE = 1.0 / NCLS       # 12-class random guess ~ 0.0833


def load_norm(kind):
    z = json.load(open(os.path.join(ASSETS, f'norm_{kind}.json')))
    return np.array(z['mu'], np.float32), np.array(z['sd'], np.float32)


def read_wav16k(path):
    """16-bit mono wav -> int16 vector padded/cut to 16000 samples."""
    from scipy.io import wavfile
    _, d = wavfile.read(path)
    out = np.zeros(16000, np.int16); out[:min(16000, len(d))] = d[:16000]
    return out
