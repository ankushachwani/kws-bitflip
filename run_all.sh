#!/bin/bash
# Reproduce everything from the model blobs and the Speech Commands test set.
#   data/test_set/   official v0.02 test set (12 class folders), from
#                    http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
# Threads are capped at 2 and the heavy steps run at background priority, so the laptop stays usable.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2

[ -d .pylib/torchaudio ] || pip install --target .pylib --no-deps torchaudio==2.11.0
[ -d data/test_set/_silence_ ] || { echo "put the official v0.02 test set in data/test_set"; exit 1; }

python3 -I tests/test_bitflip.py
taskpolicy -b nice -n 20 python3 scripts/01_features.py
taskpolicy -b nice -n 20 python3 scripts/02_attack.py --models s,m,tc8,tc14_15 --variants int8,w4a8,w4a4 --seeds 3 --max-flips 30
python3 scripts/03_tables_figs.py
python3 scripts/verify.py
