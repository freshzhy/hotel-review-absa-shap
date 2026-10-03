#!/usr/bin/env bash
# Finish Steps 11-13 after the DistilBERT grouped-CV hang.
# Fine-tuning runs in its own process first (before XGBoost/LightGBM load OpenMP),
# then Step 11 re-runs and loads the fold checkpoints, then Steps 12 and 13.
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
run () { echo ">>> $1  ($(date '+%H:%M:%S'))"; python "$@" 2>&1 | tee "logs/log_${1%.py}${LOGSUFFIX}.txt"; }
LOGSUFFIX=_finetune run v3_11_grouped_cv.py --ft-only
LOGSUFFIX=        run v3_11_grouped_cv.py --finetune
run v3_12_absa_validation.py
run v3_13_checks.py
echo ">>> STEPS 11-13 FINISHED ($(date '+%H:%M:%S'))"
