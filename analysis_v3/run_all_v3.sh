#!/usr/bin/env bash
# Run the complete v3 pipeline. Each step's console output is saved to logs/.
# Usage (from inside analysis_v3/):
#   bash run_all_v3.sh              # all steps; transformer fine-tuning re-uses checkpoints if present
#   bash run_all_v3.sh --finetune   # also fine-tune DistilBERT (Step 9 random CV, Step 11 hotel-grouped CV)
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # mainland-China mirror; unset if huggingface.co is reachable
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

run () { echo ">>> $1  ($(date '+%H:%M:%S'))"; python "$@" 2>&1 | tee "logs/log_${1%.py}.txt"; }

run v3_01_preprocess_booking.py
run v3_02_preprocess_tripadvisor.py
run v3_03_model_comparison.py
run v3_04_ablation.py
run v3_05_imbalance.py
run v3_06_shap.py
run v3_07_cross_platform.py
run v3_08_supplementary.py
run v3_09_plm_baselines.py "$@"
run v3_10_shap_agreement.py
run v3_11_grouped_cv.py "$@"
run v3_12_absa_validation.py
run v3_13_checks.py
echo ">>> ALL STEPS FINISHED ($(date '+%H:%M:%S'))"
