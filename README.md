# How much accuracy does interpretability cost? Aspect-based features versus transformer models for hotel guest satisfaction

Reproducibility package for the manuscript *"How much accuracy does interpretability cost?
Aspect-based features versus transformer models for classifying hotel guest satisfaction across
two review platforms"* (submitted to *PeerJ Computer Science*).

This is the **v3.1 pipeline**, which produced every number, table and figure in the manuscript.
`analysis_v3/CHANGES_v3.md` documents all changes relative to the earlier v2 release
(word-boundary aspect matching with inflectional endings, segment-level aspect sentiment on raw
text, one-hot metadata with aggregated SHAP values, transformer baselines, hotel-grouped
cross-validation, validation against human annotations, language check and equivalence tests).

## Structure

```
analysis_v3/
  common.py, stats_utils.py        configuration, lexicon, text features, one-hot schema, statistical tests
  v3_01_preprocess_booking.py      placeholder cleaning, labels, 26 features / 37 columns (Booking.com)
  v3_02_preprocess_tripadvisor.py  14 shared features (TripAdvisor)
  v3_03_model_comparison.py        7 models, corrected resampled t-tests with Holm correction (Table 2)
  v3_04_ablation.py                ablation and TF-IDF baselines (Table 3)
  v3_05_imbalance.py               class-imbalance strategies (Table 7)
  v3_06_shap.py                    TreeSHAP on the evaluated model, held-out reviews (Figs 3-5, Table 6)
  v3_07_cross_platform.py          cross-platform transfer, matched and strictly matched (Tables 8-9, S5)
  v3_08_supplementary.py           thresholds + SHAP stability, learning curve, nationality audit (S6, S7, Fig S1)
  v3_09_plm_baselines.py           frozen MiniLM/MPNet + LR, fine-tuned DistilBERT, compute cost (Table 4, Fig 2)
  v3_10_shap_agreement.py          RF vs explained-model SHAP agreement on the same reviews
  v3_11_grouped_cv.py              hotel-grouped cross-validation incl. DistilBERT (Table 5)
  v3_12_absa_validation.py         aspect detection and VADER polarity vs HRAST human labels (Table 1, S3)
  v3_13_checks.py                  language identification and TOST equivalence tests (Table S4)
  run_all_v3.sh                    runs every step and writes logs/
  rerun_step11_13.sh               macOS-safe rerun of Steps 11-13 (fine-tuning in a separate process)
  results/  figures/  logs/        outputs of the run reported in the manuscript
dataset/README.md                  where to download the three public datasets
```

## Running

```bash
pip install -r analysis_v3/requirements_v3.txt
cd analysis_v3
bash run_all_v3.sh --finetune     # all steps incl. DistilBERT fine-tuning (GPU recommended)
```

On macOS (MPS), run the fine-tuning of Step 11 in its own process with
`bash rerun_step11_13.sh` after Steps 1-10 (see CHANGES item 19). Fine-tuning results are
checkpointed per fold, so reruns reuse them. Hugging Face models are downloaded on first use
(`run_all_v3.sh` sets `HF_ENDPOINT` to a mirror; unset it if huggingface.co is reachable).
Random seed 42 throughout.

The reported run used an Apple Mac Studio (M1 Ultra, 20 CPU cores, 128 GB) with macOS 26.6.2,
Python 3.12.0, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, imbalanced-learn
0.14.2, XGBoost 3.4.1, LightGBM 4.7.0, SHAP 0.52.0, vaderSentiment and lingua; transformers ran
on the GPU via Metal Performance Shaders.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences.
