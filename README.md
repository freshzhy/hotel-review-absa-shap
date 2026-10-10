# How much accuracy does interpretability cost? Aspect-based features versus transformer models for classifying hotel guest satisfaction across two review platforms

Code, results and run logs for the manuscript of the same title by Haoyu Zhang, Mei Wang and Ling Li
(Sichuan Tourism University), submitted to *PeerJ Computer Science*.



---

## 1. Description

The repository contains the complete analysis pipeline (version 3.2 of the code, identical in its computations to
v3.1) that produced every number, table and figure in the manuscript. The pipeline classifies hotel reviews into three
guest-satisfaction classes (Low, Medium, High) and asks how much accuracy is lost when an interpretable,
semantically defined feature representation is used instead of a fine-tuned transformer. It

1. cleans Booking.com and TripAdvisor reviews (including removal of interface placeholder strings such as
   "No Negative") and checks their language;
2. engineers 26 interpretable features: segment-level aspect-based sentiment for eight hotel aspects (seed-word
   lexicon + VADER), overall sentiment, review statistics and contextual metadata;
3. validates aspect detection and VADER polarity against the human-annotated HRAST corpus;
4. compares seven classifiers, an ablation, TF-IDF, frozen sentence-encoder and fine-tuned DistilBERT baselines, and
   ten class-imbalance strategies under leakage-safe 5-fold cross-validation, with corrected resampled t-tests,
   Holm–Bonferroni correction, TOST equivalence tests and hotel-grouped cross-validation;
5. explains the evaluated XGBoost model with TreeSHAP on held-out reviews and checks the stability of the ranking;
6. tests cross-platform transfer (Booking.com ↔ TripAdvisor) with and without matching on review length and class
   distribution, plus robustness analyses (label thresholds, learning curve, nationality audit, language).

## 2. Dataset information

Three public, third-party datasets are used. They are **not redistributed** here; download them from the links below
and place them in `dataset/` as described in `dataset/README.md`.

| Dataset | Role | Size | Source | Licence |
|---|---|---|---|---|
| 515K Hotel Reviews Data in Europe (Booking.com) | main data (modelling, SHAP, transfer) | 515,738 reviews, 1,492 hotels, 2015–2017 | https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe | see Kaggle page |
| TripAdvisor Hotel Reviews | second platform (transfer) | 20,491 reviews, 1–5 stars | https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews | see Kaggle page |
| HRAST: Hotel Reviews – Aspects, Sentiments and Topics | validation only (never used for training) | 23,113 human-annotated sentences | https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics | CC BY-NC 4.0 |

Expected layout:

```
dataset/
├── 515K Hotel Reviews Data in Europe/Hotel_Reviews.csv
├── Trip Advisor Hotel Reviews/tripadvisor_hotel_reviews.csv
└── Hotel Reviews- Aspects, Sentiments and Topics .zip   (or HRAST.csv)
```

Labels: Booking.com scores Low = [2.5, 4], Medium = (4, 7], High = (7, 10]; TripAdvisor 1–2 stars Low, 3 Medium,
4–5 High. After cleaning, 499,061 Booking.com reviews remain; all comparisons use a stratified sample of 50,000.

## 3. Code information

```
analysis_v3/
  common.py                        paths, seed, aspect lexicon, placeholder patterns, text features, one-hot schema
  stats_utils.py                   corrected resampled t-test, Holm–Bonferroni, TOST, bootstrap CIs
  v3_01_preprocess_booking.py      placeholder cleaning, labels, 26 features / 37 columns (Booking.com)
  v3_02_preprocess_tripadvisor.py  14 shared features (TripAdvisor)
  v3_03_model_comparison.py        seven models + pairwise tests                     -> Table 2
  v3_04_ablation.py                ablation and TF-IDF baselines                      -> Table 3
  v3_05_imbalance.py               ten class-imbalance strategies                     -> Table S7
  v3_06_shap.py                    TreeSHAP on the evaluated model, held-out reviews  -> Figs 3, 4, S1; Table S6
  v3_07_cross_platform.py          unmatched, matched and strictly matched transfer  -> Tables 5, 6, S8
  v3_08_supplementary.py           thresholds + SHAP stability, learning curve, nationality audit -> Tables S9, S10; Fig S2
  v3_09_plm_baselines.py           frozen MiniLM/MPNet + LR, fine-tuned DistilBERT, compute cost -> Table 4, Fig 2
  v3_10_shap_agreement.py          RF vs XGBoost SHAP agreement on the same reviews  -> text
  v3_11_grouped_cv.py              hotel-grouped cross-validation incl. DistilBERT   -> Table S5
  v3_12_absa_validation.py         aspect detection and VADER polarity vs HRAST       -> Table 1, Table S3
  v3_13_checks.py                  language identification and TOST equivalence tests -> Table S4
  run_all_v3.sh                    runs all steps in order and writes logs/
  rerun_step11_13.sh               macOS-safe rerun of Steps 11–13 (fine-tuning in a separate process)
  requirements_v3.txt              Python dependencies
  results/                         all result files (JSON/CSV) of the reported run
  figures/                         raw figures written by the pipeline
  logs/                            console logs of the reported run
  CHANGES_v3.md                    changes relative to the earlier v2 pipeline
manuscript_figures/
  make_manuscript_figures.py       builds Figs 1–4 and S1–S2 of the manuscript from analysis_v3/results and figures
dataset/README.md                  download instructions for the three datasets
CITATION.cff                       citation metadata
LICENSE                            MIT licence
```

Figure 1 (framework diagram) is drawn by `make_manuscript_figures.py`; all other figures and every table value are
computed from the files in `analysis_v3/results/`.

## 4. Requirements

- Python 3.12 (tested with 3.12.0). A GPU (CUDA or Apple MPS) is recommended for Steps 9 and 11 (DistilBERT
  fine-tuning), which otherwise run on the CPU but slowly.
- Packages (see `analysis_v3/requirements_v3.txt`). Versions used for the reported run: numpy 2.5.3, pandas 3.0.6,
  scipy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, lightgbm 4.7.0, shap 0.52.0,
  vaderSentiment, lingua-language-detector, matplotlib, joblib; for the transformer baselines torch, transformers and
  sentence-transformers.
- Hardware of the reported run: Apple Mac Studio, M1 Ultra (20 CPU cores), 128 GB memory, macOS 26.6.2.

## 5. Usage instructions

```bash
# 1. get the code and create an environment
git clone <this repository>   # or download and unzip the Zenodo archive
cd <repository>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r analysis_v3/requirements_v3.txt

# 2. download the three datasets into dataset/ (see Section 2)

# 3. run the full pipeline (all steps, including DistilBERT fine-tuning)
cd analysis_v3
bash run_all_v3.sh --finetune          # logs go to analysis_v3/logs/, results to analysis_v3/results/

# 4. rebuild the manuscript figures
cd ..
python manuscript_figures/make_manuscript_figures.py   # -> manuscript_figures/output/
```

Notes:
- On macOS with MPS, run the fine-tuning of Step 11 in its own process: after Steps 1–10, use
  `bash rerun_step11_13.sh` (see item 19 of `CHANGES_v3.md`). Fine-tuning results are checkpointed per fold, so reruns
  reuse them.
- Hugging Face models are downloaded on first use; `run_all_v3.sh` sets `HF_ENDPOINT` to a mirror, which can be unset
  where huggingface.co is reachable.
- The random seed is 42 throughout. Steps 9 and 11 (DistilBERT fine-tuned on five folds each) take by far the
  longest; the per-fold timings of the reported run are in `analysis_v3/results/plm_baselines.json`.
- To inspect the reported results without rerunning anything, open the JSON/CSV files in `analysis_v3/results/`.

## 6. Methodology (summary)

1. **Preprocessing.** 14 case-insensitive regular expressions detect placeholder comments; they are blanked and their
   word counts set to zero; reviews with fewer than three content words are dropped; language is checked with lingua.
2. **Features.** Reviews are split into segments (punctuation on TripAdvisor, sentence-case boundaries on Booking.com);
   segments are assigned to aspects by word-boundary keyword matching with inflectional endings and scored with the
   VADER compound score; aspect scores are segment means (0 if unmentioned). Overall VADER sentiment, review statistics
   and one-hot metadata complete the 26 features (37 columns).
3. **Validation.** HRAST topics are mapped onto the eight aspects; precision/recall/F1 of aspect detection and agreement
   of VADER polarity with human labels are computed.
4. **Evaluation.** Stratified 5-fold CV on 50,000 reviews with identical folds for all experiments; scaling, resampling
   and TF-IDF are fitted inside training folds; corrected resampled t-test (Nadeau & Bengio) with Holm–Bonferroni
   correction; TOST equivalence at ±0.02 macro-F1; StratifiedGroupKFold by hotel as a sensitivity analysis.
5. **Explanation.** TreeSHAP on the evaluated XGBoost model for 1,000 held-out reviews; one-hot SHAP values summed per
   nominal variable; stability checked against a random forest and across label thresholds.
6. **Transfer.** 14 shared features; unmatched, approximately matched and strictly matched (30–150 words, equal counts
   per length bin and class) conditions; bootstrap confidence intervals.

## 7. Citation

If you use this code, please cite the article (citation details will be added on publication) and this archive:

> Zhang H, Wang M, Li L. 2026. Code for "How much accuracy does interpretability cost? Aspect-based features versus
> transformer models for classifying hotel guest satisfaction across two review platforms". Zenodo.
> https://doi.org/10.5281/zenodo.23114483

Please also cite the datasets:

- Liu J. 2017. 515K hotel reviews data in Europe. Kaggle.
- Alam MH. 2019. TripAdvisor hotel reviews. Kaggle.
- Andreou C, Tsapatsoulis N, Anastasopoulou V. 2023. A dataset of hotel reviews for aspect-based sentiment analysis and
  topic modeling. In: 2023 18th International Workshop on Semantic and Social Media Adaptation & Personalization
  (SMAP). IEEE, 1–9. https://doi.org/10.1109/SMAP59435.2023.10255219

## 8. License and contributions

- Code: MIT licence (see `LICENSE`). The datasets keep their own licences; HRAST is CC BY-NC 4.0 (non-commercial use).
- Contributions: bug reports and questions are welcome through the repository's issue tracker or by e-mail to the
  corresponding author (Ling Li, linglidg@qq.com). Please describe the step, the command and the error message; pull
  requests should not change the reported results without documenting the change in `CHANGES_v3.md`.
