# analysis_v3 — changes relative to analysis_v2

analysis_v2 is left untouched. v3 is a copy of v2 with the fixes below; every
step is re-run so that the manuscript reports numbers from one consistent
pipeline.

## Methodological fixes (change results)

1. **Word-boundary aspect matching** (`common.detect_aspects`, `common.ASPECT_RE`).
   v2 matched keywords as raw substrings, so `eat` fired inside *great*,
   `view` inside *review*, `spa` inside *space*, `bus` inside *business*,
   `ac` inside *place* / *back*, `fee` inside *coffee* / *feel*, etc. This
   inflated aspect mention rates (e.g. Facilities, Food & Beverage, Value) and
   the aspect-sentiment features. v3 matches whole words with an optional
   plural suffix; `welcom`, `disturb` and `air condition` are intentional
   stems and match as prefixes.

2. **Segment-level aspect sentiment** (`common.split_segments`,
   `common.aspect_sentiments`). v2 split "sentences" on `.!?` **after**
   `clean_text()` had removed all punctuation, so every review was one
   sentence and each mentioned aspect received the whole-review VADER score.
   v3 segments the **raw** text on punctuation (TripAdvisor) and on
   sentence-case boundaries (Booking.com comment fields contain no
   punctuation but keep sentence case); Booking.com's negative and positive
   fields are always separate segments. Each segment is scored with VADER on
   the raw segment and aspect scores are averaged over matched segments.

3. **Overall sentiment (Group B) on raw text**, so VADER can use
   capitalisation cues.

## Additions

4. `booking_mentions.csv` / `tripadvisor_mentions.csv`: binary aspect-mention
   indicators; used by a new ablation configuration
   *Full + aspect-mention indicators* (34 features), which tests the
   "unmentioned aspect = 0" imputation.
5. Ablation: the duplicate configuration *Text only (A+B+C)* (identical to
   *Full − D*) is removed.
6. Step 8 (a2): SHAP importance-ranking stability across the three label
   binning schemes (Spearman ρ, Kendall τ, top-10 overlap).
7. Step 9 (new): pretrained language model baselines on the same sample and
   folds — frozen all-MiniLM-L6-v2 and all-mpnet-base-v2 embeddings +
   class-weighted logistic regression, and optionally fine-tuned DistilBERT;
   corrected resampled t-tests against the 26-feature RF and TF-IDF + LR;
   wall-clock cost of every configuration.
8. Text-feature extraction uses a process pool (much faster on multi-core
   machines).

## Unchanged

Placeholder cleaning, label binning, sample (50,000, seed 42), folds, all
model hyper-parameters, statistical tests, SHAP protocol, cross-platform
matching procedure.

## Post-run fix (explanation sample)

9. `v3_03` now stores the ORIGINAL row ids of the 50,000-review modelling
   sample (`train_index`). Earlier versions stored the reset index 0..49,999,
   so Step 6 treated rows >= 50,000 as held out and roughly 10% of the SHAP
   explanation sample had in fact been used for training. `v3_06` now asserts
   that the explanation sample and the modelling sample are disjoint and
   stores the explanation row ids.
10. New Step 10 (`v3_10_shap_agreement.py`): TreeSHAP for the reference random
    forest on exactly the same 1,000 held-out reviews, and agreement with the
    ranking of the explained model (Spearman, Kendall, top-10 overlap, group shares).

## v3.1 (response to pre-submission review)

11. Nominal metadata (trip type, traveller type, hotel country) are one-hot
    encoded with fixed, schema-defined levels (37 model columns for the 26
    conceptual features); SHAP values of the dummies are summed back to the
    nominal variable (`common.aggregate_shap`). v3.0 used integer label codes.
12. Keyword matching accepts standard inflectional endings
    (-s, -es, -d, -ed, -ing, -ly, -ness, -liness, -er, -ers); v3.0 accepted
    only -s/-es and missed e.g. "cleaned", "cleanliness".
13. `booking_meta.csv` stores the hotel name (for grouped CV).
14. Step 7: a strictly matched condition (common length window 30-150 words,
    ten length bins, identical counts per bin x class on both platforms).
15. Step 11 (new): hotel-grouped StratifiedGroupKFold for all core comparisons,
    incl. optional DistilBERT fine-tuning.
16. Step 12 (new): external validation of aspect detection and VADER polarity
    against the human-annotated HRAST sentences (Andreou et al. 2023).
17. Step 13 (new): language identification (lingua) and TOST equivalence tests.
18. Figures are written at 600 dpi.
19. Step 11: `--ft-only` runs the DistilBERT fine-tuning on the hotel-grouped folds in a
    separate process, and XGBoost/LightGBM are imported only for the classical part. On macOS,
    torch/MPS deadlocked or crashed when the OpenMP runtime of XGBoost/LightGBM had been loaded
    in the same process. `rerun_step11_13.sh` runs the fine-tuning first, then Step 11 (loading
    the fold checkpoints), Step 12 and Step 13. Results are unaffected.

20. v3.2 (documentation only, no change to any computation): README rewritten to the PeerJ code-sharing structure (description, dataset information, code information, usage, requirements, methodology, citations, licence/contributions); pinned package versions of the reported run in requirements_v3.txt; HRAST download link in dataset/README.md; CITATION.cff; manuscript_figures/make_manuscript_figures.py recreates the manuscript figures (Figs 1-4, S1-S2) from results/ and figures/.
