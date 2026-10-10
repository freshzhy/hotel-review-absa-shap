"""
v3 Step 3 - Model comparison (manuscript Table 2 / Table 4).

Changes relative to step2_modeling.py
-------------------------------------
1. Two naive baselines (majority-class and stratified-random) are included,
   so that every model can be read against the trivial reference point.
2. Standardisation is fitted INSIDE each training fold.  v1 called
   StandardScaler().fit_transform() on the whole sample before splitting,
   which let validation-fold statistics inform the training representation.
3. XGBoost no longer carries scale_pos_weight, which is a binary-only
   parameter and is silently ignored under multi:softprob.  It now receives
   per-sample balanced weights, matching the class_weight='balanced'
   treatment given to the other learners.  (Note: in v1, XGBoost was not
   left unbalanced - it received SMOTE inside each fold - but its handling
   still differed from every other model.)
4. Per-fold scores are retained and written out, and model differences are
   tested with the corrected resampled t-test (Nadeau & Bengio 2003) plus a
   Friedman omnibus test with Nemenyi critical difference.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import stats_utils as S

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


log('=' * 70)
log('v3 Step 3 | Model comparison with naive baselines and significance tests')
log('=' * 70)

X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values

X_s, _, y_s, _ = train_test_split(X_all, y_all, train_size=C.SAMPLE_SIZE,
                                  stratify=y_all, random_state=C.RANDOM_STATE)
# v3 fix: keep the ORIGINAL row numbers of the modelling sample so that Step 6
# can draw its explanation sample from reviews that were genuinely not used for
# training (v2/v3-initial saved the reset index 0..49,999 instead).
sample_row_ids = X_s.index.to_numpy().copy()
X_s = X_s.reset_index(drop=True)

le = LabelEncoder().fit(C.CLASS_NAMES)          # Low=0, Medium=1, High=2
y_enc = le.transform(y_s)
log(f'Sample: {X_s.shape}   class order {le.classes_.tolist()}')
for i, c in enumerate(le.classes_):
    log(f'  {c:<8s} {(y_enc == i).sum():>7,}  {(y_enc == i).mean()*100:5.2f}%')


# ------------------------------------------------------------------
# Model factories.  Each returns a fresh estimator per fold.
# ------------------------------------------------------------------
def make_models():
    return {
        'Majority-class baseline': lambda: DummyClassifier(
            strategy='most_frequent'),
        'Stratified-random baseline': lambda: DummyClassifier(
            strategy='stratified', random_state=C.RANDOM_STATE),
        'Logistic Regression': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', solver='lbfgs',
            random_state=C.RANDOM_STATE, n_jobs=-1),
        'Linear SVM': lambda: LinearSVC(
            class_weight='balanced', max_iter=2000, dual='auto',
            random_state=C.RANDOM_STATE),
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=C.RANDOM_STATE, n_jobs=-1),
        'XGBoost': lambda: XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective='multi:softprob', num_class=3,
            random_state=C.RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss',
            verbosity=0),
        'LightGBM': lambda: LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, class_weight='balanced',
            random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1),
    }


NEEDS_SAMPLE_WEIGHT = {'XGBoost'}
NEEDS_SCALING = {'Logistic Regression', 'Linear SVM'}

cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
splits = list(cv.split(X_s, y_enc))
n_train = len(splits[0][0])
n_test = len(splits[0][1])
log(f'\n{C.N_FOLDS}-fold CV: {n_train:,} train / {n_test:,} validation per fold')

results, fold_f1 = {}, {}

for name, factory in make_models().items():
    log(f'\n{"-"*60}\n{name}')
    t0 = time.time()
    per_fold = {k: [] for k in ['accuracy', 'precision_macro', 'recall_macro',
                                'f1_macro', 'f1_weighted', 'kappa']}
    per_class = []
    cm_sum = np.zeros((3, 3), dtype=int)

    for fi, (tr, va) in enumerate(splits):
        Xtr, Xva = X_s.iloc[tr].values, X_s.iloc[va].values
        ytr, yva = y_enc[tr], y_enc[va]

        # Scaling fitted on the training fold only.
        scaler = StandardScaler().fit(Xtr)
        Xtr_p, Xva_p = scaler.transform(Xtr), scaler.transform(Xva)

        model = factory()
        if name in NEEDS_SAMPLE_WEIGHT:
            sw = compute_sample_weight('balanced', ytr)
            model.fit(Xtr_p, ytr, sample_weight=sw)
        else:
            model.fit(Xtr_p, ytr)
        pred = model.predict(Xva_p)

        per_fold['accuracy'].append(accuracy_score(yva, pred))
        per_fold['precision_macro'].append(precision_score(yva, pred, average='macro', zero_division=0))
        per_fold['recall_macro'].append(recall_score(yva, pred, average='macro', zero_division=0))
        per_fold['f1_macro'].append(f1_score(yva, pred, average='macro', zero_division=0))
        per_fold['f1_weighted'].append(f1_score(yva, pred, average='weighted', zero_division=0))
        per_fold['kappa'].append(cohen_kappa_score(yva, pred))
        per_class.append(f1_score(yva, pred, average=None, zero_division=0,
                                  labels=[0, 1, 2]))
        cm_sum += confusion_matrix(yva, pred, labels=[0, 1, 2])
        log(f'  fold {fi+1}: acc={per_fold["accuracy"][-1]:.4f} '
            f'F1m={per_fold["f1_macro"][-1]:.4f}')

    entry = {m: S.mean_ci(v) for m, v in per_fold.items()}
    entry['f1_per_class'] = {c: S.mean_ci([row[i] for row in per_class])
                             for i, c in enumerate(le.classes_)}
    entry['confusion_matrix'] = cm_sum.tolist()
    entry['confusion_matrix_labels'] = le.classes_.tolist()
    entry['fold_scores'] = {m: [float(x) for x in v] for m, v in per_fold.items()}
    entry['fit_seconds'] = round(time.time() - t0, 1)
    results[name] = entry
    fold_f1[name] = per_fold['f1_macro']

    log(f'  => F1_macro {S.fmt_ci(entry["f1_macro"])}   '
        f'acc {S.fmt_ci(entry["accuracy"])}   ({entry["fit_seconds"]}s)')

# ------------------------------------------------------------------
# Significance testing on macro-F1
# ------------------------------------------------------------------
log(f'\n{"="*70}\nSignificance testing (macro-F1)')

pairwise = S.pairwise_comparisons(fold_f1, n_train, n_test)
log('\nCorrected resampled paired t-tests (Holm-adjusted):')
for r in sorted(pairwise, key=lambda r: r['p_holm']):
    flag = '*' if r['significant_05'] else ' '
    log(f'  {flag} {r["model_a"]:<28s} vs {r["model_b"]:<28s} '
        f'd={r["mean_diff"]:+.4f}  p={S.fmt_p(r["p_holm"])}')

learners = {k: v for k, v in fold_f1.items() if 'baseline' not in k}
fried = S.friedman_nemenyi(learners)
log(f'\nFriedman across the {fried.get("k_models")} learners: '
    f'chi2={fried.get("chi2", float("nan")):.3f}, p={S.fmt_p(fried.get("p", 1))}')
if fried.get('critical_difference'):
    log(f'Nemenyi critical difference (alpha=0.05): {fried["critical_difference"]:.3f}')
    for nm, rk in sorted(fried['mean_ranks'].items(), key=lambda kv: kv[1]):
        log(f'  mean rank {rk:.2f}  {nm}')

# ------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------
rows = []
for name, r in results.items():
    rows.append({
        'Model': name,
        'Accuracy': f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['sd']:.4f}",
        'Precision_macro': f"{r['precision_macro']['mean']:.4f} ± {r['precision_macro']['sd']:.4f}",
        'Recall_macro': f"{r['recall_macro']['mean']:.4f} ± {r['recall_macro']['sd']:.4f}",
        'F1_macro': f"{r['f1_macro']['mean']:.4f} ± {r['f1_macro']['sd']:.4f}",
        'F1_macro_95CI': f"[{r['f1_macro']['ci_low']:.4f}, {r['f1_macro']['ci_high']:.4f}]",
        'F1_weighted': f"{r['f1_weighted']['mean']:.4f} ± {r['f1_weighted']['sd']:.4f}",
        'Kappa': f"{r['kappa']['mean']:.4f} ± {r['kappa']['sd']:.4f}",
        'F1_Low': f"{r['f1_per_class']['Low']['mean']:.4f}",
        'F1_Medium': f"{r['f1_per_class']['Medium']['mean']:.4f}",
        'F1_High': f"{r['f1_per_class']['High']['mean']:.4f}",
        'Seconds': r['fit_seconds'],
    })
df_out = pd.DataFrame(rows)
log('\n' + df_out.to_string(index=False))
df_out.to_csv(os.path.join(C.RESULT_DIR, 'model_comparison.csv'), index=False)

pd.DataFrame(pairwise).to_csv(
    os.path.join(C.RESULT_DIR, 'model_comparison_pairwise_tests.csv'), index=False)

with open(os.path.join(C.RESULT_DIR, 'model_comparison.json'), 'w') as f:
    json.dump({'results': results, 'pairwise': pairwise, 'friedman': fried,
               'n_train_per_fold': n_train, 'n_test_per_fold': n_test,
               'class_order': le.classes_.tolist(),
               'versions': C.version_record()}, f, indent=2)

# ------------------------------------------------------------------
# Refit the best learner on the full sample and persist it, so that the
# SHAP step explains exactly the model that was evaluated here.
# ------------------------------------------------------------------
best = max(learners, key=lambda k: np.mean(fold_f1[k]))
log(f'\nBest learner by macro-F1: {best}')

scaler_full = StandardScaler().fit(X_s.values)
final_model = make_models()[best]()
Xp = scaler_full.transform(X_s.values)
if best in NEEDS_SAMPLE_WEIGHT:
    final_model.fit(Xp, y_enc, sample_weight=compute_sample_weight('balanced', y_enc))
else:
    final_model.fit(Xp, y_enc)

os.makedirs(os.path.join(C.V2_DIR, 'models'), exist_ok=True)
joblib.dump({'model': final_model, 'scaler': scaler_full, 'label_encoder': le,
             'model_name': best, 'feature_names': C.ALL_FEATURES,
             'train_index': sample_row_ids.tolist()},
            os.path.join(C.V2_DIR, 'models', 'final_model.joblib'))
log(f'Saved models/final_model.joblib ({best})')
log('\nv3 Step 3 complete.')
