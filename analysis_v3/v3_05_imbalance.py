"""
v3 Step 5 - Class imbalance strategies (manuscript Table 5).

Changes relative to step6_imbalance_optimization.py
---------------------------------------------------
1. Naive baselines are included as reference rows.
2. Resampling is expressed as an imbalanced-learn Pipeline so that the
   "no leakage" property is structural rather than incidental.  (v1 did
   already resample inside the fold loop; the pipeline makes that explicit
   and verifiable.)  Standardisation is likewise fitted per training fold,
   which v1 did not do.
3. Per-fold scores are retained; each strategy is tested against the
   class-weighted baseline with the corrected resampled t-test, and
   95% confidence intervals are reported for macro-F1 and Low-class F1.
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
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


log('=' * 70)
log('v3 Step 5 | Class imbalance strategies')
log('=' * 70)

X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
X_s, _, y_s, _ = train_test_split(X_all, y_all, train_size=C.SAMPLE_SIZE,
                                  stratify=y_all, random_state=C.RANDOM_STATE)
X_s = X_s.reset_index(drop=True).values
le = LabelEncoder().fit(C.CLASS_NAMES)          # Low=0, Medium=1, High=2
y_enc = le.transform(y_s)

cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
splits = list(cv.split(X_s, y_enc))
n_train, n_test = len(splits[0][0]), len(splits[0][1])
log(f'{C.SAMPLE_SIZE:,} reviews, {C.N_FOLDS}-fold CV '
    f'({n_train:,} train / {n_test:,} validation)')
log(f'class order: {le.classes_.tolist()}')

RF_KW = dict(n_estimators=300, max_depth=20, min_samples_split=10,
             min_samples_leaf=5, random_state=C.RANDOM_STATE, n_jobs=-1)


def pipe(sampler=None, **rf_kw):
    steps = [('scaler', StandardScaler())]
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('clf', RandomForestClassifier(**{**RF_KW, **rf_kw})))
    return ImbPipeline(steps)


def evaluate(make_estimator, predict_fn=None, name=''):
    fold = {'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'kappa': []}
    per_class = []
    for tr, va in splits:
        est = make_estimator()
        est.fit(X_s[tr], y_enc[tr])
        p = predict_fn(est, X_s[va]) if predict_fn else est.predict(X_s[va])
        fold['accuracy'].append(accuracy_score(y_enc[va], p))
        fold['f1_macro'].append(f1_score(y_enc[va], p, average='macro', zero_division=0))
        fold['f1_weighted'].append(f1_score(y_enc[va], p, average='weighted', zero_division=0))
        fold['kappa'].append(cohen_kappa_score(y_enc[va], p))
        per_class.append(f1_score(y_enc[va], p, average=None, zero_division=0, labels=[0, 1, 2]))
    return fold, per_class


def threshold_predict(est, Xva):
    """Re-weight predicted class probabilities to favour the minority classes."""
    proba = est.predict_proba(Xva).copy()
    proba[:, le.transform(['Low'])[0]] *= 1.5
    proba[:, le.transform(['Medium'])[0]] *= 1.2
    proba[:, le.transform(['High'])[0]] *= 0.7
    return proba.argmax(axis=1)


class TwoStage:
    """High vs non-High, then Low vs Medium within the non-High branch."""

    def __init__(self):
        self.high = le.transform(['High'])[0]
        self.low = le.transform(['Low'])[0]
        self.med = le.transform(['Medium'])[0]

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xp = self.scaler.transform(X)
        self.s1 = RandomForestClassifier(n_estimators=200, max_depth=20,
                                         class_weight='balanced',
                                         random_state=C.RANDOM_STATE, n_jobs=-1)
        self.s1.fit(Xp, (y != self.high).astype(int))
        mask = y != self.high
        self.s2 = RandomForestClassifier(n_estimators=200, max_depth=15,
                                         class_weight='balanced',
                                         random_state=C.RANDOM_STATE, n_jobs=-1)
        self.s2.fit(Xp[mask], (y[mask] == self.med).astype(int))
        return self

    def predict(self, X):
        Xp = self.scaler.transform(X)
        out = np.full(len(Xp), self.high, dtype=int)
        nh = self.s1.predict(Xp) == 1
        if nh.any():
            out[nh] = np.where(self.s2.predict(Xp[nh]) == 1, self.med, self.low)
        return out


STRATEGIES = [
    ('Majority-class baseline', lambda: ImbPipeline(
        [('clf', DummyClassifier(strategy='most_frequent'))]), None),
    ('Stratified-random baseline', lambda: ImbPipeline(
        [('clf', DummyClassifier(strategy='stratified', random_state=C.RANDOM_STATE))]), None),
    ('Baseline (class_weight=balanced)', lambda: pipe(None, class_weight='balanced'), None),
    ('Custom weights (Low 20 : Med 5 : High 1)',
     lambda: pipe(None, class_weight={le.transform(['Low'])[0]: 20,
                                      le.transform(['Medium'])[0]: 5,
                                      le.transform(['High'])[0]: 1}), None),
    ('SMOTE + RF', lambda: pipe(SMOTE(random_state=C.RANDOM_STATE)), None),
    ('ADASYN + RF', lambda: pipe(ADASYN(random_state=C.RANDOM_STATE)), None),
    ('SMOTE + class-weighted RF',
     lambda: pipe(SMOTE(random_state=C.RANDOM_STATE), class_weight='balanced'), None),
    ('EasyEnsemble', lambda: ImbPipeline(
        [('scaler', StandardScaler()),
         ('clf', EasyEnsembleClassifier(n_estimators=20,
                                        random_state=C.RANDOM_STATE, n_jobs=-1))]), None),
    ('Two-stage classification', TwoStage, None),
    ('Threshold tuning', lambda: pipe(None, class_weight='balanced'), threshold_predict),
]

results, fold_f1, fold_low = {}, {}, {}
for name, factory, pred_fn in STRATEGIES:
    log(f'\n{"-"*60}\n{name}')
    t0 = time.time()
    fold, per_class = evaluate(factory, pred_fn, name)
    entry = {m: S.mean_ci(v) for m, v in fold.items()}
    entry['f1_per_class'] = {c: S.mean_ci([r[i] for r in per_class])
                             for i, c in enumerate(le.classes_)}
    entry['fold_scores'] = {m: [float(x) for x in v] for m, v in fold.items()}
    entry['fold_f1_low'] = [float(r[le.transform(['Low'])[0]]) for r in per_class]
    entry['seconds'] = round(time.time() - t0, 1)
    results[name] = entry
    fold_f1[name] = fold['f1_macro']
    fold_low[name] = entry['fold_f1_low']
    log(f'  F1_macro {S.fmt_ci(entry["f1_macro"])}   '
        f'F1_Low {S.fmt_ci(entry["f1_per_class"]["Low"])}   ({entry["seconds"]}s)')

BASE = 'Baseline (class_weight=balanced)'
tests = []
for name in results:
    if name == BASE:
        continue
    t1, p1, d1 = S.corrected_resampled_ttest(fold_f1[name], fold_f1[BASE], n_train, n_test)
    t2, p2, d2 = S.corrected_resampled_ttest(fold_low[name], fold_low[BASE], n_train, n_test)
    tests.append({'strategy': name,
                  'delta_f1_macro': d1, 'p_f1_macro': p1,
                  'delta_f1_low': d2, 'p_f1_low': p2})
for key, pk in [('p_f1_macro', 'p_f1_macro_holm'), ('p_f1_low', 'p_f1_low_holm')]:
    order = np.argsort([r[key] for r in tests])
    m = len(tests)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tests[i][key]))
        tests[i][pk] = run

log(f'\n{"="*70}\nEach strategy vs. the class-weighted baseline (Holm-adjusted):')
for r in sorted(tests, key=lambda r: r['p_f1_macro_holm']):
    log(f'  {r["strategy"]:<42s} dF1m={r["delta_f1_macro"]:+.4f} p={S.fmt_p(r["p_f1_macro_holm"])}'
        f'   dF1_Low={r["delta_f1_low"]:+.4f} p={S.fmt_p(r["p_f1_low_holm"])}')

fried = S.friedman_nemenyi({k: v for k, v in fold_f1.items() if 'baseline' not in k})
log(f'\nFriedman across strategies: chi2={fried.get("chi2", float("nan")):.3f}, '
    f'p={S.fmt_p(fried.get("p", 1))}')

tmap = {t['strategy']: t for t in tests}
base_mean = results[BASE]['f1_macro']['mean']
rows = []
for name, r in results.items():
    t = tmap.get(name)
    rows.append({
        'Strategy': name,
        'Accuracy': f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['sd']:.4f}",
        'F1_macro': f"{r['f1_macro']['mean']:.4f} ± {r['f1_macro']['sd']:.4f}",
        'F1_macro_95CI': f"[{r['f1_macro']['ci_low']:.4f}, {r['f1_macro']['ci_high']:.4f}]",
        'Kappa': f"{r['kappa']['mean']:.4f} ± {r['kappa']['sd']:.4f}",
        'F1_Low': f"{r['f1_per_class']['Low']['mean']:.4f} ± {r['f1_per_class']['Low']['sd']:.4f}",
        'F1_Medium': f"{r['f1_per_class']['Medium']['mean']:.4f}",
        'F1_High': f"{r['f1_per_class']['High']['mean']:.4f}",
        'Delta_F1': 'reference' if name == BASE else f"{r['f1_macro']['mean']-base_mean:+.4f}",
        'p_vs_baseline': '—' if name == BASE else S.fmt_p(t['p_f1_macro_holm']),
    })
df_out = pd.DataFrame(rows)
log('\n' + df_out.to_string(index=False))
df_out.to_csv(os.path.join(C.RESULT_DIR, 'imbalance_results.csv'), index=False)
with open(os.path.join(C.RESULT_DIR, 'imbalance_results.json'), 'w') as f:
    json.dump({'results': results, 'tests_vs_baseline': tests, 'friedman': fried,
               'n_train_per_fold': n_train, 'n_test_per_fold': n_test}, f, indent=2)
log('\nv3 Step 5 complete.')
