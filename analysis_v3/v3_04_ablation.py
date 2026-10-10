"""
v3 Step 4 - Ablation study and baseline comparison (manuscript Table 3).

Changes relative to step5_ablation_study.py
-------------------------------------------
1. The TF-IDF vectoriser is now fitted inside each training fold.  v1 fitted
   it on the full 50,000-review sample before cross-validation, so document
   frequencies from the validation folds shaped the training representation.
2. The TF-IDF baseline is additionally run with Logistic Regression.  A
   random forest on a 5,000-dimensional sparse bag of words is a weak
   learner for that representation, which flattered the compact feature set.
3. A text-only configuration (A+B+C, no contextual metadata) is added, so
   that the compact representation is compared against TF-IDF on equal
   footing - TF-IDF has no access to reviewer or trip metadata.
4. Standardisation is fitted inside the training fold, per-fold scores are
   retained, and every configuration is tested against the full model with
   the corrected resampled t-test.
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


log('=' * 70)
log('v3 Step 4 | Ablation study')
log('=' * 70)

X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'), usecols=['Full_Review'])

idx_all = np.arange(len(X_all))
idx_s, _, y_s, _ = train_test_split(idx_all, y_all, train_size=C.SAMPLE_SIZE,
                                    stratify=y_all, random_state=C.RANDOM_STATE)
X_s = X_all.iloc[idx_s].reset_index(drop=True)
# v3: binary aspect-mention indicators (used only by one ablation configuration)
M_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_mentions.csv'))
X_s = pd.concat([X_s, M_all.iloc[idx_s].reset_index(drop=True)], axis=1)
text_s = meta['Full_Review'].iloc[idx_s].fillna('').reset_index(drop=True)

le = LabelEncoder().fit(C.CLASS_NAMES)
y_enc = le.transform(y_s)

cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
splits = list(cv.split(X_s, y_enc))
n_train, n_test = len(splits[0][0]), len(splits[0][1])


def rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced',
        random_state=C.RANDOM_STATE, n_jobs=-1)


def logreg():
    return LogisticRegression(max_iter=1000, class_weight='balanced',
                              solver='lbfgs', random_state=C.RANDOM_STATE, n_jobs=-1)


def run_dense(features, factory):
    """CV over a subset of the engineered feature matrix."""
    Xf = X_s[features].values
    fold = {'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'kappa': []}
    per_class = []
    for tr, va in splits:
        scaler = StandardScaler().fit(Xf[tr])
        m = factory()
        m.fit(scaler.transform(Xf[tr]), y_enc[tr])
        p = m.predict(scaler.transform(Xf[va]))
        fold['accuracy'].append(accuracy_score(y_enc[va], p))
        fold['f1_macro'].append(f1_score(y_enc[va], p, average='macro', zero_division=0))
        fold['f1_weighted'].append(f1_score(y_enc[va], p, average='weighted', zero_division=0))
        fold['kappa'].append(cohen_kappa_score(y_enc[va], p))
        per_class.append(f1_score(y_enc[va], p, average=None, zero_division=0, labels=[0, 1, 2]))
    return fold, per_class, len(features)


def run_tfidf(factory, add_aspect=False, max_features=5000):
    """CV over TF-IDF, with the vectoriser fitted on each training fold only."""
    fold = {'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'kappa': []}
    per_class = []
    dims = []
    from scipy.sparse import hstack, csr_matrix
    for tr, va in splits:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=5)
        Xtr = vec.fit_transform(text_s.iloc[tr])
        Xva = vec.transform(text_s.iloc[va])
        if add_aspect:
            sc = StandardScaler().fit(X_s[C.GROUP_A].values[tr])
            Xtr = hstack([Xtr, csr_matrix(sc.transform(X_s[C.GROUP_A].values[tr]))]).tocsr()
            Xva = hstack([Xva, csr_matrix(sc.transform(X_s[C.GROUP_A].values[va]))]).tocsr()
        dims.append(Xtr.shape[1])
        m = factory()
        m.fit(Xtr, y_enc[tr])
        p = m.predict(Xva)
        fold['accuracy'].append(accuracy_score(y_enc[va], p))
        fold['f1_macro'].append(f1_score(y_enc[va], p, average='macro', zero_division=0))
        fold['f1_weighted'].append(f1_score(y_enc[va], p, average='weighted', zero_division=0))
        fold['kappa'].append(cohen_kappa_score(y_enc[va], p))
        per_class.append(f1_score(y_enc[va], p, average=None, zero_division=0, labels=[0, 1, 2]))
    return fold, per_class, int(np.mean(dims))


CONFIGS = [
    ('Full (A+B+C+D)',                 'dense', C.ALL_FEATURES, rf),
    ('Full - A (no aspect sentiment)', 'dense', C.GROUP_B + C.GROUP_C + C.GROUP_D, rf),
    ('Full - B (no overall sentiment)', 'dense', C.GROUP_A + C.GROUP_C + C.GROUP_D, rf),
    ('Full - C (no review statistics)', 'dense', C.GROUP_A + C.GROUP_B + C.GROUP_D, rf),
    ('Full - D (no metadata; text only)', 'dense', C.GROUP_A + C.GROUP_B + C.GROUP_C, rf),
    ('Full + aspect-mention indicators', 'dense', C.ALL_FEATURES + C.MENTION_FEATURES, rf),
    ('Only A (aspect sentiment)',       'dense', C.GROUP_A, rf),
    ('Only A+B (sentiment)',            'dense', C.GROUP_A + C.GROUP_B, rf),
    ('TF-IDF + Random Forest',          'tfidf', False, rf),
    ('TF-IDF + Logistic Regression',    'tfidf', False, logreg),
    ('TF-IDF + aspect sentiment (LR)',  'tfidf', True, logreg),
]

results, fold_f1 = {}, {}
for name, kind, spec, factory in CONFIGS:
    log(f'\n{"-"*60}\n{name}')
    t0 = time.time()
    if kind == 'dense':
        fold, per_class, nfeat = run_dense(spec, factory)
    else:
        fold, per_class, nfeat = run_tfidf(factory, add_aspect=spec)
    entry = {m: S.mean_ci(v) for m, v in fold.items()}
    entry['f1_per_class'] = {c: S.mean_ci([r[i] for r in per_class])
                             for i, c in enumerate(le.classes_)}
    entry['n_features'] = nfeat
    entry['fold_scores'] = {m: [float(x) for x in v] for m, v in fold.items()}
    entry['seconds'] = round(time.time() - t0, 1)
    results[name] = entry
    fold_f1[name] = fold['f1_macro']
    log(f'  {nfeat} features  F1_macro {S.fmt_ci(entry["f1_macro"])}  ({entry["seconds"]}s)')

# Every configuration versus the full model
FULL = 'Full (A+B+C+D)'
tests = []
for name in results:
    if name == FULL:
        continue
    t, p, d = S.corrected_resampled_ttest(fold_f1[name], fold_f1[FULL], n_train, n_test)
    tests.append({'configuration': name, 'delta_f1_vs_full': d, 't': t, 'p_raw': p})
order = np.argsort([r['p_raw'] for r in tests])
m = len(tests)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * tests[i]['p_raw']))
    tests[i]['p_holm'] = run
    tests[i]['significant_05'] = bool(run < 0.05)

log(f'\n{"="*70}\nEach configuration vs. the full model (corrected resampled t-test, Holm):')
for r in sorted(tests, key=lambda r: r['p_holm']):
    flag = '*' if r['significant_05'] else ' '
    log(f'  {flag} {r["configuration"]:<34s} dF1={r["delta_f1_vs_full"]:+.4f}  p={S.fmt_p(r["p_holm"])}')

rows = []
full_mean = results[FULL]['f1_macro']['mean']
tmap = {t['configuration']: t for t in tests}
for name, r in results.items():
    t = tmap.get(name)
    rows.append({
        'Configuration': name,
        'Features': r['n_features'],
        'Accuracy': f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['sd']:.4f}",
        'F1_macro': f"{r['f1_macro']['mean']:.4f} ± {r['f1_macro']['sd']:.4f}",
        'F1_macro_95CI': f"[{r['f1_macro']['ci_low']:.4f}, {r['f1_macro']['ci_high']:.4f}]",
        'Kappa': f"{r['kappa']['mean']:.4f} ± {r['kappa']['sd']:.4f}",
        'Delta_F1': 'reference' if name == FULL else f"{r['f1_macro']['mean']-full_mean:+.4f}",
        'p_vs_full': '—' if name == FULL else S.fmt_p(t['p_holm']),
    })
df_out = pd.DataFrame(rows)
log('\n' + df_out.to_string(index=False))
df_out.to_csv(os.path.join(C.RESULT_DIR, 'ablation_results.csv'), index=False)
with open(os.path.join(C.RESULT_DIR, 'ablation_results.json'), 'w') as f:
    json.dump({'results': results, 'tests_vs_full': tests,
               'n_train_per_fold': n_train, 'n_test_per_fold': n_test}, f, indent=2)
log('\nv3 Step 4 complete.')
