"""
v3 Step 8 - Supplementary analyses newly added in revision.

(a) Threshold sensitivity - does the three-class conclusion survive
    alternative cut points for the rating-to-class mapping?
(b) Learning curve - how much of the 499k corpus is the 50k modelling
    sample actually giving up?
(c) Fairness audit - is classification performance uniform across
    reviewer nationalities, given that Country_encoded is a model input?
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
import warnings

warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=20,
                                  min_samples_split=10, min_samples_leaf=5,
                                  class_weight='balanced',
                                  random_state=C.RANDOM_STATE, n_jobs=-1)


log('=' * 70)
log('v3 Step 8 | Supplementary analyses')
log('=' * 70)

X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
Y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))
meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'),
                   usecols=['Reviewer_Nationality', 'Country'])
le = LabelEncoder().fit(C.CLASS_NAMES)
out = {}


def cv_scores(X, y_enc, n_splits=C.N_FOLDS, return_oof=False):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    fold = {'accuracy': [], 'f1_macro': [], 'kappa': []}
    per_class = []
    oof = np.full(len(y_enc), -1, dtype=int)
    for tr, va in cv.split(X, y_enc):
        sc = StandardScaler().fit(X[tr])
        m = rf().fit(sc.transform(X[tr]), y_enc[tr])
        p = m.predict(sc.transform(X[va]))
        oof[va] = p
        fold['accuracy'].append(accuracy_score(y_enc[va], p))
        fold['f1_macro'].append(f1_score(y_enc[va], p, average='macro', zero_division=0))
        fold['kappa'].append(cohen_kappa_score(y_enc[va], p))
        per_class.append(f1_score(y_enc[va], p, average=None, zero_division=0, labels=[0, 1, 2]))
    res = {k: S.mean_ci(v) for k, v in fold.items()}
    res['f1_per_class'] = {c: float(np.mean([r[i] for r in per_class]))
                           for i, c in enumerate(le.classes_)}
    res['fold_scores'] = {k: [float(x) for x in v] for k, v in fold.items()}
    return (res, oof) if return_oof else res


# ------------------------------------------------------------------
# (a) Threshold sensitivity
# ------------------------------------------------------------------
log('\n--- (a) label-threshold sensitivity ---')
idx_s, _ = train_test_split(np.arange(len(X_all)), train_size=C.SAMPLE_SIZE,
                            stratify=Y_all['satisfaction_class'].values,
                            random_state=C.RANDOM_STATE)
Xs = X_all.values[idx_s]

SCHEMES = [
    ('Primary: Low [2.5,4] / Medium (4,7] / High (7,10]', 'satisfaction_class', C.BOOKING_BINS),
    ('S1 (strict): Low [2.5,3.5] / Medium (3.5,7.5] / High (7.5,10]',
     'satisfaction_class_S1', C.BOOKING_BINS_ALT_STRICT),
    ('S2 (wide): Low [2.5,5] / Medium (5,8] / High (8,10]',
     'satisfaction_class_S2', C.BOOKING_BINS_ALT_WIDE),
]

rows = []
threshold_detail = {}
for label, col, bins in SCHEMES:
    ys = Y_all[col].values[idx_s]
    y_enc = le.transform(ys)
    dist = {c: float((ys == c).mean() * 100) for c in C.CLASS_NAMES}
    t0 = time.time()
    res = cv_scores(Xs, y_enc)
    threshold_detail[label] = {'bins': bins, 'class_distribution_pct': dist, **res}
    rows.append({
        'Binning scheme': label,
        'Low %': f"{dist['Low']:.2f}", 'Medium %': f"{dist['Medium']:.2f}",
        'High %': f"{dist['High']:.2f}",
        'Accuracy': f"{res['accuracy']['mean']:.4f} ± {res['accuracy']['sd']:.4f}",
        'F1_macro': f"{res['f1_macro']['mean']:.4f} ± {res['f1_macro']['sd']:.4f}",
        'F1_macro_95CI': f"[{res['f1_macro']['ci_low']:.4f}, {res['f1_macro']['ci_high']:.4f}]",
        'Kappa': f"{res['kappa']['mean']:.4f}",
        'F1_Low': f"{res['f1_per_class']['Low']:.4f}",
    })
    log(f'  {label}')
    log(f'    dist {dist}  F1m {S.fmt_ci(res["f1_macro"])}  ({time.time()-t0:.0f}s)')

df_thr = pd.DataFrame(rows)
df_thr.to_csv(os.path.join(C.RESULT_DIR, 'threshold_sensitivity.csv'), index=False)
out['threshold_sensitivity'] = threshold_detail

# ------------------------------------------------------------------
# (a2) SHAP ranking stability across the three binning schemes  (v3)
# ------------------------------------------------------------------
log('\n--- (a2) SHAP importance ranking stability across binning schemes ---')
import shap
from scipy.stats import spearmanr, kendalltau

SHAP_STAB_N = int(os.environ.get('SHAP_STAB_N', 500))
holdout = np.setdiff1d(np.arange(len(X_all)), idx_s)
stab_idx, _ = train_test_split(holdout, train_size=SHAP_STAB_N,
                               stratify=Y_all['satisfaction_class'].values[holdout],
                               random_state=C.RANDOM_STATE)
feat_names = list(C.BASE_FEATURES)
importances = {}
for label, col, bins in SCHEMES:
    ys = Y_all[col].values[idx_s]
    y_enc = le.transform(ys)
    sc = StandardScaler().fit(Xs)
    m = rf().fit(sc.transform(Xs), y_enc)
    t0 = time.time()
    sv = shap.TreeExplainer(m).shap_values(sc.transform(X_all.values[stab_idx]),
                                           check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv, _ = C.aggregate_shap(np.asarray(sv), list(X_all.columns))
    imp = np.abs(sv).mean(axis=(0, 2))
    importances[label] = imp
    top = [feat_names[i] for i in np.argsort(-imp)[:10]]
    log(f'  {label.split(":")[0]}: top-5 {top[:5]}  ({time.time()-t0:.0f}s)')

prim = SCHEMES[0][0]
stab_rows = []
for label, _, _ in SCHEMES[1:]:
    rho, p_rho = spearmanr(importances[prim], importances[label])
    tau, p_tau = kendalltau(importances[prim], importances[label])
    top_p = set(np.argsort(-importances[prim])[:10])
    top_s = set(np.argsort(-importances[label])[:10])
    same_top3 = list(np.argsort(-importances[prim])[:3]) == list(np.argsort(-importances[label])[:3])
    stab_rows.append({'comparison': f'Primary vs {label.split(":")[0]}',
                      'spearman_rho': float(rho), 'spearman_p': float(p_rho),
                      'kendall_tau': float(tau), 'top10_overlap': len(top_p & top_s),
                      'identical_top3_order': bool(same_top3)})
    log(f'  Primary vs {label.split(":")[0]}: Spearman rho={rho:.3f} (p={S.fmt_p(p_rho)}), '
        f'Kendall tau={tau:.3f}, top-10 overlap={len(top_p & top_s)}/10, '
        f'identical top-3 order={same_top3}')
pd.DataFrame(stab_rows).to_csv(os.path.join(C.RESULT_DIR, 'shap_stability_thresholds.csv'), index=False)
out['shap_stability_thresholds'] = {
    'n_explained': SHAP_STAB_N, 'comparisons': stab_rows,
    'importance': {lab: {f: float(v) for f, v in zip(feat_names, imp)}
                   for lab, imp in importances.items()}}

# ------------------------------------------------------------------
# (b) Learning curve
# ------------------------------------------------------------------
log('\n--- (b) learning curve ---')
y_full = Y_all['satisfaction_class'].values
y_full_enc = le.transform(y_full)
SIZES = [k for k in [10_000, 25_000, 50_000, 100_000, 200_000] if k < len(X_all)] + [len(X_all)]

lc_rows, lc_detail = [], {}
for n in SIZES:
    if n >= len(X_all):
        sub = np.arange(len(X_all))
        tag = f'{len(X_all):,} (full corpus)'
    else:
        sub, _ = train_test_split(np.arange(len(X_all)), train_size=n,
                                  stratify=y_full_enc, random_state=C.RANDOM_STATE)
        tag = f'{n:,}'
    t0 = time.time()
    res = cv_scores(X_all.values[sub], y_full_enc[sub])
    lc_detail[str(n)] = res
    lc_rows.append({
        'n_reviews': int(len(sub)),
        'Accuracy': f"{res['accuracy']['mean']:.4f} ± {res['accuracy']['sd']:.4f}",
        'F1_macro': f"{res['f1_macro']['mean']:.4f} ± {res['f1_macro']['sd']:.4f}",
        'F1_macro_95CI': f"[{res['f1_macro']['ci_low']:.4f}, {res['f1_macro']['ci_high']:.4f}]",
        'Kappa': f"{res['kappa']['mean']:.4f}",
        'F1_Low': f"{res['f1_per_class']['Low']:.4f}",
        'F1_Medium': f"{res['f1_per_class']['Medium']:.4f}",
        'F1_High': f"{res['f1_per_class']['High']:.4f}",
    })
    log(f'  n={tag:<22s} F1m {S.fmt_ci(res["f1_macro"])}  '
        f'F1_Low {res["f1_per_class"]["Low"]:.4f}  ({time.time()-t0:.0f}s)')

df_lc = pd.DataFrame(lc_rows)
df_lc.to_csv(os.path.join(C.RESULT_DIR, 'learning_curve.csv'), index=False)
out['learning_curve'] = lc_detail

# Learning-curve figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'savefig.dpi': 600, 'savefig.bbox': 'tight'})
ns = [int(r['n_reviews']) for r in lc_rows]
means = [lc_detail[str(k)]['f1_macro']['mean'] if str(k) in lc_detail
         else lc_detail[str(len(X_all))]['f1_macro']['mean'] for k in SIZES]
los = [lc_detail[str(k)]['f1_macro']['ci_low'] for k in SIZES]
his = [lc_detail[str(k)]['f1_macro']['ci_high'] for k in SIZES]
lows = [lc_detail[str(k)]['f1_per_class']['Low'] for k in SIZES]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(ns, means, 'o-', color='#1f77b4', label='Macro-F1')
ax.fill_between(ns, los, his, color='#1f77b4', alpha=0.18)
ax.plot(ns, lows, 's--', color='#d62728', label='Low-class F1')
ax.set_xscale('log')
ax.set_xlabel('Training corpus size (reviews, log scale)')
ax.set_ylabel('F1')
ax.set_title('Learning curve, random forest on the 26-feature representation',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(C.FIGURE_DIR, 'fig_learning_curve.png'))
plt.close()
log('  wrote fig_learning_curve.png')

# ------------------------------------------------------------------
# (c) Fairness audit by reviewer nationality
# ------------------------------------------------------------------
log('\n--- (c) disaggregated performance by reviewer nationality ---')
ys = Y_all['satisfaction_class'].values[idx_s]
y_enc = le.transform(ys)
res_oof, oof = cv_scores(Xs, y_enc, return_oof=True)
nat = meta['Reviewer_Nationality'].fillna('Unknown').str.strip().values[idx_s]

counts = pd.Series(nat).value_counts()
top = counts[counts >= 300].index.tolist()
log(f'  overall out-of-fold macro-F1 {res_oof["f1_macro"]["mean"]:.4f}; '
    f'{len(top)} nationalities with n>=300')

frows = []
for g in top:
    m = nat == g
    frows.append({
        'Reviewer nationality': g,
        'n': int(m.sum()),
        'Pct_High_true': round((ys[m] == 'High').mean() * 100, 1),
        'Accuracy': round(accuracy_score(y_enc[m], oof[m]), 4),
        'F1_macro': round(f1_score(y_enc[m], oof[m], average='macro', zero_division=0), 4),
        'F1_Low': round(f1_score(y_enc[m], oof[m], average=None, zero_division=0,
                                 labels=[0, 1, 2])[le.transform(['Low'])[0]], 4),
        'Pct_predicted_Low': round((oof[m] == le.transform(['Low'])[0]).mean() * 100, 2),
    })
df_fair = pd.DataFrame(frows).sort_values('F1_macro')
log('\n' + df_fair.to_string(index=False))
df_fair.to_csv(os.path.join(C.RESULT_DIR, 'fairness_by_nationality.csv'), index=False)

spread = {
    'n_groups': len(top),
    'f1_macro_min': float(df_fair['F1_macro'].min()),
    'f1_macro_max': float(df_fair['F1_macro'].max()),
    'f1_macro_range': float(df_fair['F1_macro'].max() - df_fair['F1_macro'].min()),
    'f1_macro_sd': float(df_fair['F1_macro'].std(ddof=1)),
    'group_min': df_fair.iloc[0]['Reviewer nationality'],
    'group_max': df_fair.iloc[-1]['Reviewer nationality'],
    'pct_predicted_Low_min': float(df_fair['Pct_predicted_Low'].min()),
    'pct_predicted_Low_max': float(df_fair['Pct_predicted_Low'].max()),
}
log(f'\n  macro-F1 range across nationalities: {spread["f1_macro_range"]:.4f} '
    f'({spread["group_min"]} {spread["f1_macro_min"]:.4f} -> '
    f'{spread["group_max"]} {spread["f1_macro_max"]:.4f})')
log(f'  share predicted Low ranges {spread["pct_predicted_Low_min"]:.2f}% - '
    f'{spread["pct_predicted_Low_max"]:.2f}%')

out['fairness'] = {'overall': res_oof, 'by_group': frows, 'spread': spread}

with open(os.path.join(C.RESULT_DIR, 'supplementary_analyses.json'), 'w') as f:
    json.dump(out, f, indent=2)
log('\nv3 Step 8 complete.')
