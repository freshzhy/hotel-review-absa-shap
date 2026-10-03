"""
v3 Step 6 - SHAP explainability (manuscript Figures 3-4, Table 4).

Changes relative to step3_shap_analysis.py
------------------------------------------
1. SHAP values are computed for the model that was actually evaluated
   (the 300-tree, max_depth=20 random forest persisted by v2 step 3).
   v1 trained a separate, smaller surrogate forest (100 trees,
   max_depth=12) purely for tractability and reported its attributions as
   explanations of the evaluated model.
2. Explanations are computed on reviews held out of the 50,000-review
   modelling sample, so the global importance ranking is not read off the
   model's own training data.
3. The "SHAP interaction analysis" of v1 reported absolute Pearson
   correlations between per-class SHAP value vectors, which are neither
   SHAP interaction values nor feature correlations.  Both quantities are
   now computed and reported under their correct names.
4. Per-class beeswarm figures are written to distinct filenames; in v1 the
   output path was a plain string containing '{cls_name}', so all three
   class panels overwrote a single file.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 600,
                     'savefig.bbox': 'tight'})

EXPLAIN_N = int(os.environ.get('SHAP_N', 1000))


def log(m=''):
    print(m, flush=True)


log('=' * 70)
log('v3 Step 6 | SHAP on the evaluated model')
log('=' * 70)

bundle = joblib.load(os.path.join(C.V2_DIR, 'models', 'final_model.joblib'))
model, scaler, le = bundle['model'], bundle['scaler'], bundle['label_encoder']
feature_names = bundle['feature_names']
model_name = bundle['model_name']
train_index = set(bundle['train_index'])

log(f'Explaining: {model_name}')
log(f'  params: {model.get_params()}')

X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values

holdout = np.setdiff1d(np.arange(len(X_all)), np.fromiter(train_index, dtype=int))
assert len(holdout) == len(X_all) - len(train_index), 'train_index must hold original row ids'
log(f'Held-out pool: {len(holdout):,} reviews')

from sklearn.model_selection import train_test_split

expl_idx, _ = train_test_split(holdout, train_size=EXPLAIN_N,
                               stratify=y_all[holdout], random_state=C.RANDOM_STATE)
assert not (set(np.asarray(expl_idx).tolist()) & train_index), 'explanation sample overlaps training data'
X_expl = X_all.iloc[expl_idx].reset_index(drop=True)
y_expl = y_all[expl_idx]
X_expl_s = scaler.transform(X_expl.values)
log(f'Explanation sample: {EXPLAIN_N} held-out reviews  '
    f'{dict(zip(*np.unique(y_expl, return_counts=True)))}')

log('\nComputing TreeSHAP (this is the slow step) ...')
t0 = time.time()
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_expl_s, check_additivity=False)
log(f'  done in {time.time()-t0:.0f}s')

# Normalise to array of shape (n_samples, n_features, n_classes)
if isinstance(sv, list):
    sv = np.stack(sv, axis=-1)
sv = np.asarray(sv)
log(f'  SHAP array shape {sv.shape}')

# v3.1: sum the SHAP values of one-hot columns back to their nominal variable
col_names = list(feature_names)
sv, feature_names = C.aggregate_shap(sv, col_names)
cat_bases = set(C.CATEGORICAL)
X_expl_s_cols = pd.DataFrame(X_expl_s, columns=col_names)
# display matrix: standardised values for numeric features, NaN (grey) for nominal ones
X_expl_s = np.column_stack([X_expl_s_cols[f].values if f not in cat_bases else np.full(len(X_expl_s_cols), np.nan)
                            for f in feature_names])
X_expl = pd.DataFrame({f: (X_expl[f].values if f not in cat_bases else np.full(len(X_expl), np.nan))
                       for f in feature_names})
log(f'  aggregated SHAP array shape {sv.shape} (one-hot columns summed per nominal variable)')

class_names = le.classes_.tolist()                    # ['High','Low','Medium'] order of encoder
ordered = C.CLASS_NAMES                               # Low, Medium, High for display
cls_pos = {c: class_names.index(c) for c in ordered}

# ------------------------------------------------------------------
# Global importance
# ------------------------------------------------------------------
mean_abs = np.abs(sv).mean(axis=(0, 2))
imp = (pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
       .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True))
imp['rank'] = np.arange(1, len(imp) + 1)
imp['share_pct'] = imp['mean_abs_shap'] / imp['mean_abs_shap'].sum() * 100
imp.to_csv(os.path.join(C.RESULT_DIR, 'feature_importance_shap.csv'), index=False)

log('\nTop 10 features by mean |SHAP|:')
for _, r in imp.head(10).iterrows():
    log(f'  {int(r["rank"]):>2}. {r["feature"]:<40s} {r["mean_abs_shap"]:.4f}  '
        f'({r["share_pct"]:.1f}%)')

aspect_share = imp[imp.feature.isin(C.GROUP_A)]['mean_abs_shap'].sum()
imp['group'] = imp['feature'].map(C.BASE_GROUP)
imp.to_csv(os.path.join(C.RESULT_DIR, 'feature_importance_shap.csv'), index=False)
total_share = imp['mean_abs_shap'].sum()
log(f'\nAspect sentiment features account for '
    f'{aspect_share/total_share*100:.1f}% of total SHAP mass')

# Figure: global importance bar
fig, ax = plt.subplots(figsize=(8, 8))
d = imp.sort_values('mean_abs_shap')
colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(d)))
ax.barh(range(len(d)), d['mean_abs_shap'], color=colors, edgecolor='gray', linewidth=0.4)
ax.set_yticks(range(len(d)))
ax.set_yticklabels(d['feature'], fontsize=9)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title(f'Global SHAP feature importance ({model_name})', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(C.FIGURE_DIR, 'fig_shap_bar.png'))
plt.close()
log('  wrote fig_shap_bar.png')

# Figure: per-class beeswarm (distinct filenames)
for cls in ordered:
    k = cls_pos[cls]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv[:, :, k], X_expl_s, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f'SHAP summary - {cls} satisfaction class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(C.FIGURE_DIR, f'fig_shap_beeswarm_{cls}.png'))
    plt.close()
    log(f'  wrote fig_shap_beeswarm_{cls}.png')

# Figure: per-class importance panel
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharex=False)
for ax, cls in zip(axes, ordered):
    k = cls_pos[cls]
    v = np.abs(sv[:, :, k]).mean(axis=0)
    o = np.argsort(v)
    ax.barh(range(len(v)), v[o], color=plt.cm.Set2(ordered.index(cls)),
            edgecolor='gray', linewidth=0.4)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([feature_names[i] for i in o], fontsize=8)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(cls, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
plt.suptitle(f'Class-specific SHAP importance ({model_name})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(C.FIGURE_DIR, 'fig_shap_by_class.png'))
plt.close()
log('  wrote fig_shap_by_class.png')

# Figure: dependence plots for the top six features (High class)
top6 = [f for f in imp['feature'] if f not in cat_bases][:6]
k_high = cls_pos['High']
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for i, f in enumerate(top6):
    shap.dependence_plot(feature_names.index(f), sv[:, :, k_high], X_expl_s,
                         feature_names=feature_names, ax=axes[i // 3, i % 3],
                         show=False)
    axes[i // 3, i % 3].set_title(f, fontsize=10, fontweight='bold')
plt.suptitle('SHAP dependence plots, High satisfaction class', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(C.FIGURE_DIR, 'fig_shap_dependence.png'))
plt.close()
log('  wrote fig_shap_dependence.png')

# ------------------------------------------------------------------
# Feature association patterns (NOT SHAP interaction values)
# ------------------------------------------------------------------
top10 = [f for f in imp['feature'] if f not in cat_bases][:10]
raw_corr, shap_corr = [], []
for i in range(len(top10)):
    for j in range(i + 1, len(top10)):
        fa, fb = top10[i], top10[j]
        ia, ib = feature_names.index(fa), feature_names.index(fb)
        r_raw = float(np.corrcoef(X_expl[fa], X_expl[fb])[0, 1])
        r_shap = float(np.corrcoef(sv[:, ia, k_high], sv[:, ib, k_high])[0, 1])
        raw_corr.append({'feature_a': fa, 'feature_b': fb, 'pearson_r': r_raw})
        shap_corr.append({'feature_a': fa, 'feature_b': fb,
                          'pearson_r_of_shap_values': r_shap})
raw_corr.sort(key=lambda d: -abs(d['pearson_r']))
shap_corr.sort(key=lambda d: -abs(d['pearson_r_of_shap_values']))

log('\nTop 5 raw-feature Pearson correlations among the top-10 features:')
for d in raw_corr[:5]:
    log(f'  {d["feature_a"]} x {d["feature_b"]}: r = {d["pearson_r"]:+.3f}')
log('\nTop 5 correlations between per-class SHAP value vectors (High class):')
for d in shap_corr[:5]:
    log(f'  {d["feature_a"]} x {d["feature_b"]}: r = {d["pearson_r_of_shap_values"]:+.3f}')

pd.DataFrame(raw_corr).to_csv(
    os.path.join(C.RESULT_DIR, 'feature_pearson_correlations.csv'), index=False)
pd.DataFrame(shap_corr).to_csv(
    os.path.join(C.RESULT_DIR, 'shap_value_correlations.csv'), index=False)

# ------------------------------------------------------------------
meta = {
    'explained_model': model_name,
    'explained_model_params': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                               for k, v in model.get_params().items()},
    'explanation_sample_size': int(EXPLAIN_N),
    'explanation_sample_source': 'stratified random draw from reviews held out of the modelling sample',
    'explanation_row_ids': [int(i) for i in expl_idx],
    'class_order_in_shap_array': class_names,
    'feature_importance': imp.to_dict('records'),
    'aspect_shap_share_pct': float(aspect_share / total_share * 100),
    'top_raw_feature_correlations': raw_corr[:10],
    'top_shap_value_correlations': shap_corr[:10],
    'versions': C.version_record(),
}
with open(os.path.join(C.RESULT_DIR, 'shap_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)
log('\nv3 Step 6 complete.')
