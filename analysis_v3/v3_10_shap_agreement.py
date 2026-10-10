"""
v3 Step 10 - Agreement between SHAP importance rankings of two model families.

Step 6 explains the model selected in Step 3 (highest mean macro-F1).  This
step fits the reference class-weighted random forest on the same 50,000-review
modelling sample, computes TreeSHAP on EXACTLY the same 1,000 held-out reviews
(row ids stored by Step 6), and compares the two global importance vectors
(Spearman rho, Kendall tau, top-10 overlap, aspect-feature share).
"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr, kendalltau
import shap


def log(m=''):
    print(m, flush=True)


def main():
    log('=' * 70); log('v3 Step 10 | SHAP ranking agreement across model families'); log('=' * 70)
    X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
    y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
    meta = json.load(open(os.path.join(C.RESULT_DIR, 'shap_metadata.json')))
    expl_ids = np.array(meta['explanation_row_ids'])
    explained = meta['explained_model']
    idx_s, _ = train_test_split(np.arange(len(X_all)), train_size=C.SAMPLE_SIZE,
                                stratify=y_all, random_state=C.RANDOM_STATE)
    assert not (set(idx_s.tolist()) & set(expl_ids.tolist())), 'explanation rows overlap the modelling sample'
    le = LabelEncoder().fit(C.CLASS_NAMES)
    Xs, ys = X_all.values[idx_s], le.transform(y_all[idx_s])
    sc = StandardScaler().fit(Xs)
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=5,
                                class_weight='balanced', random_state=C.RANDOM_STATE, n_jobs=-1).fit(sc.transform(Xs), ys)
    t0 = time.time()
    sv = shap.TreeExplainer(rf).shap_values(sc.transform(X_all.values[expl_ids]), check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv, base = C.aggregate_shap(np.asarray(sv), list(X_all.columns))
    rf_imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=base)
    log(f'RF TreeSHAP on {len(expl_ids)} held-out reviews in {time.time()-t0:.0f}s')
    other = pd.DataFrame(meta['feature_importance']).set_index('feature')['mean_abs_shap'][rf_imp.index]
    rho, p_rho = spearmanr(rf_imp, other); tau, p_tau = kendalltau(rf_imp, other)
    top_rf = set(rf_imp.sort_values(ascending=False).index[:10]); top_o = set(other.sort_values(ascending=False).index[:10])
    share = lambda s: {g: float(s[cols].sum() / s.sum() * 100) for g, cols in
                       [('A', C.GROUP_A), ('B', C.GROUP_B), ('C', C.GROUP_C), ('D', C.GROUP_D_NUMERIC + list(C.CATEGORICAL))]}
    out = {'explained_model': explained, 'n_reviews': int(len(expl_ids)),
           'spearman_rho': float(rho), 'spearman_p': float(p_rho), 'kendall_tau': float(tau), 'kendall_p': float(p_tau),
           'top10_overlap': len(top_rf & top_o),
           'group_share_pct_rf': share(rf_imp), 'group_share_pct_explained': share(other),
           'rf_importance': {k: float(v) for k, v in rf_imp.sort_values(ascending=False).items()}}
    log(f'RF vs {explained}: Spearman rho = {rho:.3f} (p = {p_rho:.2e}), Kendall tau = {tau:.3f}, '
        f'top-10 overlap = {len(top_rf & top_o)}/10')
    log(f'Group shares RF: ' + ', '.join(f'{g} {v:.1f}%' for g, v in out['group_share_pct_rf'].items()))
    log(f'Group shares {explained}: ' + ', '.join(f'{g} {v:.1f}%' for g, v in out['group_share_pct_explained'].items()))
    r = rf_imp.sort_values(ascending=False)
    log('RF top 10: ' + ', '.join(f'{k} ({v/r.sum()*100:.1f}%)' for k, v in r.head(10).items()))
    json.dump(out, open(os.path.join(C.RESULT_DIR, 'shap_agreement_models.json'), 'w'), indent=2)
    log('\nv3 Step 10 complete.')


if __name__ == '__main__':
    main()
