"""
v3 Step 13 - Language check and equivalence tests.

(a) Language identification (lingua) of the 50,000-review modelling sample and
    of all TripAdvisor reviews.  A review is flagged as non-English when the
    detected language is not English AND the English confidence is below 0.05
    (very short or ambiguous texts such as "Location" are therefore not flagged).
    The random forest is re-evaluated with flagged reviews removed.
(b) Equivalence (TOST) for the key "no significant difference" comparisons,
    using the corrected resampled variance (Nadeau & Bengio) on the paired fold
    scores: the 90% CI of the macro-F1 difference is compared with an
    equivalence margin of +/-0.02 (and reported so readers can apply their own).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import stats_utils as S
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

MARGIN = 0.02


def log(m=''):
    print(m, flush=True)


def tost(a, b, n_train, n_test, margin=MARGIN):
    d = np.asarray(a, float) - np.asarray(b, float); k = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / k + n_test / n_train)); md = d.mean()
    tcrit = stats.t.ppf(0.95, k - 1)
    lo, hi = md - tcrit * se, md + tcrit * se
    p_lower = stats.t.sf((md + margin) / se, k - 1); p_upper = stats.t.cdf((md - margin) / se, k - 1)
    return {'mean_diff': float(md), 'ci90_low': float(lo), 'ci90_high': float(hi),
            'p_tost': float(max(p_lower, p_upper)), 'equivalent_at_margin': bool(lo > -margin and hi < margin)}


def main():
    log('=' * 70); log('v3 Step 13 | Language check and equivalence tests'); log('=' * 70)
    out = {}
    # ---------------- (a) language
    from lingua import LanguageDetectorBuilder, Language
    det = LanguageDetectorBuilder.from_all_languages().build()
    X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
    y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'), usecols=['Full_Review'])
    idx_s, _, y_s, _ = train_test_split(np.arange(len(X_all)), y_all, train_size=C.SAMPLE_SIZE,
                                        stratify=y_all, random_state=C.RANDOM_STATE)
    ta = pd.read_csv(os.path.join(C.DATA_DIR, 'tripadvisor_text.csv'))['Review'].fillna('').astype(str).tolist()
    lang = {}
    for name, texts in [('booking_sample', meta['Full_Review'].iloc[idx_s].fillna('').astype(str).tolist()), ('tripadvisor', ta)]:
        t0 = time.time()
        langs = det.detect_languages_in_parallel_of(texts)
        conf = np.array(det.compute_language_confidence_in_parallel(texts, Language.ENGLISH))
        names = np.array([l.name if l is not None else 'UNKNOWN' for l in langs])
        flag = (names != 'ENGLISH') & (conf < 0.05)
        vc = pd.Series(names[flag]).value_counts().head(8).to_dict()
        lang[name] = {'n': len(texts), 'detected_english_pct': float(np.mean(names == 'ENGLISH') * 100),
                      'flagged_non_english': int(flag.sum()), 'flagged_pct': float(flag.mean() * 100),
                      'flagged_languages': {k: int(v) for k, v in vc.items()},
                      'examples': [texts[i][:120] for i in np.where(flag)[0][:10]]}
        log(f"  {name}: {len(texts):,} texts, detected English {lang[name]['detected_english_pct']:.2f}%, "
            f"flagged non-English {flag.sum():,} ({flag.mean()*100:.2f}%) {vc}  ({time.time()-t0:.0f}s)")
        for ex in lang[name]['examples'][:5]:
            log(f'      e.g. {ex}')
        if name == 'booking_sample':
            flag_bk = flag
    out['language'] = lang

    # RF with flagged reviews removed (same seed / fold scheme)
    le = LabelEncoder().fit(C.CLASS_NAMES)
    keep = ~flag_bk
    Xk = X_all.iloc[idx_s][C.ALL_FEATURES].values[keep]; yk = le.transform(y_s[keep])
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    f1s = []
    for tr, va in cv.split(Xk, yk):
        sc = StandardScaler().fit(Xk[tr])
        m = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=5,
                                   class_weight='balanced', random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(sc.transform(Xk[tr]), yk[tr]); f1s.append(f1_score(yk[va], m.predict(sc.transform(Xk[va])), average='macro'))
    out['rf_english_only'] = {'n': int(keep.sum()), 'f1_macro': S.mean_ci(f1s)}
    log(f"  RF on English-only sample (n = {keep.sum():,}): F1m {S.fmt_ci(out['rf_english_only']['f1_macro'])}")

    # ---------------- (b) equivalence
    R = lambda f: json.load(open(os.path.join(C.RESULT_DIR, f)))
    mc = R('model_comparison.json'); ab = R('ablation_results.json'); im = R('imbalance_results.json')
    ntr, nte = mc['n_train_per_fold'], mc['n_test_per_fold']
    fs = lambda d, k: d['results'][k]['fold_scores']['f1_macro']
    pairs = [('26 features + RF vs TF-IDF + LR', fs(ab, 'Full (A+B+C+D)'), fs(ab, 'TF-IDF + Logistic Regression')),
             ('RF vs XGBoost', fs(mc, 'Random Forest'), fs(mc, 'XGBoost')),
             ('RF vs LightGBM', fs(mc, 'Random Forest'), fs(mc, 'LightGBM')),
             ('RF vs Linear SVM', fs(mc, 'Random Forest'), fs(mc, 'Linear SVM')),
             ('Class weighting vs two-stage', fs(im, 'Baseline (class_weight=balanced)'), fs(im, 'Two-stage classification'))]
    gpath = os.path.join(C.RESULT_DIR, 'grouped_cv_results.json')
    if os.path.exists(gpath):
        g = json.load(open(gpath)); gr = g['results']
        gf = lambda k: gr[k]['fold_scores']['f1_macro']
        pairs += [('[grouped CV] 26 features + RF vs TF-IDF + LR', gf('Random Forest'), gf('TF-IDF + LR')),
                  ('[grouped CV] RF vs XGBoost', gf('Random Forest'), gf('XGBoost'))]
        gtr, gte = g['n_train_mean'], g['n_test_mean']
    eq = []
    for name, a, b in pairs:
        n1, n2 = (gtr, gte) if name.startswith('[grouped') else (ntr, nte)
        r = tost(a, b, n1, n2); r['comparison'] = name; eq.append(r)
        log(f"  {name:<48s} diff {r['mean_diff']:+.4f}, 90% CI [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}], "
            f"TOST p = {r['p_tost']:.3f} -> {'equivalent' if r['equivalent_at_margin'] else 'not shown equivalent'} at ±{MARGIN}")
    out['equivalence'] = {'margin': MARGIN, 'tests': eq}
    pd.DataFrame(eq).to_csv(os.path.join(C.RESULT_DIR, 'equivalence_tests.csv'), index=False)
    json.dump(out, open(os.path.join(C.RESULT_DIR, 'checks_language_equivalence.json'), 'w'), indent=2)
    log('\nv3 Step 13 complete.')


if __name__ == '__main__':
    main()
