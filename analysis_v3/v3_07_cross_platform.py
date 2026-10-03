"""
v3 Step 7 - Cross-platform comparison (manuscript Tables 6-7).

Changes relative to step7_cross_platform.py
-------------------------------------------
1. Both platforms are processed with the SAME aspect lexicon and the SAME
   Group B definitions (see common.py).  In v1 the two pipelines used
   different aspect dictionaries, and 'positive_sentiment' /
   'negative_sentiment' / 'sentiment_gap' were VADER compound scores of the
   two Booking.com comment fields but VADER pos/neg *proportions* on
   TripAdvisor.  Features sharing a name but not a definition were enough
   on their own to produce apparent "transfer degradation".
2. Confounders are measured and controlled: a descriptive comparison table,
   aspect mention rates normalised per 100 words, and a matched-subsample
   replication in which the two platforms are equalised on review length
   and on class proportions.
3. Naive baselines are reported for every experiment, and all transfer
   estimates carry bootstrap 95% confidence intervals.
"""

import json
import os
import sys

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


log('=' * 70)
log('v3 Step 7 | Cross-platform comparison with confounder control')
log('=' * 70)

le = LabelEncoder().fit(C.CLASS_NAMES)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
X_bk = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))[C.COMMON_FEATURES]
y_bk = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
bk_meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'))

X_ta = pd.read_csv(os.path.join(C.DATA_DIR, 'tripadvisor_X.csv'))[C.COMMON_FEATURES]
ta_meta = pd.read_csv(os.path.join(C.DATA_DIR, 'tripadvisor_y.csv'))
y_ta = ta_meta['satisfaction_class'].values

log(f'Booking {X_bk.shape},  TripAdvisor {X_ta.shape},  '
    f'{len(C.COMMON_FEATURES)} shared features')

# ------------------------------------------------------------------
# 1. Descriptive comparison
# ------------------------------------------------------------------
def describe(name, wc, y, extra=None):
    d = {'platform': name, 'n_reviews': int(len(wc)),
         'words_mean': float(np.mean(wc)), 'words_median': float(np.median(wc)),
         'words_q25': float(np.percentile(wc, 25)),
         'words_q75': float(np.percentile(wc, 75))}
    for c in C.CLASS_NAMES:
        d[f'pct_{c}'] = float((y == c).mean() * 100)
    if extra:
        d.update(extra)
    return d


bk_years = bk_meta['review_year']
desc = [
    describe('Booking.com', bk_meta['review_word_count'].values, y_bk,
             {'rating_scale': '2.5-10 (0.5 steps)',
              'review_format': 'two fields (positive, negative)',
              'year_range': f'{int(bk_years.min())}-{int(bk_years.max())}',
              'geography': 'six European countries',
              'n_countries': int(bk_meta['Country'].nunique()),
              'n_reviewer_nationalities': int(bk_meta['Reviewer_Nationality'].nunique())}),
    describe('TripAdvisor', ta_meta['review_word_count'].values, y_ta,
             {'rating_scale': '1-5 stars',
              'review_format': 'single free-text field',
              'year_range': 'not supplied with the dataset',
              'geography': 'not supplied with the dataset',
              'n_countries': None, 'n_reviewer_nationalities': None}),
]
df_desc = pd.DataFrame(desc)
log('\n--- descriptive comparison ---')
log(df_desc.to_string(index=False))
df_desc.to_csv(os.path.join(C.RESULT_DIR, 'cross_platform_descriptives.csv'), index=False)

# ------------------------------------------------------------------
# 2. Aspect mention rates, raw and length-normalised
# ------------------------------------------------------------------
log('\n--- aspect mention rates (recomputed from text, shared lexicon) ---')


def mention_table(texts, word_counts, label):
    flags = {a: np.zeros(len(texts), dtype=bool) for a in C.ASPECTS}
    for i, t in enumerate(texts):
        det = C.detect_aspects(C.clean_text(t))      # word-boundary matching (v3)
        for a in det:
            flags[a][i] = True
        if (i + 1) % 100_000 == 0:
            log(f'    {label}: {i+1:,}')
    wc = np.asarray(word_counts, dtype=float)
    out = {}
    for a in C.ASPECTS:
        rate = flags[a].mean() * 100
        # mentions per 100 words: proportion of reviews mentioning the aspect,
        # divided by the mean review length in hundreds of words
        out[a] = {'mention_rate_pct': float(rate),
                  'rate_per_100_words': float(rate / (wc.mean() / 100))}
    return out


bk_mentions = mention_table(bk_meta['Full_Review'].fillna('').tolist(),
                            bk_meta['review_word_count'].values, 'booking')
ta_raw = pd.read_csv(C.TRIPADVISOR_CSV)
ta_raw['wc'] = ta_raw['Review'].map(C.clean_text).str.split().str.len().fillna(0)
ta_raw = ta_raw[ta_raw['wc'] >= 3].reset_index(drop=True)
ta_mentions = mention_table(ta_raw['Review'].fillna('').tolist(),
                            ta_raw['wc'].values, 'tripadvisor')

rows = []
for a in C.ASPECTS:
    b, t = bk_mentions[a], ta_mentions[a]
    rows.append({
        'Aspect': a.replace('_', ' & '),
        'Booking_pct': round(b['mention_rate_pct'], 1),
        'TripAdvisor_pct': round(t['mention_rate_pct'], 1),
        'Raw_difference_pp': round(t['mention_rate_pct'] - b['mention_rate_pct'], 1),
        'Booking_per100w': round(b['rate_per_100_words'], 2),
        'TripAdvisor_per100w': round(t['rate_per_100_words'], 2),
        'Normalised_ratio_TA_over_BK': round(
            t['rate_per_100_words'] / b['rate_per_100_words'], 2),
    })
df_mention = pd.DataFrame(rows).sort_values('Raw_difference_pp', ascending=False)
log('\n' + df_mention.to_string(index=False))
df_mention.to_csv(os.path.join(C.RESULT_DIR, 'cross_platform_mention_rates.csv'), index=False)

# ------------------------------------------------------------------
# 3. Classification experiments
# ------------------------------------------------------------------
def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=20,
                                  min_samples_split=10, min_samples_leaf=5,
                                  class_weight='balanced',
                                  random_state=C.RANDOM_STATE, n_jobs=-1)


def within_cv(X, y_enc, tag):
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    fold = {'accuracy': [], 'f1_macro': [], 'kappa': []}
    dummy = {'accuracy': [], 'f1_macro': [], 'kappa': []}
    for tr, va in cv.split(X, y_enc):
        sc = StandardScaler().fit(X[tr])
        m = rf().fit(sc.transform(X[tr]), y_enc[tr])
        p = m.predict(sc.transform(X[va]))
        fold['accuracy'].append(accuracy_score(y_enc[va], p))
        fold['f1_macro'].append(f1_score(y_enc[va], p, average='macro', zero_division=0))
        fold['kappa'].append(cohen_kappa_score(y_enc[va], p))
        dm = DummyClassifier(strategy='most_frequent').fit(X[tr], y_enc[tr])
        dp = dm.predict(X[va])
        dummy['accuracy'].append(accuracy_score(y_enc[va], dp))
        dummy['f1_macro'].append(f1_score(y_enc[va], dp, average='macro', zero_division=0))
        dummy['kappa'].append(cohen_kappa_score(y_enc[va], dp))
    res = {k: S.mean_ci(v) for k, v in fold.items()}
    res['majority_baseline'] = {k: S.mean_ci(v) for k, v in dummy.items()}
    res['fold_scores'] = {k: [float(x) for x in v] for k, v in fold.items()}
    log(f'  {tag:<44s} F1m {S.fmt_ci(res["f1_macro"])}  '
        f'(majority baseline {res["majority_baseline"]["f1_macro"]["mean"]:.4f})')
    return res


def transfer(Xsrc, ysrc, Xtgt, ytgt, tag):
    sc = StandardScaler().fit(Xsrc)
    m = rf().fit(sc.transform(Xsrc), ysrc)
    p = m.predict(sc.transform(Xtgt))
    out = {
        'accuracy': S.bootstrap_metric_ci(ytgt, p, accuracy_score,
                                          random_state=C.RANDOM_STATE),
        'f1_macro': S.bootstrap_metric_ci(
            ytgt, p, lambda a, b: f1_score(a, b, average='macro', zero_division=0),
            random_state=C.RANDOM_STATE),
        'kappa': S.bootstrap_metric_ci(ytgt, p, cohen_kappa_score,
                                       random_state=C.RANDOM_STATE),
        'f1_per_class': {c: float(v) for c, v in zip(
            le.classes_, f1_score(ytgt, p, average=None, zero_division=0, labels=[0, 1, 2]))},
    }
    dm = DummyClassifier(strategy='most_frequent').fit(Xsrc, ysrc)
    dp = dm.predict(Xtgt)
    out['majority_baseline_f1_macro'] = float(f1_score(ytgt, dp, average='macro', zero_division=0))
    log(f'  {tag:<44s} F1m {S.fmt_ci(out["f1_macro"])}  '
        f'(source-majority baseline {out["majority_baseline_f1_macro"]:.4f})')
    return out


SUBSAMPLE = 20_000
y_bk_enc = le.transform(y_bk)
y_ta_enc = le.transform(y_ta)

bk_idx, _ = train_test_split(np.arange(len(X_bk)), train_size=SUBSAMPLE,
                             stratify=y_bk_enc, random_state=C.RANDOM_STATE)
Xb, yb = X_bk.values[bk_idx], y_bk_enc[bk_idx]
Xt, yt = X_ta.values, y_ta_enc

log('\n--- unmatched experiments ---')
results = {
    'booking_within_cv': within_cv(Xb, yb, 'Booking.com within-platform CV'),
    'tripadvisor_within_cv': within_cv(Xt, yt, 'TripAdvisor within-platform CV'),
    'booking_to_tripadvisor': transfer(Xb, yb, Xt, yt, 'Booking -> TripAdvisor'),
    'tripadvisor_to_booking': transfer(Xt, yt, Xb, yb, 'TripAdvisor -> Booking'),
}

# ------------------------------------------------------------------
# 4. Matched subsample: equalise review length and class proportions
# ------------------------------------------------------------------
log('\n--- matched subsample (length-stratified, class-proportion equalised) ---')
rng = np.random.default_rng(C.RANDOM_STATE)

bk_wc = bk_meta['review_word_count'].values
ta_wc = ta_meta['review_word_count'].values

# Length bins from the pooled distribution
edges = np.unique(np.percentile(np.concatenate([bk_wc, ta_wc]),
                                [0, 20, 40, 60, 80, 100]))
bk_bin = np.digitize(bk_wc, edges[1:-1])
ta_bin = np.digitize(ta_wc, edges[1:-1])

# Target class proportions: TripAdvisor's, which is the less skewed of the two
target_props = {c: float((y_ta == c).mean()) for c in C.CLASS_NAMES}
log(f'  length bin edges (words): {[round(e) for e in edges]}')
log(f'  target class proportions: '
    + ', '.join(f'{c} {p*100:.1f}%' for c, p in target_props.items()))

matched = {}
for plat, Xp, yp, binp in [('booking', X_bk.values, y_bk_enc, bk_bin),
                           ('tripadvisor', X_ta.values, y_ta_enc, ta_bin)]:
    take = []
    for b in range(len(edges) - 1):
        for c in C.CLASS_NAMES:
            k = le.transform([c])[0]
            pool = np.where((binp == b) & (yp == k))[0]
            # equal mass per length bin, class proportions set to the target
            want = int(round(target_props[c] * 8000 / (len(edges) - 1)))
            if len(pool) == 0 or want == 0:
                continue
            take.append(rng.choice(pool, size=min(want, len(pool)), replace=False))
    sel = np.concatenate(take)
    matched[plat] = (Xp[sel], yp[sel], binp[sel])
    log(f'  {plat}: n={len(sel):,}  mean length bin={binp[sel].mean():.2f}  '
        + ', '.join(f'{c} {(yp[sel]==le.transform([c])[0]).mean()*100:.1f}%'
                    for c in C.CLASS_NAMES))

Xbm, ybm, _ = matched['booking']
Xtm, ytm, _ = matched['tripadvisor']

results['matched'] = {
    'booking_within_cv': within_cv(Xbm, ybm, 'Booking within-CV (matched)'),
    'tripadvisor_within_cv': within_cv(Xtm, ytm, 'TripAdvisor within-CV (matched)'),
    'booking_to_tripadvisor': transfer(Xbm, ybm, Xtm, ytm, 'Booking -> TripAdvisor (matched)'),
    'tripadvisor_to_booking': transfer(Xtm, ytm, Xbm, ybm, 'TripAdvisor -> Booking (matched)'),
    'n_booking': int(len(ybm)), 'n_tripadvisor': int(len(ytm)),
    'length_bin_edges': [float(e) for e in edges],
    'target_class_proportions': target_props,
}

# ------------------------------------------------------------------
# 4b. Strict matching (v3.1): common length window, fine length bins, and
#     IDENTICAL counts per (length bin x class) on both platforms
# ------------------------------------------------------------------
log('\n--- strictly matched subsample (common length window, equal counts per length bin x class) ---')
LO, HI = 30, 150
in_bk = (bk_wc >= LO) & (bk_wc <= HI)
in_ta = (ta_wc >= LO) & (ta_wc <= HI)
s_edges = np.unique(np.percentile(ta_wc[in_ta], np.linspace(0, 100, 11)))
bk_b = np.clip(np.digitize(bk_wc, s_edges[1:-1]), 0, len(s_edges) - 2)
ta_b = np.clip(np.digitize(ta_wc, s_edges[1:-1]), 0, len(s_edges) - 2)
rng2 = np.random.default_rng(C.RANDOM_STATE + 1)
take_bk, take_ta = [], []
for b in range(len(s_edges) - 1):
    for c in C.CLASS_NAMES:
        k = le.transform([c])[0]
        pb = np.where(in_bk & (bk_b == b) & (y_bk_enc == k))[0]
        pt = np.where(in_ta & (ta_b == b) & (y_ta_enc == k))[0]
        m = min(len(pb), len(pt))
        if m == 0:
            continue
        take_bk.append(rng2.choice(pb, size=m, replace=False))
        take_ta.append(rng2.choice(pt, size=m, replace=False))
sb, st = np.concatenate(take_bk), np.concatenate(take_ta)
Xbs, ybs = X_bk.values[sb], y_bk_enc[sb]
Xts, yts = X_ta.values[st], y_ta_enc[st]
len_stats = {plat: {'n': int(len(w)), 'mean': float(np.mean(w)), 'median': float(np.median(w)),
                    'q25': float(np.percentile(w, 25)), 'q75': float(np.percentile(w, 75))}
             for plat, w in [('booking', bk_wc[sb]), ('tripadvisor', ta_wc[st])]}
from scipy.stats import ks_2samp
ks = ks_2samp(bk_wc[sb], ta_wc[st])
for plat, d in len_stats.items():
    log(f"  {plat}: n={d['n']:,}  words mean {d['mean']:.1f}, median {d['median']:.0f} "
        f"[IQR {d['q25']:.0f}-{d['q75']:.0f}]")
log(f'  class proportions (identical): ' + ', '.join(
    f'{c} {(ybs == le.transform([c])[0]).mean()*100:.1f}%' for c in C.CLASS_NAMES))
log(f'  KS test on length distributions: D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}')
results['strict'] = {
    'booking_within_cv': within_cv(Xbs, ybs, 'Booking within-CV (strict)'),
    'tripadvisor_within_cv': within_cv(Xts, yts, 'TripAdvisor within-CV (strict)'),
    'booking_to_tripadvisor': transfer(Xbs, ybs, Xts, yts, 'Booking -> TripAdvisor (strict)'),
    'tripadvisor_to_booking': transfer(Xts, yts, Xbs, ybs, 'TripAdvisor -> Booking (strict)'),
    'length_window': [LO, HI], 'length_bin_edges': [float(e) for e in s_edges],
    'length_stats': len_stats, 'ks_D': float(ks.statistic), 'ks_p': float(ks.pvalue),
    'class_proportions': {c: float((ybs == le.transform([c])[0]).mean()) for c in C.CLASS_NAMES},
}

# ------------------------------------------------------------------
# 5. Summary table
# ------------------------------------------------------------------
def srow(exp, d, matched_flag):
    if 'fold_scores' in d:                       # within-platform CV
        return {'Experiment': exp, 'Matched': matched_flag,
                'Accuracy': f"{d['accuracy']['mean']:.4f} ± {d['accuracy']['sd']:.4f}",
                'F1_macro': f"{d['f1_macro']['mean']:.4f}",
                'F1_macro_95CI': f"[{d['f1_macro']['ci_low']:.4f}, {d['f1_macro']['ci_high']:.4f}]",
                'Kappa': f"{d['kappa']['mean']:.4f}",
                'Majority_baseline_F1': f"{d['majority_baseline']['f1_macro']['mean']:.4f}"}
    return {'Experiment': exp, 'Matched': matched_flag,
            'Accuracy': f"{d['accuracy']['value']:.4f}",
            'F1_macro': f"{d['f1_macro']['value']:.4f}",
            'F1_macro_95CI': f"[{d['f1_macro']['ci_low']:.4f}, {d['f1_macro']['ci_high']:.4f}]",
            'Kappa': f"{d['kappa']['value']:.4f}",
            'Majority_baseline_F1': f"{d['majority_baseline_f1_macro']:.4f}"}


NAMES = {'booking_within_cv': 'Booking.com (within-platform CV)',
         'tripadvisor_within_cv': 'TripAdvisor (within-platform CV)',
         'booking_to_tripadvisor': 'Booking -> TripAdvisor (transfer)',
         'tripadvisor_to_booking': 'TripAdvisor -> Booking (transfer)'}

summary = [srow(NAMES[k], results[k], 'no') for k in NAMES]
summary += [srow(NAMES[k], results['matched'][k], 'yes') for k in NAMES]
summary += [srow(NAMES[k], results['strict'][k], 'strict') for k in NAMES]
df_sum = pd.DataFrame(summary)
log('\n' + df_sum.to_string(index=False))
df_sum.to_csv(os.path.join(C.RESULT_DIR, 'cross_platform_summary.csv'), index=False)

with open(os.path.join(C.RESULT_DIR, 'cross_platform_results.json'), 'w') as f:
    json.dump({'results': results, 'descriptives': desc,
               'mention_rates': df_mention.to_dict('records'),
               'common_features': C.COMMON_FEATURES}, f, indent=2)
log('\nv3 Step 7 complete.')
