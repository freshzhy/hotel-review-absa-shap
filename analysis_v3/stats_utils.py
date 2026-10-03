"""
Statistical comparison utilities for cross-validated classifier scores.

Implements the corrected resampled t-test of Nadeau & Bengio (2003), which
accounts for the dependence between cross-validation folds that the naive
paired t-test ignores, plus Friedman / Nemenyi for the omnibus comparison
of several models, and bootstrap confidence intervals for single-split
(transfer) estimates.
"""

import itertools

import numpy as np
from scipy import stats


def mean_ci(scores, alpha=0.05):
    """Mean, SD and Student-t confidence interval of a set of fold scores."""
    a = np.asarray(scores, dtype=float)
    n = len(a)
    m, sd = a.mean(), a.std(ddof=1)
    if n < 2 or sd == 0:
        return {'mean': float(m), 'sd': float(sd), 'ci_low': float(m), 'ci_high': float(m), 'n': n}
    half = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    return {'mean': float(m), 'sd': float(sd),
            'ci_low': float(m - half), 'ci_high': float(m + half), 'n': n}


def corrected_resampled_ttest(a, b, n_train, n_test):
    """
    Corrected resampled paired t-test (Nadeau & Bengio 2003).

    The variance of the fold-score differences is inflated by
    (1/k + n_test/n_train) instead of 1/k, correcting for the overlap
    between the training sets of different folds.

    Returns (t statistic, two-sided p value, mean difference).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    k = len(d)
    md = d.mean()
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0, float(md)
    corrected_var = var * (1.0 / k + n_test / n_train)
    t = md / np.sqrt(corrected_var)
    p = 2 * stats.t.sf(abs(t), k - 1)
    return float(t), float(p), float(md)


def pairwise_comparisons(fold_scores: dict, n_train: int, n_test: int,
                         correction='holm'):
    """
    All pairwise corrected resampled t-tests between models.

    `fold_scores` maps model name -> list of per-fold scores.
    Returns a list of dicts, multiplicity-adjusted with Holm-Bonferroni.
    """
    names = list(fold_scores)
    rows = []
    for a, b in itertools.combinations(names, 2):
        t, p, md = corrected_resampled_ttest(fold_scores[a], fold_scores[b],
                                             n_train, n_test)
        rows.append({'model_a': a, 'model_b': b, 'mean_diff': md,
                     't': t, 'p_raw': p})
    if correction == 'holm' and rows:
        order = np.argsort([r['p_raw'] for r in rows])
        m = len(rows)
        running = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (m - rank) * rows[idx]['p_raw'])
            running = max(running, adj)          # enforce monotonicity
            rows[idx]['p_holm'] = running
    else:
        for r in rows:
            r['p_holm'] = r['p_raw']
    for r in rows:
        r['significant_05'] = bool(r['p_holm'] < 0.05)
    return rows


def friedman_nemenyi(fold_scores: dict):
    """
    Friedman omnibus test across models over folds, plus the Nemenyi
    critical difference at alpha = 0.05.

    Returns a dict with the chi-square statistic, p value, mean ranks and CD.
    """
    names = list(fold_scores)
    mat = np.array([fold_scores[n] for n in names], dtype=float)  # models x folds
    k, n = mat.shape
    if k < 3:
        return {'applicable': False,
                'reason': 'Friedman requires at least three models'}
    chi2, p = stats.friedmanchisquare(*mat)
    # rank within each fold: 1 = best (highest score)
    ranks = np.zeros_like(mat)
    for j in range(n):
        ranks[:, j] = stats.rankdata(-mat[:, j])
    mean_ranks = ranks.mean(axis=1)
    # Nemenyi critical difference; q_alpha = studentised range / sqrt(2)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k)
    cd = (q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))) if q_alpha else None
    return {'applicable': True, 'chi2': float(chi2), 'p': float(p),
            'mean_ranks': {nm: float(r) for nm, r in zip(names, mean_ranks)},
            'critical_difference': float(cd) if cd else None,
            'k_models': k, 'n_folds': n}


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=2000,
                        alpha=0.05, random_state=42):
    """
    Percentile bootstrap CI for a metric computed on a single held-out set.
    Used for the cross-platform transfer results, which have no folds.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'value': point, 'ci_low': float(lo), 'ci_high': float(hi),
            'n_boot': n_boot}


def fmt_ci(d, digits=4):
    """'0.5537 [0.5412, 0.5662]'"""
    return f"{d['mean' if 'mean' in d else 'value']:.{digits}f} " \
           f"[{d['ci_low']:.{digits}f}, {d['ci_high']:.{digits}f}]"


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'
