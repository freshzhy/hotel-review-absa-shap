"""
v3 Step 2 - TripAdvisor preprocessing (v3: shared common.extract_text_features).

Uses exactly the same aspect lexicon, sentence splitter and Group B
definitions as the Booking.com pipeline (see common.py), so that the
cross-platform comparison measures platform differences rather than
pipeline differences.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

os.environ.setdefault('NLTK_ALLOW_PROXIED_URLOPEN', '1')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def log(msg=''):
    print(msg, flush=True)


def main():
    log('=' * 70)
    log('v3 Step 2 | TripAdvisor preprocessing')
    log('=' * 70)

    audit = {}

    df = pd.read_csv(C.TRIPADVISOR_CSV)
    audit['raw_rows'] = int(len(df))
    log(f'Raw rows: {len(df):,}  columns: {df.columns.tolist()}')
    audit['rating_counts'] = {int(k): int(v) for k, v in
                              df['Rating'].value_counts().sort_index().items()}

    df['satisfaction_class'] = df['Rating'].map(C.tripadvisor_class)
    df['Clean_Review'] = df['Review'].map(C.clean_text)
    df['review_word_count'] = df['Clean_Review'].str.split().str.len().fillna(0).astype(int)

    keep = df['review_word_count'] >= 3
    audit['dropped_short'] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    audit['rows_after_filter'] = int(len(df))
    log(f'After <3-word filter: {len(df):,}')

    log('Segment-level aspect sentiment, overall sentiment and aspect mentions ...')
    items = [([t], t) for t in df['Review'].fillna('').astype(str).tolist()]
    n = len(items)
    feat = pd.DataFrame(C.extract_text_features(items, log=log, chunksize=500))
    feat['review_word_count'] = df['review_word_count'].values
    mention_counts = {a: int(feat[f'mention_{a}'].sum()) for a in C.ASPECTS}
    audit['segments_per_review'] = {'mean': float(feat['n_segments'].mean()),
                                    'median': float(feat['n_segments'].median())}

    audit['aspect_mention_rate_pct'] = {a: float(c / n * 100)
                                        for a, c in mention_counts.items()}
    log('\nAspect mention rates (word-boundary matching):')
    for a, pct in sorted(audit['aspect_mention_rate_pct'].items(), key=lambda kv: -kv[1]):
        log(f'  {a:<16s} {pct:5.1f}%')

    vc = df['satisfaction_class'].value_counts()
    audit['class_distribution'] = {c: {'n': int(vc.get(c, 0)),
                                       'pct': float(vc.get(c, 0) / n * 100)}
                                   for c in C.CLASS_NAMES}
    log('\nClass distribution:')
    for c in C.CLASS_NAMES:
        d = audit['class_distribution'][c]
        log(f'  {c:<8s} {d["n"]:>7,}  {d["pct"]:5.2f}%')

    audit['review_word_count'] = {
        'mean': float(df['review_word_count'].mean()),
        'median': float(df['review_word_count'].median()),
        'q25': float(df['review_word_count'].quantile(.25)),
        'q75': float(df['review_word_count'].quantile(.75)),
    }
    log(f"\nReview length (words): mean={audit['review_word_count']['mean']:.1f}, "
        f"median={audit['review_word_count']['median']:.0f}, "
        f"IQR=[{audit['review_word_count']['q25']:.0f}, {audit['review_word_count']['q75']:.0f}]")

    X = feat[C.COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X.to_csv(os.path.join(C.DATA_DIR, 'tripadvisor_X.csv'), index=False)
    feat[C.MENTION_FEATURES].to_csv(os.path.join(C.DATA_DIR, 'tripadvisor_mentions.csv'), index=False)
    df[['Review']].to_csv(os.path.join(C.DATA_DIR, 'tripadvisor_text.csv'), index=False)
    df[['satisfaction_class', 'Rating', 'review_word_count']].to_csv(
        os.path.join(C.DATA_DIR, 'tripadvisor_y.csv'), index=False)

    with open(os.path.join(C.RESULT_DIR, 'preprocessing_audit_tripadvisor.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    log(f'\nFeature matrix: {X.shape}')
    log('v3 Step 2 complete.')



if __name__ == '__main__':
    main()
