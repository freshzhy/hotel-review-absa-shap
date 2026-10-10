"""
v3 Step 1 - Booking.com preprocessing and feature engineering.

v3: text features are computed by common.extract_text_features
(word-boundary aspect matching, segment-level aspect sentiment on the raw
comment fields, overall VADER on the raw text, multiprocessing).  Everything
else is unchanged from v2.  Original v2 notes follow.

Changes relative to the original step1_data_preprocessing.py
------------------------------------------------------------
1. Placeholder handling is widened.  v1 removed only the exact strings
   'No Negative', 'No Positive', 'Nothing', 'nothing', 'N A', 'n a', 'NA';
   'None', 'none', 'n/a', 'nil', '-' and blank-ish variants survived into
   the VADER input.  v2 uses a case-insensitive regex family and reports
   how many comments each pattern removed.
2. The aspect lexicon now comes from common.ASPECT_KEYWORDS, which is the
   same object used by the TripAdvisor pipeline.  v1 used two different
   dictionaries, which confounded the cross-platform mention-rate table.
3. Group B is computed from the full review text on both platforms
   (compound / pos / neg / pos-neg).  v1 computed Booking.com's
   positive_sentiment and negative_sentiment as the *compound* scores of
   the two separate comment fields while TripAdvisor used VADER's pos/neg
   *proportions* - features with the same name but different units and
   ranges, which alone could explain the reported transfer degradation.
4. The label binning is documented explicitly and verified to be
   exhaustive, and the two alternative schemes used in the threshold
   sensitivity analysis are stored alongside the primary labels.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

os.environ.setdefault('NLTK_ALLOW_PROXIED_URLOPEN', '1')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pd.options.mode.chained_assignment = None


def log(msg=''):
    print(msg, flush=True)


def main():
    global audit
    log('=' * 70)
    log('v3 Step 1 | Booking.com preprocessing')
    log('=' * 70)

    audit = {}

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    _nrows = int(os.environ.get('V3_MAX_ROWS', 0)) or None   # smoke tests only
    df = pd.read_csv(C.BOOKING_CSV, nrows=_nrows)
    audit['raw_rows'] = int(len(df))
    audit['raw_hotels'] = int(df['Hotel_Name'].nunique())
    log(f'Raw rows: {len(df):,}   hotels: {df["Hotel_Name"].nunique():,}')

    df['Country'] = (df['Hotel_Address'].fillna('Unknown')
                     .str.strip().str.split().str[-1]
                     .replace({'Kingdom': 'United Kingdom'}))
    audit['country_counts'] = {k: int(v) for k, v in df['Country'].value_counts().items()}

    # ------------------------------------------------------------------
    # 2. Placeholder audit and removal
    # ------------------------------------------------------------------
    log('\n--- placeholder audit ---')
    neg_raw = df['Negative_Review'].fillna('').astype(str).str.strip()
    pos_raw = df['Positive_Review'].fillna('').astype(str).str.strip()

    placeholder_detail = {}
    for pat in C.PLACEHOLDER_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        n_neg = int(neg_raw.map(lambda s: bool(rx.match(s))).sum())
        n_pos = int(pos_raw.map(lambda s: bool(rx.match(s))).sum())
        if n_neg or n_pos:
            placeholder_detail[pat] = {'negative_field': n_neg, 'positive_field': n_pos}
            log(f'  {pat:<28s} neg={n_neg:>7,}  pos={n_pos:>7,}')

    neg_is_ph = neg_raw.map(C.is_placeholder)
    pos_is_ph = pos_raw.map(C.is_placeholder)
    audit['placeholder_patterns'] = placeholder_detail
    audit['negative_field_placeholder'] = int(neg_is_ph.sum())
    audit['positive_field_placeholder'] = int(pos_is_ph.sum())
    audit['negative_field_placeholder_pct'] = float(neg_is_ph.mean() * 100)
    audit['positive_field_placeholder_pct'] = float(pos_is_ph.mean() * 100)

    # How many of these were NOT caught by the v1 rule?
    V1_EXACT = {'No Negative', 'No Positive', 'Nothing', 'nothing', 'N A', 'n a', 'NA'}
    v1_neg = neg_raw.isin(V1_EXACT)
    v1_pos = pos_raw.isin(V1_EXACT)
    audit['placeholders_missed_by_v1'] = {
        'negative_field': int((neg_is_ph & ~v1_neg).sum()),
        'positive_field': int((pos_is_ph & ~v1_pos).sum()),
    }
    log(f'\nPlaceholder comments: negative field {neg_is_ph.sum():,} '
        f'({neg_is_ph.mean()*100:.1f}%), positive field {pos_is_ph.sum():,} '
        f'({pos_is_ph.mean()*100:.1f}%)')
    log(f'Additionally removed relative to v1 rule: '
        f'{audit["placeholders_missed_by_v1"]["negative_field"]:,} negative / '
        f'{audit["placeholders_missed_by_v1"]["positive_field"]:,} positive')

    # The dataset's own word counts already report 0 for placeholder comments;
    # verify this rather than assume it, because neg_pos_word_ratio depends on it.
    audit['word_count_when_placeholder'] = {
        'negative_nonzero': int((df.loc[neg_is_ph, 'Review_Total_Negative_Word_Counts'] > 0).sum()),
        'positive_nonzero': int((df.loc[pos_is_ph, 'Review_Total_Positive_Word_Counts'] > 0).sum()),
    }
    log(f'Placeholder rows whose supplied word count is non-zero: '
        f'{audit["word_count_when_placeholder"]}')

    # Force the supplied word counts to 0 wherever the comment is a placeholder,
    # so that the count features cannot encode platform boilerplate.
    df.loc[neg_is_ph, 'Review_Total_Negative_Word_Counts'] = 0
    df.loc[pos_is_ph, 'Review_Total_Positive_Word_Counts'] = 0

    df['Negative_Review'] = np.where(neg_is_ph, '', neg_raw)
    df['Positive_Review'] = np.where(pos_is_ph, '', pos_raw)
    df['has_negative_text'] = (~neg_is_ph).astype(int)
    df['has_positive_text'] = (~pos_is_ph).astype(int)

    # ------------------------------------------------------------------
    # 3. Text assembly and filtering
    # ------------------------------------------------------------------
    df['Full_Review'] = (df['Negative_Review'] + ' ' + df['Positive_Review']).str.strip()
    df['Clean_Review'] = df['Full_Review'].map(C.clean_text)
    df['review_word_count'] = df['Clean_Review'].str.split().str.len().fillna(0).astype(int)

    n_before = len(df)
    keep = df['review_word_count'] >= 3
    audit['dropped_empty_or_short'] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    audit['rows_after_filter'] = int(len(df))
    audit['retention_pct'] = float(len(df) / audit['raw_rows'] * 100)
    log(f'\nDropped {n_before - len(df):,} reviews with <3 content words; '
        f'{len(df):,} remain ({audit["retention_pct"]:.1f}% of raw)')

    # ------------------------------------------------------------------
    # 4. Tags
    # ------------------------------------------------------------------
    import ast


    def parse_tags(tags_str):
        try:
            tags = [t.strip().lower() for t in ast.literal_eval(tags_str)]
        except Exception:
            return 'Unknown', 'Unknown', 0, 0
        trip = 'Unknown'
        for t in tags:
            if 'leisure' in t:
                trip = 'Leisure'
                break
            if 'business' in t:
                trip = 'Business'
                break
        traveler = 'Unknown'
        for t in tags:
            if 'couple' in t:
                traveler = 'Couple'
            elif 'solo' in t:
                traveler = 'Solo'
            elif 'family' in t:
                traveler = 'Family'
            elif 'group' in t:
                traveler = 'Group'
        nights = 0
        for t in tags:
            m = re.search(r'(\d+)\s*night', t)
            if m:
                nights = int(m.group(1))
        mobile = int(any('mobile' in t for t in tags))
        return trip, traveler, nights, mobile


    log('\nParsing Tags ...')
    parsed = df['Tags'].map(parse_tags)
    df['Trip_Type'] = parsed.str[0]
    df['Traveler_Type'] = parsed.str[1]
    df['Nights'] = parsed.str[2]
    df['Is_Mobile'] = parsed.str[3]

    # ------------------------------------------------------------------
    # 5. Sentiment features
    # ------------------------------------------------------------------
    log('Segment-level aspect sentiment, overall sentiment and aspect mentions ...')
    items = list(zip(zip(df['Negative_Review'].tolist(), df['Positive_Review'].tolist()),
                     df['Full_Review'].tolist()))
    feat_rows = C.extract_text_features(items, log=log)
    sent_df = pd.DataFrame(feat_rows, index=df.index)
    df = pd.concat([df, sent_df], axis=1)

    mention_counts = {a: int(df[f'mention_{a}'].sum()) for a in C.ASPECTS}
    audit['aspect_mention_rate_pct'] = {a: float(c / len(df) * 100)
                                        for a, c in mention_counts.items()}
    audit['segments_per_review'] = {'mean': float(df['n_segments'].mean()),
                                    'median': float(df['n_segments'].median())}
    log('Aspect mention rates (word-boundary matching):')
    for a, pct in sorted(audit['aspect_mention_rate_pct'].items(),
                         key=lambda kv: -kv[1]):
        log(f'  {a:<16s} {pct:5.1f}%')
    log(f"Segments per review: mean {audit['segments_per_review']['mean']:.2f}, "
        f"median {audit['segments_per_review']['median']:.0f}")

    # ------------------------------------------------------------------
    # 6. Remaining engineered features
    # ------------------------------------------------------------------
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], format='mixed')
    df['review_month'] = df['Review_Date'].dt.month
    df['review_year'] = df['Review_Date'].dt.year
    df['is_peak_season'] = df['review_month'].isin([6, 7, 8, 9]).astype(int)
    df['is_experienced_reviewer'] = (
        df['Total_Number_of_Reviews_Reviewer_Has_Given'] >= 5).astype(int)

    # neg_pos_word_ratio: negative words per positive word, with a +1 guard on the
    # denominator.  Kept identical to the original definition so the feature is
    # comparable with the published v1 results; the manuscript text is corrected
    # to describe it accurately.
    df['neg_pos_word_ratio'] = (df['Review_Total_Negative_Word_Counts'] /
                                df['Review_Total_Positive_Word_Counts'].clip(lower=1))

    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in ['Trip_Type', 'Traveler_Type', 'Country']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = {str(k): int(v) for k, v in zip(le.classes_,
                                                        le.transform(le.classes_))}
    audit['label_encoders'] = encoders   # kept for reference only; models use one-hot columns
    df = C.one_hot(df)                    # v3.1: fixed-level one-hot encoding of nominal metadata

    # ------------------------------------------------------------------
    # 7. Labels (primary + sensitivity schemes)
    # ------------------------------------------------------------------
    score = df['Reviewer_Score']
    audit['score_min'] = float(score.min())
    audit['score_max'] = float(score.max())


    def bin_scores(s, bins):
        return pd.cut(s, bins=bins, labels=C.CLASS_NAMES, include_lowest=True)


    df['satisfaction_class'] = bin_scores(score, C.BOOKING_BINS)
    df['satisfaction_class_S1'] = bin_scores(score, C.BOOKING_BINS_ALT_STRICT)
    df['satisfaction_class_S2'] = bin_scores(score, C.BOOKING_BINS_ALT_WIDE)

    assert df['satisfaction_class'].isna().sum() == 0, 'binning left scores unassigned'
    audit['binning'] = {
        'primary_bins': C.BOOKING_BINS,
        'primary_intervals': 'Low=[2.5,4], Medium=(4,7], High=(7,10]',
        'unassigned_scores': 0,
        'S1_bins': C.BOOKING_BINS_ALT_STRICT,
        'S2_bins': C.BOOKING_BINS_ALT_WIDE,
    }
    for key, col in [('primary', 'satisfaction_class'),
                     ('S1', 'satisfaction_class_S1'),
                     ('S2', 'satisfaction_class_S2')]:
        vc = df[col].value_counts()
        audit['binning'][f'{key}_distribution'] = {
            c: {'n': int(vc.get(c, 0)), 'pct': float(vc.get(c, 0) / len(df) * 100)}
            for c in C.CLASS_NAMES}

    log('\nClass distribution (primary binning Low=[2.5,4], Medium=(4,7], High=(7,10]):')
    for c in C.CLASS_NAMES:
        d = audit['binning']['primary_distribution'][c]
        log(f'  {c:<8s} {d["n"]:>8,}  {d["pct"]:5.2f}%')

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    X = df[C.ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    assert X.shape[1] == len(C.ALL_FEATURES) == 37, X.shape

    X.to_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'), index=False)
    df[['satisfaction_class', 'satisfaction_class_S1', 'satisfaction_class_S2']] \
        .to_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'), index=False)
    df[C.MENTION_FEATURES].to_csv(os.path.join(C.DATA_DIR, 'booking_mentions.csv'), index=False)
    df[['Reviewer_Score', 'Country', 'Hotel_Name', 'Reviewer_Nationality', 'review_year',
        'has_negative_text', 'has_positive_text', 'review_word_count',
        'Full_Review', 'Negative_Review', 'Positive_Review']].to_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'), index=False)

    audit['n_features'] = int(X.shape[1])
    audit['feature_names'] = C.ALL_FEATURES
    audit['versions'] = C.version_record()

    with open(os.path.join(C.RESULT_DIR, 'preprocessing_audit_booking.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    log(f'\nFeature matrix: {X.shape}')
    log(f'Wrote {C.DATA_DIR}/booking_X.csv, booking_y.csv, booking_meta.csv')
    log('v3 Step 1 complete.')



if __name__ == '__main__':
    main()
