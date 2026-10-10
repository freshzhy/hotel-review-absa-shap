"""
Shared configuration for the v3 analysis pipeline.

v3 changes relative to v2 (see CHANGES_v3.md for the full list)
---------------------------------------------------------------
1. Aspect keywords are matched on WORD BOUNDARIES (with optional plural
   suffix; a few intentional stems such as 'welcom' match as prefixes).
   v2 used raw substring matching, so e.g. 'eat' fired inside 'great',
   'view' inside 'review', 'spa' inside 'space' and 'bus' inside
   'business', inflating aspect mention rates.
2. Aspect sentiment is computed at SEGMENT level on the RAW text.  v2
   split sentences on [.!?] AFTER clean_text() had already removed all
   punctuation, so every review was a single "sentence" and each mentioned
   aspect simply received the whole-review VADER score.  v3 segments the
   raw text on punctuation (TripAdvisor) and on sentence-case boundaries
   (Booking.com, whose comment fields carry no punctuation), scores each
   segment with VADER, and averages per aspect.
3. Group B (overall sentiment) is computed with VADER on the raw text,
   so that VADER's capitalisation and punctuation cues are available.
4. Binary aspect-mention indicators are exported alongside the 26
   features (used only in an ablation configuration).

This module centralises everything that must be IDENTICAL across the
Booking.com and TripAdvisor pipelines, so that cross-platform comparisons
are not confounded by differing feature definitions.  In the original
(v1) code, step1 and step7 used *different* aspect dictionaries and
*different* definitions of the overall-sentiment features; both are fixed
here.
"""

import os
import re

import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
BOOKING_CSV = os.path.join(DATASET_DIR, '515K Hotel Reviews Data in Europe', 'Hotel_Reviews.csv')
TRIPADVISOR_CSV = os.path.join(DATASET_DIR, 'Trip Advisor Hotel Reviews', 'tripadvisor_hotel_reviews.csv')

DATA_DIR = os.path.join(V2_DIR, 'data')
RESULT_DIR = os.path.join(V2_DIR, 'results')
FIGURE_DIR = os.path.join(V2_DIR, 'figures')
for _d in (DATA_DIR, RESULT_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
SAMPLE_SIZE = int(os.environ.get('V3_SAMPLE_SIZE', 50_000))   # env override only for smoke tests

# ------------------------------------------------------------------
# Placeholder strings
# ------------------------------------------------------------------
# Booking.com renders an empty comment box as a platform-generated
# placeholder.  Reviewers also type their own "nothing to report" tokens.
# None of these are guest-authored evaluative text, so they are stripped
# before any sentiment scoring.
#
# Deliberately EXCLUDED: 'everything', 'all good', 'no'.  These are terse
# but genuine guest evaluations, not boilerplate, and treating them as
# placeholders would zero out real content for ~4,000 reviews.
PLACEHOLDER_PATTERNS = [
    r'^no\s+negative$',
    r'^no\s+positive$',
    r'^nothing$',
    r'^nothing\s+really$',
    r'^nothing\s+at\s+all$',
    r'^nothing\s+to\s+report$',
    r'^nothing\s+i\s+can\s+think\s+of$',
    r'^none$',
    r'^n\s*/?\s*a$',
    r'^na$',
    r'^nil$',
    r'^-+$',
    r'^\.+$',
    r'^$',
]
_PLACEHOLDER_RE = re.compile('|'.join(PLACEHOLDER_PATTERNS), re.IGNORECASE)


def is_placeholder(text) -> bool:
    """True if `text` is a platform placeholder rather than guest-written content."""
    if text is None:
        return True
    s = str(text).strip()
    return bool(_PLACEHOLDER_RE.match(s))


def strip_placeholder(text) -> str:
    """Replace a placeholder string with the empty string; otherwise return the text."""
    return '' if is_placeholder(text) else str(text)


# ------------------------------------------------------------------
# Aspect lexicon  (single source of truth for BOTH platforms)
# ------------------------------------------------------------------
# Union of the two v1 dictionaries, de-duplicated and sorted.  Using one
# lexicon on both platforms is required for the cross-platform aspect
# mention-rate comparison to be meaningful.
ASPECT_KEYWORDS = {
    'Location': [
        'access', 'accessible', 'airport', 'area', 'bus', 'central',
        'city center', 'city centre', 'close', 'convenient', 'distance',
        'downtown', 'located', 'location', 'metro', 'nearby',
        'neighborhood', 'neighbourhood', 'station', 'street', 'subway',
        'taxi', 'train', 'tram', 'transport', 'walk',
    ],
    'Cleanliness': [
        'bug', 'clean', 'cockroach', 'dirty', 'dust', 'dusty', 'filthy',
        'fresh', 'housekeeping', 'hygiene', 'immaculate', 'maintained',
        'mess', 'mold', 'mould', 'neat', 'odor', 'odour', 'pest',
        'sanitary', 'smell', 'smelly', 'spotless', 'stain', 'tidy',
    ],
    'Service': [
        'accommodating', 'assist', 'attentive', 'check in', 'check-in',
        'checkin', 'check out', 'checkout', 'concierge', 'courteous',
        'efficient', 'friendly', 'helpful', 'manager', 'polite',
        'professional', 'reception', 'receptionist', 'responsive',
        'rude', 'service', 'slow', 'staff', 'welcom',
    ],
    'Room': [
        'air condition', 'air conditioning', 'balcony', 'bathroom', 'bed',
        'comfortable', 'comfy', 'cozy', 'decor', 'furniture', 'heating',
        'mattress', 'minibar', 'modern', 'pillow', 'renovation', 'room',
        'shower', 'small', 'space', 'spacious', 'suite', 'tiny', 'toilet',
        'towel', 'tv', 'view', 'window',
    ],
    'Food_Beverage': [
        'bar', 'breakfast', 'buffet', 'chef', 'coffee', 'cook', 'cuisine',
        'delicious', 'dining', 'dinner', 'drink', 'eat', 'food', 'lunch',
        'meal', 'menu', 'restaurant', 'snack', 'tasty', 'tea', 'variety',
        'wine',
    ],
    'Value': [
        'affordable', 'bargain', 'budget', 'charge', 'cheap', 'cost',
        'deal', 'expensive', 'fee', 'money', 'overpriced', 'paid', 'pay',
        'price', 'rate', 'reasonable', 'value', 'worth',
    ],
    'Facilities': [
        'ac', 'amenities', 'amenity', 'elevator', 'facilities', 'facility',
        'fitness', 'garden', 'gym', 'internet', 'lift', 'lobby', 'lounge',
        'parking', 'pool', 'sauna', 'spa', 'terrace', 'wifi',
    ],
    'Noise': [
        'construction', 'disturb', 'ear plug', 'hear', 'loud', 'noise',
        'noisy', 'party', 'peaceful', 'quiet', 'silent', 'sound',
        'soundproof', 'street noise', 'thin wall', 'traffic',
    ],
}

ASPECTS = list(ASPECT_KEYWORDS)

# ------------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------------
GROUP_A = [f'sent_{a}' for a in ASPECTS]                      # 8 aspect sentiment
GROUP_B = ['overall_sentiment', 'positive_sentiment',
           'negative_sentiment', 'sentiment_gap']              # 4 overall sentiment
GROUP_C = ['review_word_count', 'Review_Total_Negative_Word_Counts',
           'Review_Total_Positive_Word_Counts', 'neg_pos_word_ratio',
           'num_aspects_mentioned']                            # 5 review statistics
# Group D: six numeric metadata features plus three NOMINAL variables.
# v3.1: the nominal variables are one-hot encoded (fixed, schema-defined level
# sets, so no statistics are estimated from the data and no fold leakage can
# occur); v3.0 used integer label codes, which impose a spurious order on
# linear models.
GROUP_D_NUMERIC = ['Nights', 'Is_Mobile', 'is_peak_season',
                   'is_experienced_reviewer', 'review_month',
                   'Total_Number_of_Reviews_Reviewer_Has_Given']
CATEGORICAL = {                         # base feature -> (source column, levels)
    'Trip_Type': ('Trip_Type', ['Business', 'Leisure', 'Unknown']),
    'Traveler_Type': ('Traveler_Type', ['Couple', 'Family', 'Group', 'Solo', 'Unknown']),
    'Hotel_Country': ('Country', ['Austria', 'France', 'Italy', 'Netherlands',
                                  'Spain', 'United Kingdom']),
}
CAT_DUMMIES = {base: [f'{base}={lvl}' for lvl in levels]
               for base, (_, levels) in CATEGORICAL.items()}
GROUP_D = GROUP_D_NUMERIC + [d for ds in CAT_DUMMIES.values() for d in ds]   # 6 + 14 columns

ALL_FEATURES = GROUP_A + GROUP_B + GROUP_C + GROUP_D           # 37 model columns
# The 26 conceptual features (each nominal variable counted once)
BASE_FEATURES = GROUP_A + GROUP_B + GROUP_C + GROUP_D_NUMERIC + list(CATEGORICAL)
BASE_GROUP = {**{f: 'A' for f in GROUP_A}, **{f: 'B' for f in GROUP_B},
              **{f: 'C' for f in GROUP_C}, **{f: 'D' for f in GROUP_D_NUMERIC + list(CATEGORICAL)}}
COLUMN_TO_BASE = {c: c for c in GROUP_A + GROUP_B + GROUP_C + GROUP_D_NUMERIC}
COLUMN_TO_BASE.update({d: base for base, ds in CAT_DUMMIES.items() for d in ds})


def one_hot(df):
    """Append fixed-level one-hot columns for the nominal metadata variables."""
    out = df.copy()
    for base, (src, levels) in CATEGORICAL.items():
        vals = out[src].astype(str)
        for lvl in levels:
            out[f'{base}={lvl}'] = (vals == lvl).astype(int)
    return out


def aggregate_shap(sv, columns):
    """
    Sum SHAP values of one-hot columns back to their base feature (SHAP values
    are additive, so the sum is the attribution of the nominal variable).
    sv: array (n, n_columns[, n_classes]); returns (sv_base, BASE_FEATURES).
    """
    sv = np.asarray(sv)
    idx = {b: [] for b in BASE_FEATURES}
    for j, c in enumerate(columns):
        idx[COLUMN_TO_BASE[c]].append(j)
    parts = [sv[:, idx[b]].sum(axis=1) for b in BASE_FEATURES]
    return np.stack(parts, axis=1), list(BASE_FEATURES)


# Features shared by both platforms (TripAdvisor has no structured metadata)
COMMON_FEATURES = GROUP_A + GROUP_B + ['review_word_count', 'num_aspects_mentioned']  # 14

CLASS_NAMES = ['Low', 'Medium', 'High']          # ordinal order used in all outputs
CLASS_ORDER_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ------------------------------------------------------------------
# Label binning
# ------------------------------------------------------------------
# Booking.com scores are on a 2.5-10 scale in half-point steps.
# pd.cut(bins=[0, 4, 7, 10], include_lowest=True) yields the half-open
# partition  Low = [2.5, 4],  Medium = (4, 7],  High = (7, 10].
# The partition is exhaustive: no score is left unassigned.
BOOKING_BINS = [0, 4, 7, 10]
BOOKING_BINS_ALT_STRICT = [0, 3.5, 7.5, 10]   # sensitivity scheme S1
BOOKING_BINS_ALT_WIDE = [0, 5, 8, 10]         # sensitivity scheme S2


def tripadvisor_class(rating: int) -> str:
    """Map a 1-5 TripAdvisor star rating onto the same three satisfaction levels."""
    if rating <= 2:
        return 'Low'
    if rating == 3:
        return 'Medium'
    return 'High'


# ------------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------------
def clean_text(text) -> str:
    """Lowercase, keep letters and whitespace only, collapse runs of spaces."""
    if not isinstance(text, str) or not text:
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# Standard English inflectional/derivational endings accepted after a keyword
# (v3.1; v3.0 accepted only -s/-es, which missed e.g. 'cleaned', 'cleanliness').
INFLECTION = r'(?:s|es|d|ed|ing|ly|ness|liness|er|ers)?'

# Keywords that are intentional stems and should match as prefixes
STEM_KEYWORDS = {'welcom', 'disturb', 'air condition'}


def _kw_pattern(kw: str) -> str:
    body = r'\s+'.join(re.escape(part) for part in kw.split())
    if kw in STEM_KEYWORDS:
        return r'\b' + body + r'\w*'
    return r'\b' + body + INFLECTION + r'\b'


ASPECT_RE = {a: re.compile('|'.join(_kw_pattern(k) for k in kws))
             for a, kws in ASPECT_KEYWORDS.items()}
MENTION_FEATURES = [f'mention_{a}' for a in ASPECTS]

# Segment boundaries: punctuation, or a lowercase->Capitalised word transition
# (Booking.com comment fields contain no punctuation but keep sentence case).
_SEG_RE = re.compile(r'[.!?;,\n]+|(?<=[a-z])\s+(?=[A-Z][a-z])')


def split_segments(text) -> list:
    """Split raw review text into clause/sentence segments (>= 3 characters)."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [seg.strip() for seg in _SEG_RE.split(text) if seg and len(seg.strip()) >= 3]


def detect_aspects(clean: str) -> dict:
    """Return {aspect: keyword_hit_count} for every aspect mentioned in cleaned text."""
    if not clean:
        return {}
    found = {}
    for aspect, rx in ASPECT_RE.items():
        count = len(rx.findall(clean))
        if count:
            found[aspect] = count
    return found


def aspect_sentiments(segments, vader) -> dict:
    """
    Segment-level aspect sentiment.

    Each raw segment is matched (after cleaning) against every aspect lexicon
    and scored with the VADER compound score of the RAW segment.  An aspect's
    score is the mean over its matched segments; aspects that are never
    mentioned receive 0 (mention indicators are exported separately).
    """
    scores = {a: [] for a in ASPECTS}
    for seg in segments:
        cs = clean_text(seg)
        hits = [a for a, rx in ASPECT_RE.items() if rx.search(cs)]
        if not hits:
            continue
        compound = vader.polarity_scores(seg)['compound']
        for a in hits:
            scores[a].append(compound)
    return {f'sent_{a}': (float(np.mean(v)) if v else 0.0) for a, v in scores.items()}


# ------------------------------------------------------------------
# Parallel text-feature extraction (shared by both platforms)
# ------------------------------------------------------------------
_VADER = None


def _init_worker():
    global _VADER
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()


def text_features(item) -> dict:
    """
    item = (list_of_raw_fields, raw_full_text).
    Returns aspect sentiments, overall sentiment, mention indicators and
    the number of aspects mentioned.
    """
    fields, raw_full = item
    if _VADER is None:
        _init_worker()
    segments = []
    for f in fields:
        segments.extend(split_segments(f))
    rec = aspect_sentiments(segments, _VADER)
    rec.update(overall_sentiment_features(raw_full, _VADER))
    det = detect_aspects(clean_text(raw_full))
    for a in ASPECTS:
        rec[f'mention_{a}'] = int(a in det)
    rec['num_aspects_mentioned'] = len(det)
    rec['n_segments'] = len(segments)
    return rec


def extract_text_features(items, log=print, chunksize=2000):
    """Run text_features over many reviews with a process pool."""
    import multiprocessing as mp
    n = len(items)
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    out = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_proc, initializer=_init_worker) as pool:
        for i, rec in enumerate(pool.imap(text_features, items, chunksize=chunksize)):
            out.append(rec)
            if (i + 1) % 50_000 == 0:
                log(f'  {i+1:,} / {n:,}')
    log(f'  {n:,} / {n:,} done  ({n_proc} worker processes)')
    return out


def overall_sentiment_features(text: str, vader) -> dict:
    """
    Group B, defined IDENTICALLY on both platforms.

    All four measures are derived from VADER applied to the full review text,
    so they carry the same meaning regardless of whether the platform uses one
    free-text field (TripAdvisor) or two (Booking.com).
    """
    if not isinstance(text, str) or not text.strip():
        return {'overall_sentiment': 0.0, 'positive_sentiment': 0.0,
                'negative_sentiment': 0.0, 'sentiment_gap': 0.0}
    s = vader.polarity_scores(text)
    return {
        'overall_sentiment': s['compound'],
        'positive_sentiment': s['pos'],
        'negative_sentiment': s['neg'],
        'sentiment_gap': s['pos'] - s['neg'],
    }


# ------------------------------------------------------------------
# Environment record (reported in Methods)
# ------------------------------------------------------------------
def version_record() -> dict:
    import sys
    import sklearn, pandas, scipy, shap, xgboost, lightgbm, imblearn
    import vaderSentiment
    return {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pandas.__version__,
        'scikit-learn': sklearn.__version__,
        'scipy': scipy.__version__,
        'imbalanced-learn': imblearn.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'shap': shap.__version__,
        'vaderSentiment': getattr(vaderSentiment, '__version__', 'n/a'),
        'random_state': RANDOM_STATE,
    }
