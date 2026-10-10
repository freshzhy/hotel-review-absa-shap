"""
v3 Step 12 - External validation of the seed-word ABSA against human annotations.

Uses HRAST (Hotel Reviews: Aspects, Sentiments and Topics; Andreou et al. 2023),
23,113 English Booking.com review sentences from 42 hotels, each annotated by
humans for sentence sentiment (positive / negative / neutral) and for hotel
topics.  HRAST topics are mapped onto the study's eight aspects:

  Location      <- Location
  Cleanliness   <- Clean
  Service       <- Staff, Reception-checkin
  Room          <- Room, Bed, Bathroom/Shower (toilet), View (Balcony), Comfort
  Food_Beverage <- Breakfast, Restaurant (dinner), Bar
  Value         <- Value for money
  Facilities    <- Facilities/Amenities, Pool, Parking, Lift, Wi-Fi
  Noise         <- Noise

Reported:
  (a) aspect detection precision / recall / F1 per aspect, micro and macro,
      for the matcher used in the study (word boundaries + inflectional endings),
      a plural-only variant and naive substring matching (the v2 behaviour);
  (b) VADER sentence polarity (compound >= 0.05 positive, <= -0.05 negative)
      versus the human sentiment label;
  (c) for sentences in which an aspect is detected, agreement between the sign
      of the aspect score and the human polarity (positive/negative sentences).
"""
import io, json, os, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MAP = {
    'Location': ['Location'],
    'Cleanliness': ['Clean'],
    'Service': ['Staff', 'Reception-checkin'],
    'Room': ['Room', 'Bed', 'Bathroom/Shower (toilet)', 'View (Balcony)', 'Comfort'],
    'Food_Beverage': ['Breakfast', 'Restaurant (dinner)', 'Bar'],
    'Value': ['Value for money'],
    'Facilities': ['Facilities/Amenities', 'Pool', 'Parking', 'Lift', 'Wi-Fi'],
    'Noise': ['Noise'],
}


def log(m=''):
    print(m, flush=True)


def load_hrast():
    d = C.DATASET_DIR
    for p in [os.path.join(d, 'Hotel Reviews- Aspects, Sentiments and Topics', 'HRAST.csv'), os.path.join(d, 'HRAST.csv')]:
        if os.path.exists(p):
            return pd.read_csv(p)
    for f in os.listdir(d):
        if f.startswith('Hotel Reviews- Aspects') and f.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(d, f)) as z:
                return pd.read_csv(io.BytesIO(z.read('HRAST.csv')))
    raise FileNotFoundError('HRAST not found in dataset/ (download "Hotel Reviews: Aspects, Sentiments and Topics")')


def plural_only_re():
    import re
    def pat(kw):
        body = r'\s+'.join(re.escape(p) for p in kw.split())
        return r'\b' + body + (r'\w*' if kw in C.STEM_KEYWORDS else r'(?:s|es)?\b')
    return {a: re.compile('|'.join(pat(k) for k in kws)) for a, kws in C.ASPECT_KEYWORDS.items()}


def substring_detect(clean):
    return {a for a, kws in C.ASPECT_KEYWORDS.items() if any(kw in clean for kw in kws)}


def main():
    log('=' * 70); log('v3 Step 12 | ABSA validation against human annotations (HRAST)'); log('=' * 70)
    df = load_hrast()
    topic_cols = [c for cols in MAP.values() for c in cols]
    for c in topic_cols + ['positive', 'negative', 'neutral']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int).clip(0, 1)
    df['review'] = df['review'].fillna('').astype(str)
    df = df[df['review'].str.strip().str.len() > 0].reset_index(drop=True)
    log(f'Sentences: {len(df):,}')
    gold = pd.DataFrame({a: (df[cols].sum(axis=1) > 0).astype(int) for a, cols in MAP.items()})
    clean = df['review'].map(C.clean_text)
    pred_wb = pd.DataFrame({a: clean.map(lambda s: int(bool(C.ASPECT_RE[a].search(s)))) for a in C.ASPECTS})
    PRE = plural_only_re()
    pred_pl = pd.DataFrame({a: clean.map(lambda s: int(bool(PRE[a].search(s)))) for a in C.ASPECTS})
    sub = clean.map(substring_detect)
    pred_sub = pd.DataFrame({a: sub.map(lambda s: int(a in s)) for a in C.ASPECTS})

    out = {'n_sentences': int(len(df)), 'mapping': MAP, 'detection': {}}
    rows = []
    for name, pred in [('word-boundary + inflections (study)', pred_wb), ('word-boundary, plural only', pred_pl),
                       ('substring (naive)', pred_sub)]:
        per = {}
        for a in C.ASPECTS:
            p, r, f, _ = precision_recall_fscore_support(gold[a], pred[a], average='binary', zero_division=0)
            per[a] = {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(gold[a].sum()),
                      'predicted': int(pred[a].sum())}
        mp, mr, mf, _ = precision_recall_fscore_support(gold.values, pred.values, average='micro', zero_division=0)
        Mp, Mr, Mf, _ = precision_recall_fscore_support(gold.values, pred.values, average='macro', zero_division=0)
        out['detection'][name] = {'per_aspect': per, 'micro': {'precision': float(mp), 'recall': float(mr), 'f1': float(mf)},
                                  'macro': {'precision': float(Mp), 'recall': float(Mr), 'f1': float(Mf)}}
        log(f'\n--- aspect detection: {name} ---')
        for a in C.ASPECTS:
            d = per[a]
            log(f"  {a:<14s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}  (gold {d['support']:,}, predicted {d['predicted']:,})")
            rows.append({'Matcher': name, 'Aspect': a, **{k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}})
        log(f'  micro P={mp:.3f} R={mr:.3f} F1={mf:.3f} | macro P={Mp:.3f} R={Mr:.3f} F1={Mf:.3f}')
        rows.append({'Matcher': name, 'Aspect': 'micro', 'precision': round(mp, 4), 'recall': round(mr, 4), 'f1': round(mf, 4)})
        rows.append({'Matcher': name, 'Aspect': 'macro', 'precision': round(Mp, 4), 'recall': round(Mr, 4), 'f1': round(Mf, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULT_DIR, 'absa_validation_detection.csv'), index=False)

    # (b) sentence polarity
    vader = SentimentIntensityAnalyzer()
    comp = df['review'].map(lambda s: vader.polarity_scores(s)['compound'])
    pred_pol = np.where(comp >= 0.05, 'positive', np.where(comp <= -0.05, 'negative', 'neutral'))
    lab = np.where(df['positive'] == 1, 'positive', np.where(df['negative'] == 1, 'negative',
                   np.where(df['neutral'] == 1, 'neutral', 'none')))
    m = lab != 'none'
    acc = accuracy_score(lab[m], pred_pol[m]); mf1 = f1_score(lab[m], pred_pol[m], average='macro')
    kap = cohen_kappa_score(lab[m], pred_pol[m])
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(lab[m], pred_pol[m], labels=labels)
    pn = np.isin(lab, ['positive', 'negative'])
    acc_pn = accuracy_score(lab[pn], np.where(comp[pn] >= 0, 'positive', 'negative'))
    log(f'\n--- VADER sentence polarity vs human label (n = {m.sum():,}) ---')
    log(f'  accuracy {acc:.3f}, macro-F1 {mf1:.3f}, kappa {kap:.3f}; binary (pos/neg sentences, sign of compound) accuracy {acc_pn:.3f}')
    log('  confusion (rows = human, cols = VADER; negative/neutral/positive):\n' + str(cm))
    out['sentence_polarity'] = {'n': int(m.sum()), 'accuracy': float(acc), 'macro_f1': float(mf1), 'kappa': float(kap),
                                'binary_accuracy_pos_neg': float(acc_pn), 'confusion_labels': labels, 'confusion': cm.tolist()}

    # (c) aspect-level sign agreement on detected aspects
    agree = {}
    for a in C.ASPECTS:
        sel = (pred_wb[a] == 1).values & pn
        if sel.sum() == 0:
            continue
        a_sign = np.where(comp[sel] >= 0, 'positive', 'negative')
        agree[a] = {'n': int(sel.sum()), 'agreement': float(np.mean(a_sign == lab[sel]))}
    allsel = (pred_wb.values.sum(axis=1) > 0) & pn
    agree['all'] = {'n': int(allsel.sum()),
                    'agreement': float(np.mean(np.where(comp[allsel] >= 0, 'positive', 'negative') == lab[allsel]))}
    out['aspect_sign_agreement'] = agree
    log('\n--- sign agreement of aspect sentiment with human sentence polarity (aspect detected, pos/neg sentences) ---')
    for a, d in agree.items():
        log(f"  {a:<14s} n={d['n']:>6,}  agreement {d['agreement']:.3f}")
    json.dump(out, open(os.path.join(C.RESULT_DIR, 'absa_validation.json'), 'w'), indent=2)
    log('\nv3 Step 12 complete.')


if __name__ == '__main__':
    main()
