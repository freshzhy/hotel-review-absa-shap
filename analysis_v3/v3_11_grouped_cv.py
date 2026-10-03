"""
v3 Step 11 - Hotel-grouped cross-validation (sensitivity analysis).

Ordinary stratified K-fold lets reviews of the same hotel appear in both the
training and the test fold.  Hotel-specific vocabulary (names, streets,
recurring complaints) can then inflate text representations in particular.
This step repeats the core comparisons on the SAME 50,000-review sample with
StratifiedGroupKFold (groups = hotels), so that every hotel is either in the
training or in the test fold:

  * the five learners on the engineered features (+ 2 naive baselines)
  * TF-IDF + logistic regression
  * frozen MiniLM / MPNet embeddings + logistic regression (cached by Step 9)
  * [--finetune] DistilBERT fine-tuned per fold (checkpointed)

Each configuration is compared with the random forest under grouped CV
(corrected resampled t-test, Holm), and grouped scores are reported next to
the ordinary-CV scores from Steps 3, 4 and 9.
"""
import argparse, faulthandler, json, os, sys, time
faulthandler.enable()  # print a traceback if the process crashes in native code
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import stats_utils as S
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


def metrics(y, p):
    return {'accuracy': accuracy_score(y, p),
            'f1_macro': f1_score(y, p, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y, p),
            'f1_per_class': f1_score(y, p, average=None, zero_division=0, labels=[0, 1, 2]).tolist()}


def summarise(folds, le):
    e = {k: S.mean_ci([f[k] for f in folds]) for k in ['accuracy', 'f1_macro', 'kappa']}
    e['f1_per_class'] = {c: float(np.mean([f['f1_per_class'][i] for f in folds])) for i, c in enumerate(le.classes_)}
    e['fold_scores'] = {k: [float(f[k]) for f in folds] for k in ['accuracy', 'f1_macro', 'kappa']}
    return e


def finetune(args, text, y, splits):
    """Fine-tune DistilBERT per grouped fold; per-fold results are checkpointed."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(args.ft_model); texts = text.tolist(); folds = []
    for k, (tr, va) in enumerate(splits):
        ck = os.path.join(C.DATA_DIR, f'plm_ft_grouped_fold{k}_{len(texts)}.json')
        if os.path.exists(ck):
            folds.append(json.load(open(ck))['metrics']); log(f'  DistilBERT fold {k+1}: checkpoint loaded'); continue
        torch.manual_seed(C.RANDOM_STATE + k)
        model = AutoModelForSequenceClassification.from_pretrained(args.ft_model, num_labels=3).to(device)
        cnt = np.bincount(y[tr], minlength=3)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (3 * cnt), dtype=torch.float32, device=device))

        def collate(batch):
            enc = tok([b[0] for b in batch], truncation=True, max_length=args.max_len, padding=True, return_tensors='pt')
            enc['labels'] = torch.tensor([b[1] for b in batch]); return enc
        dl = DataLoader([(texts[i], int(y[i])) for i in tr], batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(C.RANDOM_STATE + k))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total = len(dl) * args.epochs; sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
        a = time.time(); model.train()
        for ep in range(args.epochs):
            for step, batch in enumerate(dl):
                labels = batch.pop('labels').to(device); batch = {kk: v.to(device) for kk, v in batch.items()}
                loss = loss_fn(model(**batch).logits, labels); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
                if (step + 1) % 200 == 0:
                    log(f'    fold {k+1} epoch {ep+1} step {step+1}/{len(dl)} loss {loss.item():.4f} ({time.time()-a:.0f}s)')
        model.eval(); preds = []
        with torch.no_grad():
            for batch in DataLoader([(texts[i], int(y[i])) for i in va], batch_size=128, collate_fn=collate):
                batch.pop('labels'); batch = {kk: v.to(device) for kk, v in batch.items()}
                preds.append(model(**batch).logits.argmax(-1).cpu().numpy())
        mt = metrics(y[va], np.concatenate(preds)); json.dump({'metrics': mt}, open(ck, 'w'))
        folds.append(mt); log(f'  DistilBERT fold {k+1}: F1m={mt["f1_macro"]:.4f}'); del model
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--finetune', action='store_true')
    ap.add_argument('--ft-only', action='store_true',
                    help='only fine-tune DistilBERT and write fold checkpoints (run in a fresh process: '
                         'on macOS, torch/MPS can deadlock after XGBoost/LightGBM have initialised OpenMP)')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--ft-model', default='distilbert-base-uncased')
    args = ap.parse_args()

    log('=' * 70); log('v3 Step 11 | Hotel-grouped cross-validation'); log('=' * 70)
    X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
    y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'), usecols=['Hotel_Name', 'Full_Review'])
    idx_s, _, y_s, _ = train_test_split(np.arange(len(X_all)), y_all, train_size=C.SAMPLE_SIZE,
                                        stratify=y_all, random_state=C.RANDOM_STATE)
    Xs = X_all.iloc[idx_s][C.ALL_FEATURES].values
    text = meta['Full_Review'].iloc[idx_s].fillna('').astype(str).reset_index(drop=True)
    groups = meta['Hotel_Name'].iloc[idx_s].astype(str).values
    le = LabelEncoder().fit(C.CLASS_NAMES)
    y = le.transform(y_s)
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    splits = list(cv.split(Xs, y, groups))
    n_train = int(np.mean([len(tr) for tr, _ in splits])); n_test = int(np.mean([len(va) for _, va in splits]))
    for k, (tr, va) in enumerate(splits):
        assert not set(groups[tr]) & set(groups[va])
        log(f'  fold {k+1}: {len(tr):,} train / {len(va):,} test reviews, '
            f'{len(set(groups[va])):,} test hotels, Low share {np.mean(y[va] == le.transform(["Low"])[0])*100:.2f}%')
    log(f'{len(set(groups)):,} hotels in the sample; no hotel appears in both training and test folds')
    if args.ft_only:
        finetune(args, text, y, splits)
        log('\nDistilBERT grouped-CV checkpoints written; now run without --ft-only.')
        return

    def dense(factory, weights=False):
        folds = []
        for tr, va in splits:
            sc = StandardScaler().fit(Xs[tr])
            m = factory()
            kw = {'sample_weight': compute_sample_weight('balanced', y[tr])} if weights else {}
            m.fit(sc.transform(Xs[tr]), y[tr], **kw)
            folds.append(metrics(y[va], m.predict(sc.transform(Xs[va]))))
        return folds

    # imported here, not at module level: on macOS their OpenMP runtime conflicts with torch/MPS
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    R = {}
    learners = {
        'Majority-class baseline': (lambda: DummyClassifier(strategy='most_frequent'), False),
        'Stratified-random baseline': (lambda: DummyClassifier(strategy='stratified', random_state=C.RANDOM_STATE), False),
        'Logistic Regression': (lambda: LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=C.RANDOM_STATE, n_jobs=-1), False),
        'Linear SVM': (lambda: LinearSVC(class_weight='balanced', max_iter=2000, dual='auto', random_state=C.RANDOM_STATE), False),
        'Random Forest': (lambda: RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=C.RANDOM_STATE, n_jobs=-1), False),
        'XGBoost': (lambda: XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='multi:softprob', num_class=3, random_state=C.RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss', verbosity=0), True),
        'LightGBM': (lambda: LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1), False),
    }
    for name, (fac, w) in learners.items():
        t0 = time.time(); R[name] = summarise(dense(fac, w), le)
        log(f'  {name:<28s} F1m {S.fmt_ci(R[name]["f1_macro"])}  ({time.time()-t0:.0f}s)')

    folds = []
    for tr, va in splits:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
        m = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(vec.fit_transform(text.iloc[tr]), y[tr])
        folds.append(metrics(y[va], m.predict(vec.transform(text.iloc[va]))))
    R['TF-IDF + LR'] = summarise(folds, le)
    log(f'  {"TF-IDF + LR":<28s} F1m {S.fmt_ci(R["TF-IDF + LR"]["f1_macro"])}')

    for tag, fname in [('MiniLM-L6 (frozen) + LR', 'all-MiniLM-L6-v2'), ('MPNet-base (frozen) + LR', 'all-mpnet-base-v2')]:
        cache = os.path.join(C.DATA_DIR, f'emb_{fname}_{len(text)}.npy')
        if not os.path.exists(cache):
            log(f'  {tag}: cached embeddings not found ({cache}); run Step 9 first - skipped')
            continue
        E = np.load(cache); folds = []
        for tr, va in splits:
            sc = StandardScaler().fit(E[tr])
            m = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=C.RANDOM_STATE, n_jobs=-1)
            m.fit(sc.transform(E[tr]), y[tr]); folds.append(metrics(y[va], m.predict(sc.transform(E[va]))))
        R[tag] = summarise(folds, le)
        log(f'  {tag:<28s} F1m {S.fmt_ci(R[tag]["f1_macro"])}')

    if args.finetune:
        R['DistilBERT (fine-tuned)'] = summarise(finetune(args, text, y, splits), le)
        log(f'  {"DistilBERT (fine-tuned)":<28s} F1m {S.fmt_ci(R["DistilBERT (fine-tuned)"]["f1_macro"])}')

    # tests vs RF under grouped CV
    tests = []
    for k in R:
        if k == 'Random Forest':
            continue
        t, p, d = S.corrected_resampled_ttest(R[k]['fold_scores']['f1_macro'], R['Random Forest']['fold_scores']['f1_macro'], n_train, n_test)
        tests.append({'configuration': k, 'delta_vs_rf': d, 't': t, 'p_raw': p})
    order = np.argsort([r['p_raw'] for r in tests]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(tests) - rank) * tests[i]['p_raw'])); tests[i]['p_holm'] = run

    # ordinary-CV reference scores
    ref = {}
    def safe(path, fn):
        try:
            return fn(json.load(open(os.path.join(C.RESULT_DIR, path))))
        except Exception:
            return None
    for k in learners:
        ref[k] = safe('model_comparison.json', lambda d: d['results'][k]['f1_macro']['mean'])
    ref['TF-IDF + LR'] = safe('ablation_results.json', lambda d: d['results']['TF-IDF + Logistic Regression']['f1_macro']['mean'])
    for k in ['MiniLM-L6 (frozen) + LR', 'MPNet-base (frozen) + LR', 'DistilBERT (fine-tuned)']:
        ref[k] = safe('plm_baselines.json', lambda d: d['results'][k]['f1_macro']['mean'])

    rows = []
    tm = {t['configuration']: t for t in tests}
    for k, r in R.items():
        rows.append({'Configuration': k,
                     'F1_macro_random_CV': '' if ref.get(k) is None else f"{ref[k]:.4f}",
                     'F1_macro_grouped_CV': f"{r['f1_macro']['mean']:.4f} ± {r['f1_macro']['sd']:.4f}",
                     'F1_macro_grouped_95CI': f"[{r['f1_macro']['ci_low']:.4f}, {r['f1_macro']['ci_high']:.4f}]",
                     'Change': '' if ref.get(k) is None else f"{r['f1_macro']['mean'] - ref[k]:+.4f}",
                     'Kappa': f"{r['kappa']['mean']:.4f}", 'F1_Low': f"{r['f1_per_class']['Low']:.4f}",
                     'p_vs_RF_grouped_holm': '—' if k == 'Random Forest' else S.fmt_p(tm[k]['p_holm'])})
    df = pd.DataFrame(rows)
    log('\n' + df.to_string(index=False))
    df.to_csv(os.path.join(C.RESULT_DIR, 'grouped_cv_results.csv'), index=False)
    json.dump({'results': R, 'tests_vs_rf': tests, 'random_cv_reference': ref,
               'n_train_mean': n_train, 'n_test_mean': n_test, 'n_hotels': int(len(set(groups)))},
              open(os.path.join(C.RESULT_DIR, 'grouped_cv_results.json'), 'w'), indent=2)
    log('\nv3 Step 11 complete.')


if __name__ == '__main__':
    main()
