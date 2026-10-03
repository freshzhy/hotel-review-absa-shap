"""
v3 Step 9 - Pretrained language model baselines and computational cost.

Question answered
-----------------
How does the compact 26-feature Random Forest compare with representations
from pretrained transformer language models, evaluated on EXACTLY the same
50,000-review sample and the same five stratified folds as Steps 3-4?

Configurations
--------------
(1) Frozen sentence encoder all-MiniLM-L6-v2  + class-weighted logistic regression
(2) Frozen sentence encoder all-mpnet-base-v2 + class-weighted logistic regression
(3) [optional, --finetune] DistilBERT (distilbert-base-uncased) fine-tuned per
    fold with class-weighted cross-entropy (2 epochs, max length 128).

A frozen encoder has no parameters fitted to the review data, so encoding all
50,000 texts once is leakage-free; the scaler and the classifier are fitted
inside each training fold.  The fine-tuned model is trained only on each
training fold.

Statistical tests
-----------------
Every PLM configuration is compared with (a) the 26-feature Random Forest and
(b) TF-IDF + logistic regression, using the fold scores stored by Step 4 and
the corrected resampled t-test; Holm correction is applied within this step.

Computational cost
------------------
Wall-clock time of feature extraction / encoding for the 50,000 reviews and of
training + prediction per fold is recorded for every configuration, including
the 26-feature RF (VADER feature extraction timed single-process).

Usage
-----
    python v3_09_plm_baselines.py                 # frozen encoders only (~10-20 min)
    python v3_09_plm_baselines.py --finetune      # + DistilBERT fine-tuning (~1-3 h on Apple silicon)
    python v3_09_plm_baselines.py --finetune --skip-frozen   # resume: fine-tuning only

Fine-tuning results are checkpointed per fold in data/plm_ft_fold*.json, so an
interrupted run can simply be restarted.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import stats_utils as S

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')


def log(m=''):
    print(m, flush=True)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def metrics(y, p):
    return {'accuracy': accuracy_score(y, p),
            'f1_macro': f1_score(y, p, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, p, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(y, p),
            'f1_per_class': f1_score(y, p, average=None, zero_division=0, labels=[0, 1, 2]).tolist()}


def summarise(name, folds, le, extra):
    keys = ['accuracy', 'f1_macro', 'f1_weighted', 'kappa']
    entry = {k: S.mean_ci([f[k] for f in folds]) for k in keys}
    entry['f1_per_class'] = {c: S.mean_ci([f['f1_per_class'][i] for f in folds])
                             for i, c in enumerate(le.classes_)}
    entry['fold_scores'] = {k: [float(f[k]) for f in folds] for k in keys}
    entry.update(extra)
    log(f'  => {name}: F1_macro {S.fmt_ci(entry["f1_macro"])}  '
        f'acc {entry["accuracy"]["mean"]:.4f}  kappa {entry["kappa"]["mean"]:.4f}  '
        f'F1_Low {entry["f1_per_class"]["Low"]["mean"]:.4f}')
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--finetune', action='store_true', help='also fine-tune DistilBERT')
    ap.add_argument('--skip-frozen', action='store_true', help='skip the frozen-encoder runs')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--minilm', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--mpnet', default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--ft-model', default='distilbert-base-uncased')
    ap.add_argument('--quick', type=int, default=0,
                    help='smoke test: use only this many reviews (results NOT for the paper)')
    args = ap.parse_args()

    log('=' * 70)
    log('v3 Step 9 | Pretrained language model baselines and computational cost')
    log('=' * 70)

    # ------------------------------------------------------------------
    # Identical sample and folds as Steps 3-4
    # ------------------------------------------------------------------
    X_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_X.csv'))
    y_all = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_y.csv'))['satisfaction_class'].values
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'booking_meta.csv'),
                       usecols=['Full_Review', 'Negative_Review', 'Positive_Review'])
    idx_all = np.arange(len(X_all))
    idx_s, _, y_s, _ = train_test_split(idx_all, y_all, train_size=C.SAMPLE_SIZE,
                                        stratify=y_all, random_state=C.RANDOM_STATE)
    X_s = X_all.iloc[idx_s].reset_index(drop=True)
    text_s = meta['Full_Review'].iloc[idx_s].fillna('').astype(str).reset_index(drop=True)
    neg_s = meta['Negative_Review'].iloc[idx_s].fillna('').astype(str).tolist()
    pos_s = meta['Positive_Review'].iloc[idx_s].fillna('').astype(str).tolist()
    le = LabelEncoder().fit(C.CLASS_NAMES)
    y_enc = le.transform(y_s)
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    splits = list(cv.split(X_s, y_enc))

    if args.quick:
        log(f'!! QUICK SMOKE TEST on {args.quick} reviews - results are NOT for the paper')
        keep = np.arange(args.quick)
        X_s, text_s, y_enc = X_s.iloc[keep].reset_index(drop=True), text_s.iloc[keep].reset_index(drop=True), y_enc[keep]
        neg_s, pos_s = neg_s[:args.quick], pos_s[:args.quick]
        splits = list(cv.split(X_s, y_enc))
    n_train, n_test = len(splits[0][0]), len(splits[0][1])
    log(f'Sample: {len(y_enc):,} reviews, {C.N_FOLDS} folds ({n_train:,} train / {n_test:,} test)')

    results = {}
    import platform
    hw = {'platform': platform.platform(), 'processor': platform.processor(),
          'cpu_count': os.cpu_count()}

    # ------------------------------------------------------------------
    # Reference 1: 26-feature RF (re-run for timing + sanity check of folds)
    # ------------------------------------------------------------------
    log('\n--- reference: 26 engineered features + Random Forest ---')
    t0 = time.time()
    C._init_worker()
    for i in range(len(neg_s)):
        C.text_features(([neg_s[i], pos_s[i]], text_s.iloc[i]))
    t_feat = time.time() - t0
    log(f'  text-feature extraction (VADER + lexicon), single process: {t_feat:.1f}s '
        f'for {len(neg_s):,} reviews ({len(neg_s)/t_feat:,.0f} reviews/s)')
    Xf = X_s[C.ALL_FEATURES].values
    folds, t_fit, t_pred = [], [], []
    for tr, va in splits:
        sc = StandardScaler().fit(Xf[tr])
        m = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10,
                                   min_samples_leaf=5, class_weight='balanced',
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        a = time.time(); m.fit(sc.transform(Xf[tr]), y_enc[tr]); t_fit.append(time.time() - a)
        a = time.time(); p = m.predict(sc.transform(Xf[va])); t_pred.append(time.time() - a)
        folds.append(metrics(y_enc[va], p))
    results['26 features + RF'] = summarise('26 features + RF', folds, le, {
        'n_dims': 26, 'extraction_seconds': t_feat,
        'fit_seconds_per_fold': float(np.mean(t_fit)), 'predict_seconds_per_fold': float(np.mean(t_pred))})

    # Sanity check against Step 4
    abl_path = os.path.join(C.RESULT_DIR, 'ablation_results.json')
    ref_rf = ref_tfidf = None
    if os.path.exists(abl_path) and not args.quick:
        abl = json.load(open(abl_path))['results']
        ref_rf = abl['Full (A+B+C+D)']['fold_scores']['f1_macro']
        ref_tfidf = abl['TF-IDF + Logistic Regression']['fold_scores']['f1_macro']
        diff = np.max(np.abs(np.array(ref_rf) - np.array(results['26 features + RF']['fold_scores']['f1_macro'])))
        log(f'  fold-score check vs Step 4 (max abs diff): {diff:.2e}  '
            f'{"OK - identical folds" if diff < 1e-9 else "WARNING - folds differ"}')

    # ------------------------------------------------------------------
    # Reference 2: TF-IDF + LR (timing only; scores taken from Step 4)
    # ------------------------------------------------------------------
    log('\n--- reference: TF-IDF + logistic regression (timing) ---')
    folds, t_fit, t_pred = [], [], []
    for tr, va in splits:
        a = time.time()
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
        Xtr = vec.fit_transform(text_s.iloc[tr])
        m = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs',
                               random_state=C.RANDOM_STATE, n_jobs=-1).fit(Xtr, y_enc[tr])
        t_fit.append(time.time() - a)
        a = time.time(); p = m.predict(vec.transform(text_s.iloc[va])); t_pred.append(time.time() - a)
        folds.append(metrics(y_enc[va], p))
    results['TF-IDF + LR'] = summarise('TF-IDF + LR', folds, le, {
        'n_dims': 5000, 'extraction_seconds': 0.0,
        'fit_seconds_per_fold': float(np.mean(t_fit)), 'predict_seconds_per_fold': float(np.mean(t_pred))})
    if ref_tfidf is None:
        ref_tfidf = results['TF-IDF + LR']['fold_scores']['f1_macro']
    if ref_rf is None:
        ref_rf = results['26 features + RF']['fold_scores']['f1_macro']

    # ------------------------------------------------------------------
    # Frozen sentence encoders + LR
    # ------------------------------------------------------------------
    if not args.skip_frozen:
        from sentence_transformers import SentenceTransformer
        device = pick_device()
        for tag, name in [('MiniLM-L6 (frozen) + LR', args.minilm),
                          ('MPNet-base (frozen) + LR', args.mpnet)]:
            log(f'\n--- {tag}  [{name}] on {device} ---')
            cache = os.path.join(C.DATA_DIR, f'emb_{name.split("/")[-1]}_{len(text_s)}.npy')
            t0 = time.time()
            if os.path.exists(cache):
                E = np.load(cache)
                t_enc = json.load(open(cache + '.json'))['seconds']
                log(f'  loaded cached embeddings {E.shape} (original encoding time {t_enc:.0f}s)')
            else:
                enc = SentenceTransformer(name, device=device)
                E = enc.encode(text_s.tolist(), batch_size=128, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=False)
                t_enc = time.time() - t0
                np.save(cache, E)
                json.dump({'seconds': t_enc, 'device': device}, open(cache + '.json', 'w'))
                log(f'  encoded {E.shape} in {t_enc:.0f}s')
            folds, t_fit, t_pred = [], [], []
            for tr, va in splits:
                sc = StandardScaler().fit(E[tr])
                m = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs',
                                       random_state=C.RANDOM_STATE, n_jobs=-1)
                a = time.time(); m.fit(sc.transform(E[tr]), y_enc[tr]); t_fit.append(time.time() - a)
                a = time.time(); p = m.predict(sc.transform(E[va])); t_pred.append(time.time() - a)
                folds.append(metrics(y_enc[va], p))
            results[tag] = summarise(tag, folds, le, {
                'model': name, 'n_dims': int(E.shape[1]), 'device': device,
                'extraction_seconds': float(t_enc),
                'fit_seconds_per_fold': float(np.mean(t_fit)),
                'predict_seconds_per_fold': float(np.mean(t_pred))})

    # ------------------------------------------------------------------
    # Optional: fine-tuned DistilBERT
    # ------------------------------------------------------------------
    if args.finetune:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        device = pick_device()
        torch.manual_seed(C.RANDOM_STATE)
        tag = 'DistilBERT (fine-tuned)'
        log(f'\n--- {tag}  [{args.ft_model}] on {device}: {args.epochs} epochs, '
            f'batch {args.batch_size}, max_len {args.max_len}, lr {args.lr} ---')
        tok = AutoTokenizer.from_pretrained(args.ft_model)
        texts = text_s.tolist()
        folds, t_fit, t_pred = [], [], []
        for k, (tr, va) in enumerate(splits):
            ck = os.path.join(C.DATA_DIR, f'plm_ft_fold{k}_{len(texts)}.json')
            if os.path.exists(ck):
                rec = json.load(open(ck))
                folds.append(rec['metrics']); t_fit.append(rec['fit_s']); t_pred.append(rec['pred_s'])
                log(f'  fold {k+1}: loaded checkpoint  F1m={rec["metrics"]["f1_macro"]:.4f}')
                continue
            torch.manual_seed(C.RANDOM_STATE + k)
            model = AutoModelForSequenceClassification.from_pretrained(args.ft_model, num_labels=3).to(device)
            counts = np.bincount(y_enc[tr], minlength=3)
            w = torch.tensor(len(tr) / (3 * counts), dtype=torch.float32, device=device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)

            def collate(batch):
                enc = tok([b[0] for b in batch], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors='pt')
                enc['labels'] = torch.tensor([b[1] for b in batch])
                return enc

            train_ds = [(texts[i], int(y_enc[i])) for i in tr]
            g = torch.Generator().manual_seed(C.RANDOM_STATE + k)
            dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate, generator=g)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            total = len(dl) * args.epochs
            sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
            a = time.time()
            model.train()
            for ep in range(args.epochs):
                for step, batch in enumerate(dl):
                    labels = batch.pop('labels').to(device)
                    batch = {kk: v.to(device) for kk, v in batch.items()}
                    loss = loss_fn(model(**batch).logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
                    if (step + 1) % 200 == 0:
                        log(f'    fold {k+1} epoch {ep+1} step {step+1}/{len(dl)} loss {loss.item():.4f} '
                            f'({time.time()-a:.0f}s)')
            fit_s = time.time() - a
            a = time.time()
            model.eval()
            preds = []
            va_ds = [(texts[i], int(y_enc[i])) for i in va]
            with torch.no_grad():
                for batch in DataLoader(va_ds, batch_size=128, collate_fn=collate):
                    batch.pop('labels')
                    batch = {kk: v.to(device) for kk, v in batch.items()}
                    preds.append(model(**batch).logits.argmax(-1).cpu().numpy())
            pred_s = time.time() - a
            mt = metrics(y_enc[va], np.concatenate(preds))
            json.dump({'metrics': mt, 'fit_s': fit_s, 'pred_s': pred_s}, open(ck, 'w'))
            folds.append(mt); t_fit.append(fit_s); t_pred.append(pred_s)
            log(f'  fold {k+1}: F1m={mt["f1_macro"]:.4f}  acc={mt["accuracy"]:.4f}  '
                f'(train {fit_s:.0f}s, predict {pred_s:.0f}s)')
            del model
        results[tag] = summarise(tag, folds, le, {
            'model': args.ft_model, 'device': device, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'max_len': args.max_len, 'lr': args.lr,
            'extraction_seconds': 0.0,
            'fit_seconds_per_fold': float(np.mean(t_fit)),
            'predict_seconds_per_fold': float(np.mean(t_pred))})

    # ------------------------------------------------------------------
    # Significance tests (Holm within this step)
    # ------------------------------------------------------------------
    plm = [k for k in results if k not in ('26 features + RF', 'TF-IDF + LR')]
    tests = []
    for k in plm:
        f = results[k]['fold_scores']['f1_macro']
        for ref_name, ref in [('26 features + RF', ref_rf), ('TF-IDF + LR', ref_tfidf)]:
            t, p, d = S.corrected_resampled_ttest(f, ref, n_train, n_test)
            tests.append({'configuration': k, 'reference': ref_name,
                          'delta_f1': d, 't': t, 'p_raw': p})
    if tests:
        order = np.argsort([r['p_raw'] for r in tests])
        m, run = len(tests), 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * tests[i]['p_raw']))
            tests[i]['p_holm'] = run
            tests[i]['significant_05'] = bool(run < 0.05)
        log(f'\n{"="*70}\nPLM configurations vs references (corrected resampled t-test, Holm within step):')
        for r in tests:
            flag = '*' if r['significant_05'] else ' '
            log(f'  {flag} {r["configuration"]:<28s} vs {r["reference"]:<18s} '
                f'dF1={r["delta_f1"]:+.4f}  t={r["t"]:+.3f}  p_raw={r["p_raw"]:.4f}  p_holm={S.fmt_p(r["p_holm"])}')

    rows = []
    for k, r in results.items():
        rows.append({
            'Configuration': k, 'Dims': r.get('n_dims', ''),
            'Accuracy': f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['sd']:.4f}",
            'F1_macro': f"{r['f1_macro']['mean']:.4f} ± {r['f1_macro']['sd']:.4f}",
            'F1_macro_95CI': f"[{r['f1_macro']['ci_low']:.4f}, {r['f1_macro']['ci_high']:.4f}]",
            'Kappa': f"{r['kappa']['mean']:.4f}",
            'F1_Low': f"{r['f1_per_class']['Low']['mean']:.4f}",
            'F1_Medium': f"{r['f1_per_class']['Medium']['mean']:.4f}",
            'F1_High': f"{r['f1_per_class']['High']['mean']:.4f}",
            'Extraction_s': f"{r.get('extraction_seconds', 0):.1f}",
            'Fit_s_per_fold': f"{r['fit_seconds_per_fold']:.1f}",
            'Predict_s_per_fold': f"{r['predict_seconds_per_fold']:.2f}",
        })
    df_out = pd.DataFrame(rows)
    log('\n' + df_out.to_string(index=False))
    log(f'\nHardware: {hw}')
    if not args.quick:
        df_out.to_csv(os.path.join(C.RESULT_DIR, 'plm_baselines.csv'), index=False)
        with open(os.path.join(C.RESULT_DIR, 'plm_baselines.json'), 'w') as f:
            json.dump({'results': results, 'tests': tests, 'hardware': hw,
                       'n_train_per_fold': n_train, 'n_test_per_fold': n_test,
                       'args': vars(args)}, f, indent=2)
    log('\nv3 Step 9 complete.')


if __name__ == '__main__':
    main()
