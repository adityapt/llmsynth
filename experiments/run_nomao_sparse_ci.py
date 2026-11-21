"""
Nomao sparse stress test with 5 seeds (Figure 3 of the Electronics manuscript).

n=500 rows are drawn from Nomao, 70% of the feature values are zeroed, and the 80/20 split is
stratified. The dense reference uses the same 500 rows before any values are zeroed.
GaussianCopula and CTGAN are run at each alpha. The downstream classifier is Gradient Boosting
(100 trees, depth 4, random_state=42), the same as `train_evaluate` in synthetic_data_eval.py.

The protocol is the one in run_nomao_sparse.py (single seed), extended to seeds [42, 123, 7, 2024, 999]:
the subsample, the zeroing mask and the split all use the seed, and the classifier keeps random_state=42.
The Baseline and DenseBaseline AUC of the earlier file ci_nomao_sparse.csv are reproduced exactly for all
five seeds by this protocol, which is how the protocol was recovered.

Output: results/ci_nomao_sparse.csv (seed, method, alpha, auc, plus f1_minority, avg_precision,
precision, recall, accuracy). `auc` is a copy of `auc_roc` kept for the plot and verification scripts.
Usage (from the repo root): python3 experiments/run_nomao_sparse_ci.py
"""
import sys, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             precision_score, recall_score, accuracy_score)
from experiments.synthetic_data_eval import (DATA_DIR, RESULTS_DIR, RANDOM_STATE,
                                             generate_ctgan, generate_gaussian_copula)

SEEDS = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_SMALL, SPARSITY, TARGET = 500, 0.70, "target"
OUT = RESULTS_DIR / "ci_nomao_sparse.csv"


def metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }


def fit_eval(X_tr, y_tr, X_te, y_te):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    return metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))


def apply_sparsity(df, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = [c for c in df.columns if c != TARGET]
    mask = rng.random((len(out), len(cols))) < SPARSITY
    out[cols] = out[cols].values * (1 - mask)
    return out


def xy(d):
    return d.drop(columns=[TARGET]).values.astype(float), d[TARGET].values


df_full = pd.read_csv(DATA_DIR / "nomao.csv")
rows = []


def add(seed, method, alpha, m):
    rows.append({"seed": seed, "method": method, "alpha": alpha, "auc": m["auc_roc"], **m})
    pd.DataFrame(rows).to_csv(OUT, index=False)


for seed in SEEDS:
    print(f"\nSeed {seed}", flush=True)
    np.random.seed(seed)
    small = df_full.sample(N_SMALL, random_state=seed).reset_index(drop=True)
    sparse = apply_sparsity(small, seed)
    tr, te = train_test_split(sparse, test_size=0.2, random_state=seed, stratify=sparse[TARGET])
    tr_d, te_d = train_test_split(small, test_size=0.2, random_state=seed, stratify=small[TARGET])
    X_tr, y_tr = xy(tr)
    X_te, y_te = xy(te)

    add(seed, "DenseBaseline", 0.0, fit_eval(*xy(tr_d), *xy(te_d)))
    add(seed, "Baseline", 0.0, fit_eval(X_tr, y_tr, X_te, y_te))
    print(f"  Dense AUC={rows[-2]['auc']:.4f}  Sparse baseline AUC={rows[-1]['auc']:.4f}", flush=True)

    for gen_name, gen_fn in [("GaussianCopula", generate_gaussian_copula), ("CTGAN", generate_ctgan)]:
        for alpha in ALPHAS:
            n_syn = int(len(tr) * alpha)
            df_syn = gen_fn(tr, TARGET, n_syn, "classification")
            X_s, y_s = xy(df_syn)
            m = fit_eval(np.vstack([X_tr, X_s]), np.concatenate([y_tr, y_s]), X_te, y_te)
            add(seed, gen_name, alpha, m)
            print(f"  {gen_name} alpha={alpha}: AUC={m['auc_roc']:.4f}", flush=True)
            gc.collect()

print(f"\nSaved: {OUT}", flush=True)
