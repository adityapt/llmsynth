"""
Bank Marketing dose-response sweep at a smaller mixing ratio (default alpha=0.2).

run_dose_response_bank_marketing.py uses alpha=1.0 only, the ratio the manuscript advises against. This script
repeats the same design at alpha=0.2 to test whether the harm seen at low minority counts is specific to alpha=1.0:
the same holdout (3,000 rows, seed 42), the same training draws (N=10,000, minority count m in
{16, 64, 256, 512, 1024}), the same seeds, and the same classifier. ADASYN, the free method the manuscript
recommends trying first, is added to the generators (GaussianCopula, CTGAN, SMOTE, ADASYN).
The Baseline rows must equal those of dose_response_bank_marketing.csv, which is how the protocol is checked.

Usage (from the repo root): python3 experiments/run_dose_response_bank_alpha.py [alpha]
Output: results/dose_response_bank_marketing_alpha<alpha>.csv
"""
import sys, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             precision_score, recall_score, accuracy_score)

from experiments.synthetic_data_eval import (RESULTS_DIR, DATA_DIR, generate_ctgan,
                                             generate_gaussian_copula, generate_smote)

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
SEEDS = [42, 123, 7, 2024, 999]
MINORITY_COUNTS = [16, 64, 256, 512, 1024]
N_TOTAL = 10_000
HOLDOUT_SIZE = 3_000
TARGET = "target"
OUT = RESULTS_DIR / f"dose_response_bank_marketing_alpha{str(ALPHA).replace('.', '')}.csv"

df = pd.read_csv(DATA_DIR / "bank_marketing.csv")
df = df.rename(columns={"Class": TARGET})
df[TARGET] = (df[TARGET] == 2).astype(int)
for c in df.columns:
    if c != TARGET and df[c].dtype == object:
        df[c] = df[c].astype("category").cat.codes

_, df_holdout = train_test_split(df, test_size=HOLDOUT_SIZE, random_state=42, stratify=df[TARGET])
df_pool = df.drop(df_holdout.index).reset_index(drop=True)
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values
pos_pool = df_pool[df_pool[TARGET] == 1].reset_index(drop=True)
neg_pool = df_pool[df_pool[TARGET] == 0].reset_index(drop=True)
print(f"alpha={ALPHA}; holdout {len(df_holdout)} rows, {y_ho.sum()} positives", flush=True)


def metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }


def fit_eval(X, y, seed):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X, y)
    return metrics(y_ho, clf.predict_proba(X_ho)[:, 1], clf.predict(X_ho))


rows, done = [], set()
if OUT.exists():
    rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["minority_count"], r["seed"], r["method"]) for r in rows}
    print(f"Resuming, {len(done)} combinations already done", flush=True)

for m_count in MINORITY_COUNTS:
    n_neg = N_TOTAL - m_count
    pos_rate = m_count / N_TOTAL * 100
    print(f"\n=== minority_count={m_count} (positive rate {pos_rate:.2f}%) ===", flush=True)
    for seed in SEEDS:
        df_tr = pd.concat([
            pos_pool.sample(m_count, random_state=seed),
            neg_pool.sample(n_neg, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET])
        y_tr = df_tr[TARGET]
        n_syn = int(len(df_tr) * ALPHA)

        def record(method, m):
            rows.append({"minority_count": m_count, "positive_rate_pct": round(pos_rate, 3), "alpha": ALPHA,
                         "seed": seed, "method": method, **m})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  seed={seed} {method}: AUC={m['auc_roc']:.4f}", flush=True)

        if (m_count, seed, "Baseline") not in done:
            record("Baseline", fit_eval(X_tr.values.astype(float), y_tr.values, seed))

        for gen_name in ["GaussianCopula", "CTGAN", "SMOTE", "ADASYN"]:
            if (m_count, seed, gen_name) in done:
                continue
            try:
                if gen_name == "SMOTE":
                    df_syn = generate_smote(X_tr, y_tr, n_syn)
                    X_syn, y_syn = df_syn.drop(columns=[TARGET]).values.astype(float), df_syn[TARGET].values
                elif gen_name == "ADASYN":
                    # same construction as run_missing_baselines_hillstrom_criteo.py
                    sampler = ADASYN(sampling_strategy={1: m_count + n_syn}, random_state=seed, n_neighbors=min(5, m_count - 1))
                    X_res, y_res = sampler.fit_resample(X_tr.values.astype(float), y_tr.values)
                    X_syn, y_syn = X_res[len(X_tr):], y_res[len(X_tr):]
                else:
                    gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
                    df_syn = gen_fn(df_tr, TARGET, n_syn, "classification")
                    X_syn, y_syn = df_syn.drop(columns=[TARGET]).values.astype(float), df_syn[TARGET].values
                X_aug = np.vstack([X_tr.values.astype(float), X_syn])
                y_aug = np.concatenate([y_tr.values, y_syn])
                record(gen_name, fit_eval(X_aug, y_aug, seed))
            except Exception as e:
                print(f"  seed={seed} {gen_name} FAILED: {e}", flush=True)
            gc.collect()

print(f"\nDone. Saved: {OUT}", flush=True)
