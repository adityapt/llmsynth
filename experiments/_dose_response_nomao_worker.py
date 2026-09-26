"""
Single-unit-of-work worker for the Nomao dose-response sweep. Run as a fresh
subprocess per (minority_count, seed, method) — guarantees full OS-level
memory release between fits, which plain gc.collect() within a long-lived
process was not achieving for Nomao's larger (119-feature) CTGAN/GC models.

Usage: python3 _dose_response_nomao_worker.py <minority_count> <seed> <method>
Appends one row to results/dose_response_nomao.csv.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from experiments.synthetic_data_eval import RESULTS_DIR, DATA_DIR, generate_ctgan, generate_gaussian_copula, generate_smote

TARGET = "target"
N_TOTAL = 10_000
HOLDOUT_SIZE = 3_000
OUT = RESULTS_DIR / "dose_response_nomao.csv"

m_count, seed, method = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
pos_rate = m_count / N_TOTAL * 100

df = pd.read_csv(DATA_DIR / "nomao.csv")
for c in df.columns:
    if c != TARGET and df[c].dtype == object:
        df[c] = df[c].astype("category").cat.codes
df = df.fillna(df.median(numeric_only=True))

_, df_holdout = train_test_split(df, test_size=HOLDOUT_SIZE, random_state=42, stratify=df[TARGET])
df_pool = df.drop(df_holdout.index).reset_index(drop=True)
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values

pos_pool = df_pool[df_pool[TARGET] == 1].reset_index(drop=True)
neg_pool = df_pool[df_pool[TARGET] == 0].reset_index(drop=True)

n_neg = N_TOTAL - m_count
df_tr = pd.concat([
    pos_pool.sample(m_count, random_state=seed),
    neg_pool.sample(n_neg, random_state=seed),
]).sample(frac=1, random_state=seed).reset_index(drop=True)
X_tr = df_tr.drop(columns=[TARGET])
y_tr = df_tr[TARGET]

if method == "Baseline":
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr.values.astype(float), y_tr.values)
    proba, preds = clf.predict_proba(X_ho)[:, 1], clf.predict(X_ho)
else:
    n_syn = len(df_tr)
    if method == "SMOTE":
        df_syn = generate_smote(X_tr, y_tr, n_syn)
    else:
        gen_fn = generate_gaussian_copula if method == "GaussianCopula" else generate_ctgan
        df_syn = gen_fn(df_tr, TARGET, n_syn, "classification")
    X_aug = np.vstack([X_tr.values.astype(float), df_syn.drop(columns=[TARGET]).values.astype(float)])
    y_aug = np.concatenate([y_tr.values, df_syn[TARGET].values])
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_aug, y_aug)
    proba, preds = clf.predict_proba(X_ho)[:, 1], clf.predict(X_ho)

row = {
    "minority_count": m_count, "positive_rate_pct": round(pos_rate, 3), "seed": seed, "method": method,
    "auc_roc": roc_auc_score(y_ho, proba),
    "f1_minority": f1_score(y_ho, preds, pos_label=1, zero_division=0),
    "avg_precision": average_precision_score(y_ho, proba),
}
print(f"minority_count={m_count} seed={seed} {method}: AUC={row['auc_roc']:.4f}", flush=True)

existing = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
existing.append(row)
pd.DataFrame(existing).to_csv(OUT, index=False)
