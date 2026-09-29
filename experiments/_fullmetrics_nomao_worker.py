"""
Single-unit-of-work worker for Nomao's full-metrics rerun. Run as a fresh
subprocess per (seed, method, alpha) -- Nomao's 119 features make CTGAN/
GaussianCopula fits heavy enough that the monolithic version of this script
(run_full_metrics_benchmark_datasets.py) kept getting OOM-killed partway
through, same underlying reason the dose-response Nomao replication needed
this same subprocess pattern earlier.

Usage: python3 _fullmetrics_nomao_worker.py <seed> <method> <alpha>
  method: Baseline | GaussianCopula | CTGAN | SMOTE
  alpha: 0 for Baseline, else one of 0.1/0.2/0.3/0.5/1.0
Appends one row to results/fullmetrics_nomao.csv.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)

from experiments.synthetic_data_eval import RESULTS_DIR, generate_ctgan, generate_gaussian_copula, generate_smote
from experiments.run_nomao import load_nomao

seed = int(sys.argv[1])
method = sys.argv[2]
alpha = float(sys.argv[3])
N_CAP = 10_000
OUT = RESULTS_DIR / "fullmetrics_nomao.csv"

df_full, target, task, name = load_nomao()
n_use = min(N_CAP, len(df_full))
df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
X_tr_df = df_train.drop(columns=[target])
y_tr = df_train[target]
X_te = df_test.drop(columns=[target]).values.astype(float)
y_te = df_test[target].values
X_tr = X_tr_df.values.astype(float)

if method == "Baseline":
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr.values)
    proba, preds = clf.predict_proba(X_te)[:, 1], clf.predict(X_te)
else:
    n_syn = int(len(df_train) * alpha)
    if method == "SMOTE":
        df_syn = generate_smote(X_tr_df, y_tr, n_syn)
    else:
        gen_fn = generate_gaussian_copula if method == "GaussianCopula" else generate_ctgan
        df_syn = gen_fn(df_train, target, n_syn, task)
    X_aug = np.vstack([X_tr, df_syn.drop(columns=[target]).values.astype(float)])
    y_aug = np.concatenate([y_tr.values, df_syn[target].values])
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_aug, y_aug)
    proba, preds = clf.predict_proba(X_te)[:, 1], clf.predict(X_te)

row = {
    "seed": seed, "method": method, "alpha": alpha if method != "Baseline" else 0,
    "auc_roc": roc_auc_score(y_te, proba),
    "f1_minority": f1_score(y_te, preds, pos_label=1, zero_division=0),
    "avg_precision": average_precision_score(y_te, proba),
    "precision": precision_score(y_te, preds, pos_label=1, zero_division=0),
    "recall": recall_score(y_te, preds, pos_label=1, zero_division=0),
    "accuracy": accuracy_score(y_te, preds),
}
print(f"seed={seed} {method} alpha={alpha}: AUC={row['auc_roc']:.4f}", flush=True)

existing = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
existing.append(row)
pd.DataFrame(existing).to_csv(OUT, index=False)
