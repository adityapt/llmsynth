"""Finishes the last few missing rows for Criteo seed 999 (CTGAN alpha>0.1, all SMOTE)."""
import sys, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from experiments.synthetic_data_eval import RESULTS_DIR, generate_ctgan, generate_smote
from experiments.run_criteo import load_criteo_uplift

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)

SEED = 999
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP = 10_000
OUT = RESULTS_DIR / "ci_criteo_fullmetrics.csv"

def full_metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }

df_full, target, task, name = load_criteo_uplift()
existing = pd.read_csv(OUT)
all_rows = existing.to_dict("records")

n_use = min(N_CAP, len(df_full))
np.random.seed(SEED)
df = df_full.sample(n_use, random_state=SEED).reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[target])
X_te = df_test.drop(columns=[target]).values.astype(float)
y_te = df_test[target].values
X_tr = df_train.drop(columns=[target])
y_tr = df_train[target]

todo = [("CTGAN", a) for a in [0.2, 0.3, 0.5, 1.0]] + [("SMOTE", a) for a in ALPHAS]
for gen_name, alpha in todo:
    n_syn = int(len(df_train) * alpha)
    if gen_name == "SMOTE":
        df_syn = generate_smote(X_tr, y_tr, n_syn)
    else:
        df_syn = generate_ctgan(df_train, target, n_syn, task)
    X_aug = np.vstack([X_tr.values.astype(float), df_syn.drop(columns=[target]).values.astype(float)])
    y_aug = np.concatenate([y_tr.values, df_syn[target].values])
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED)
    clf.fit(X_aug, y_aug)
    m = full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))
    print(f"{gen_name} α={alpha}: AUC={m['auc_roc']:.4f}", flush=True)
    all_rows.append({"seed": SEED, "method": gen_name, "condition": "augmented", "alpha": alpha, **m})
    pd.DataFrame(all_rows).to_csv(OUT, index=False)
    del clf, df_syn, X_aug, y_aug
    gc.collect()

print("Done.")
