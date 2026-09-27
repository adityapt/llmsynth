"""
Full metrics (F1/AP/Precision/Recall/Accuracy, alongside AUC) for the
class_weight='balanced' cost-sensitive baseline on Hillstrom/Criteo.
No synthetic generator involved -- just sample-weight reweighting -- so
this is fast (no CTGAN/torch dependency at all).

Output: results/fullmetrics_balanced_hillstrom.csv, results/fullmetrics_balanced_criteo.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)

from experiments.synthetic_data_eval import RESULTS_DIR
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

SEEDS = [42, 123, 7, 2024, 999]
N_CAP = 10_000


def full_metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }


def run_dataset(df_full, target, name, out_csv):
    print(f"\n=== {name} ===", flush=True)
    n_use = min(N_CAP, len(df_full))
    rows = []
    for seed in SEEDS:
        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
        X_tr = df_train.drop(columns=[target]).values.astype(float)
        y_tr = df_train[target].values
        X_te = df_test.drop(columns=[target]).values.astype(float)
        y_te = df_test[target].values

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
        m = full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))
        print(f"  seed={seed}: AUC={m['auc_roc']:.4f} F1={m['f1_minority']:.4f}", flush=True)
        rows.append({"seed": seed, "method": "class_weight_balanced", **m})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)


if __name__ == "__main__":
    df_hill, hill_target, _, hill_name = load_hillstrom()
    run_dataset(df_hill, hill_target, hill_name, RESULTS_DIR / "fullmetrics_balanced_hillstrom.csv")

    df_crit, crit_target, _, crit_name = load_criteo_uplift()
    run_dataset(df_crit, crit_target, crit_name, RESULTS_DIR / "fullmetrics_balanced_criteo.csv")

    print("\nDone.")
