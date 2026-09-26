"""
Full-metrics rerun for Hillstrom and Criteo (GaussianCopula/CTGAN/SMOTE).

Adds Accuracy, Precision, and Recall alongside the existing AUC-ROC, F1
(minority), and Average Precision — none of these three were previously
saved, since the original harness (run_confidence_intervals.py) only
computed metrics available from clf.predict()/predict_proba() at the time,
without capturing Accuracy/Precision/Recall explicitly.

Writes to NEW files (does not overwrite the already-verified
results/ci_hillstrom.csv / results/ci_criteo.csv) so the existing audited
numbers remain untouched. Same seeds, same alphas, same generator calls —
only the metrics computed on the already-existing prediction arrays are
extended, so AUC-ROC/F1/AP here should closely match the original files
(same protocol, freshly regenerated synthetic data — expect only the
generator-fit stochasticity already present in the original runs, not a
new source of variance).

Output:
  results/ci_hillstrom_fullmetrics.csv
  results/ci_criteo_fullmetrics.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from experiments.synthetic_data_eval import (
    RESULTS_DIR,
    generate_ctgan, generate_gaussian_copula, generate_smote,
)
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)

SEEDS  = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP  = 10_000


def full_metrics(y_true, proba, preds):
    return {
        "auc_roc":       roc_auc_score(y_true, proba),
        "f1_minority":   f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision":     precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall":        recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy":      accuracy_score(y_true, preds),
    }


def augment_and_eval(df_train, df_test, target, task, gen_name, alpha, seed):
    X_tr = df_train.drop(columns=[target])
    y_tr = df_train[target]
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values
    n_syn = int(len(df_train) * alpha)

    if gen_name == "SMOTE":
        df_syn = generate_smote(X_tr, y_tr, n_syn)
        X_aug = np.vstack([X_tr.values.astype(float),
                           df_syn.drop(columns=[target]).values.astype(float)])
        y_aug = np.concatenate([y_tr.values, df_syn[target].values])
    else:
        gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
        df_syn = gen_fn(df_train, target, n_syn, task)
        X_syn  = df_syn.drop(columns=[target]).values.astype(float)
        y_syn  = df_syn[target].values
        X_aug  = np.vstack([X_tr.values.astype(float), X_syn])
        y_aug  = np.concatenate([y_tr.values, y_syn])

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_aug, y_aug)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return full_metrics(y_te, proba, preds)


def baseline_eval(df_train, df_test, target, seed):
    X_tr = df_train.drop(columns=[target]).values.astype(float)
    y_tr = df_train[target].values
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return full_metrics(y_te, proba, preds)


def run_experiment(df_full, target, task, name, generators, n_cap=N_CAP, out_csv=None):
    print(f"\n{'='*60}\nFull-metrics rerun: {name} ({len(SEEDS)} seeds)\n{'='*60}", flush=True)
    n_use = min(n_cap, len(df_full))
    all_rows = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        np.random.seed(seed)
        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[target]
        )

        m = baseline_eval(df_train, df_test, target, seed)
        print(f"    Baseline: AUC={m['auc_roc']:.4f} F1={m['f1_minority']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f}", flush=True)
        all_rows.append({"seed": seed, "method": "Baseline",
                         "condition": "real_only", "alpha": 0, **m})
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)

        for gen_name in generators:
            for alpha in ALPHAS:
                try:
                    m = augment_and_eval(df_train, df_test, target, task, gen_name, alpha, seed)
                    print(f"    {gen_name} α={alpha}: AUC={m['auc_roc']:.4f} F1={m['f1_minority']:.4f} "
                          f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f}", flush=True)
                    all_rows.append({"seed": seed, "method": gen_name,
                                     "condition": "augmented", "alpha": alpha, **m})
                except Exception as e:
                    print(f"    {gen_name} α={alpha} failed: {e}", flush=True)
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    print(f"\n  Saved: {out_csv}", flush=True)
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df_hill, hill_target, hill_task, hill_name = load_hillstrom()
    run_experiment(df_hill, hill_target, hill_task, hill_name,
                   generators=["GaussianCopula", "CTGAN", "SMOTE"],
                   out_csv=RESULTS_DIR / "ci_hillstrom_fullmetrics.csv")

    df_crit, crit_target, crit_task, crit_name = load_criteo_uplift()
    run_experiment(df_crit, crit_target, crit_task, crit_name,
                   generators=["GaussianCopula", "CTGAN", "SMOTE"],
                   out_csv=RESULTS_DIR / "ci_criteo_fullmetrics.csv")

    print("\nDone.")
