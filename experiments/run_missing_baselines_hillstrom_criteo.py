"""
Adds the cheap baselines flagged as missing by the original review (R1):
ADASYN, Borderline-SMOTE, and random majority undersampling. class_weight=
'balanced' (cost-sensitive reweighting) was already evaluated elsewhere
(§4.4) — this script fills the remaining gap.

ADASYN and Borderline-SMOTE follow the same 5-value alpha sweep as the
other generators (they produce additional synthetic minority rows, same
as SMOTE). Random undersampling is structurally different (it removes
majority rows rather than adding minority ones), so it is evaluated once
at the standard 1:1 balanced ratio, parallel to how class_weight='balanced'
was evaluated as a single configuration rather than swept.

Unlike generate_smote() in synthetic_data_eval.py (which hardcodes
random_state=RANDOM_STATE=42 for every seed — a real bug found while
auditing this codebase, not fixed retroactively here since it would
invalidate already-published SMOTE numbers), this script passes the
per-iteration seed explicitly to every imbalanced-learn call, so each
of the 5 seeds gets genuinely independent resampling stochasticity.

Output: results/missing_baselines_hillstrom.csv, results/missing_baselines_criteo.csv
"""
import sys, warnings, gc
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
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from experiments.synthetic_data_eval import RESULTS_DIR
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

SEEDS = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
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


def eval_clf(X_tr, y_tr, X_te, y_te, seed):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr)
    return full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))


def run_dataset(df_full, target, name, out_csv):
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    n_use = min(N_CAP, len(df_full))
    rows = []

    for seed in SEEDS:
        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
        X_tr_df = df_train.drop(columns=[target])
        y_tr = df_train[target]
        X_te = df_test.drop(columns=[target]).values.astype(float)
        y_te = df_test[target].values
        X_tr = X_tr_df.values.astype(float)
        minority_n = int(y_tr.sum())

        m = eval_clf(X_tr, y_tr.values, X_te, y_te, seed)
        print(f"  seed={seed} Baseline: AUC={m['auc_roc']:.4f}", flush=True)
        rows.append({"seed": seed, "method": "Baseline", "alpha": 0, **m})

        # ADASYN, Borderline-SMOTE: alpha sweep, same target-minority framing as generate_smote
        for method_name, cls in [("ADASYN", ADASYN), ("BorderlineSMOTE", BorderlineSMOTE)]:
            for alpha in ALPHAS:
                n_synthetic = int(len(df_train) * alpha)
                target_minority = minority_n + n_synthetic
                try:
                    kw = {"sampling_strategy": {1: target_minority}, "random_state": seed}
                    if cls is ADASYN:
                        kw["n_neighbors"] = min(5, minority_n - 1)
                    else:
                        kw["k_neighbors"] = min(5, minority_n - 1)
                    sampler = cls(**kw)
                    X_res, y_res = sampler.fit_resample(X_tr, y_tr.values)
                    m = eval_clf(X_res, y_res, X_te, y_te, seed)
                    print(f"  seed={seed} {method_name} α={alpha}: AUC={m['auc_roc']:.4f}", flush=True)
                    rows.append({"seed": seed, "method": method_name, "alpha": alpha, **m})
                except Exception as e:
                    print(f"  seed={seed} {method_name} α={alpha} FAILED: {e}", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)

        # Random majority undersampling: single 1:1 balanced configuration (parallel to class_weight='balanced')
        try:
            rus = RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
            X_res, y_res = rus.fit_resample(X_tr, y_tr.values)
            m = eval_clf(X_res, y_res, X_te, y_te, seed)
            print(f"  seed={seed} RandomUnderSampler (1:1): AUC={m['auc_roc']:.4f}", flush=True)
            rows.append({"seed": seed, "method": "RandomUnderSampler", "alpha": "1:1", **m})
        except Exception as e:
            print(f"  seed={seed} RandomUnderSampler FAILED: {e}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        gc.collect()

    print(f"\nSaved: {out_csv}", flush=True)


if __name__ == "__main__":
    df_hill, hill_target, hill_task, hill_name = load_hillstrom()
    run_dataset(df_hill, hill_target, hill_name, RESULTS_DIR / "missing_baselines_hillstrom.csv")

    df_crit, crit_target, crit_task, crit_name = load_criteo_uplift()
    run_dataset(df_crit, crit_target, crit_name, RESULTS_DIR / "missing_baselines_criteo.csv")

    print("\nDone.")
