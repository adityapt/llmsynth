"""
Full-metrics rerun (Accuracy/Precision/Recall/F1/AP, alongside AUC) for the
four benchmark datasets (Telco, Bank Marketing, German Credit, Nomao Lead)
-- the original ci_telco_churn.csv/ci_bank_marketing.csv/ci_credit_default.csv/
ci_nomao_lead.csv only ever saved raw AUC, nothing else. This extends the
same 5-seed/alpha-sweep protocol to capture the full metric suite, matching
what already exists for Hillstrom/Criteo (§4.4) and the missing baselines
(§4.10). GaussianCopula/CTGAN/SMOTE only (matching the original §4.2 scope --
ADASYN/Borderline-SMOTE/RandomUnderSampler were introduced specifically for
the marketing datasets in §4.10 and were never part of the §4.2 comparison).

Writes to NEW files so the original (AUC-only, already-verified) CSVs are
untouched: results/fullmetrics_{telco,bank_marketing,german_credit,nomao}.csv

Uses explicit gc.collect() between fits (the same memory-safe pattern that
got the dose-response reruns through on this machine).
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

from experiments.synthetic_data_eval import (
    RESULTS_DIR, generate_ctgan, generate_gaussian_copula, generate_smote,
    load_telco_churn, load_bank_marketing, load_credit_default,
)
from experiments.run_nomao import load_nomao

SEEDS = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP = 15_000


def full_metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }


def run_dataset(df_full, target, task, name, out_csv):
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    n_use = min(N_CAP, len(df_full))
    rows = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["seed"], r["method"], r["alpha"]) for r in rows}
        print(f"  Resuming -- {len(done)} combos already done", flush=True)
    else:
        done = set()

    for seed in SEEDS:
        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
        X_tr_df = df_train.drop(columns=[target])
        y_tr = df_train[target]
        X_te = df_test.drop(columns=[target]).values.astype(float)
        y_te = df_test[target].values
        X_tr = X_tr_df.values.astype(float)

        if (seed, "Baseline", 0) not in done:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
            clf.fit(X_tr, y_tr.values)
            m = full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))
            print(f"  seed={seed} Baseline: AUC={m['auc_roc']:.4f}", flush=True)
            rows.append({"seed": seed, "method": "Baseline", "alpha": 0, **m})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            del clf; gc.collect()

        for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
            for alpha in ALPHAS:
                if (seed, gen_name, alpha) in done:
                    continue
                try:
                    n_syn = int(len(df_train) * alpha)
                    if gen_name == "SMOTE":
                        df_syn = generate_smote(X_tr_df, y_tr, n_syn)
                    else:
                        gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
                        df_syn = gen_fn(df_train, target, n_syn, task)
                    X_aug = np.vstack([X_tr, df_syn.drop(columns=[target]).values.astype(float)])
                    y_aug = np.concatenate([y_tr.values, df_syn[target].values])
                    clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
                    clf2.fit(X_aug, y_aug)
                    m = full_metrics(y_te, clf2.predict_proba(X_te)[:, 1], clf2.predict(X_te))
                    print(f"  seed={seed} {gen_name} α={alpha}: AUC={m['auc_roc']:.4f}", flush=True)
                    rows.append({"seed": seed, "method": gen_name, "alpha": alpha, **m})
                    del clf2, df_syn, X_aug, y_aug
                except Exception as e:
                    print(f"  seed={seed} {gen_name} α={alpha} FAILED: {e}", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                gc.collect()

    print(f"\n  Saved: {out_csv}", flush=True)


if __name__ == "__main__":
    datasets = [
        (*load_telco_churn(), "fullmetrics_telco.csv"),
        (*load_bank_marketing(), "fullmetrics_bank_marketing.csv"),
        (*load_credit_default(), "fullmetrics_german_credit.csv"),
        (*load_nomao(), "fullmetrics_nomao.csv"),
    ]
    for df_full, target, task, name, out_name in datasets:
        run_dataset(df_full, target, task, name, RESULTS_DIR / out_name)
    print("\nAll done.")
