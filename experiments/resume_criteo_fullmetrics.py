"""
Resumes run_full_metrics_hillstrom_criteo.py for Criteo only — the original
background run was OOM-killed partway through seed 7 (seeds 42 and 123
completed cleanly and are kept; seed 7 is redone from scratch since it was
only partially written). Adds explicit garbage collection between calls,
since CTGAN/SDV's internal PyTorch objects were the likely source of the
memory growth that triggered the OOM kill.
"""
import sys, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from experiments.synthetic_data_eval import RESULTS_DIR, generate_ctgan, generate_gaussian_copula, generate_smote
from experiments.run_criteo import load_criteo_uplift

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)

SEEDS_REMAINING = [7, 2024, 999]  # 42, 123 already completed and preserved
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
existing = existing[~existing["seed"].isin(SEEDS_REMAINING)]  # drop partial seed 7 rows, keep 42/123
all_rows = existing.to_dict("records")
pd.DataFrame(all_rows).to_csv(OUT, index=False)
print(f"Kept {len(all_rows)} rows from seeds {sorted(existing['seed'].unique())}", flush=True)

n_use = min(N_CAP, len(df_full))
for seed in SEEDS_REMAINING:
    print(f"\n  Seed {seed}:", flush=True)
    np.random.seed(seed)
    df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(df_train.drop(columns=[target]).values.astype(float), df_train[target].values)
    m = full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))
    print(f"    Baseline: AUC={m['auc_roc']:.4f}", flush=True)
    all_rows.append({"seed": seed, "method": "Baseline", "condition": "real_only", "alpha": 0, **m})
    pd.DataFrame(all_rows).to_csv(OUT, index=False)
    del clf; gc.collect()

    for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
        for alpha in ALPHAS:
            try:
                X_tr = df_train.drop(columns=[target])
                y_tr = df_train[target]
                n_syn = int(len(df_train) * alpha)
                if gen_name == "SMOTE":
                    df_syn = generate_smote(X_tr, y_tr, n_syn)
                else:
                    gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
                    df_syn = gen_fn(df_train, target, n_syn, task)
                X_aug = np.vstack([X_tr.values.astype(float), df_syn.drop(columns=[target]).values.astype(float)])
                y_aug = np.concatenate([y_tr.values, df_syn[target].values])

                clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
                clf2.fit(X_aug, y_aug)
                m = full_metrics(y_te, clf2.predict_proba(X_te)[:, 1], clf2.predict(X_te))
                print(f"    {gen_name} α={alpha}: AUC={m['auc_roc']:.4f} F1={m['f1_minority']:.4f} Acc={m['accuracy']:.4f}", flush=True)
                all_rows.append({"seed": seed, "method": gen_name, "condition": "augmented", "alpha": alpha, **m})
                del clf2, df_syn, X_aug, y_aug
            except Exception as e:
                print(f"    {gen_name} α={alpha} failed: {e}", flush=True)
            pd.DataFrame(all_rows).to_csv(OUT, index=False)
            gc.collect()

print(f"\nDone. Saved: {OUT}", flush=True)
