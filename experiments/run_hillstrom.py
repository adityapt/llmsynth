"""
Hillstrom Email Marketing Dataset — synthetic data augmentation experiment.

Kevin Hillstrom's MineThatData Email Analytics Challenge (2008).
64,000 customers, email campaign A/B/holdout, target = conversion (purchase).
Positive rate ~9.2%. Classic "real" marketing dataset for campaign response modeling.

Reference: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
"""
import sys
sys.path.insert(0, '.')
from experiments.synthetic_data_eval import (
    DATA_DIR, RESULTS_DIR, RANDOM_STATE,
    generate_ctgan, generate_gaussian_copula, generate_smote,
    train_evaluate
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP = 10000   # cap for tractable CTGAN; ~15% of full dataset


def load_hillstrom():
    path = DATA_DIR / "hillstrom.csv"
    if not path.exists():
        mirrors = [
            "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
            "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        ]
        df = None
        for url in mirrors:
            try:
                df = pd.read_csv(url)
                if len(df) > 1000:   # real dataset has 64K rows
                    print(f"  Downloaded from {url}")
                    break
            except Exception:
                continue

        if df is None or len(df) < 1000:
            raise RuntimeError(
                "Could not download Hillstrom dataset from any mirror.\n"
                "Download manually from: https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv\n"
                f"and save as {path}"
            )

        # Standard Hillstrom column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Target: conversion (purchase)
        if "conversion" in df.columns:
            target_col = "conversion"
        elif "purchase" in df.columns:
            target_col = "purchase"
        else:
            target_col = df.columns[-1]

        df = df.rename(columns={target_col: "target"})
        df["target"] = df["target"].astype(int)

        # Drop segment/treatment columns to predict conversion from features only
        # (we're testing augmentation for propensity modeling, not uplift)
        drop_cols = [c for c in ["segment", "visit", "spend"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Encode categoricals
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) == "category":
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))

        df.to_csv(path, index=False)
        print(f"  Saved to {path}")

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}, positive rate: {df['target'].mean()*100:.1f}%")
    return df, "target", "classification", "Hillstrom Email"


if __name__ == "__main__":
    # ── Load ────────────────────────────────────────────────────
    print("=" * 60)
    df_full, target, task, name = load_hillstrom()
    print(f"Loaded {name}: {df_full.shape}")

    # Cap for tractability
    n_use = min(N_CAP, len(df_full))
    df = df_full.sample(n_use, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Using n={n_use}, positive rate: {df[target].mean()*100:.1f}%")

    # Train/test split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[target]
    )
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    X_tr = df_train.drop(columns=[target])
    y_tr = df_train[target]
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values

    # Baseline
    baseline = train_evaluate(X_tr.values.astype(float), y_tr.values, X_te, y_te, task)
    print(f"\nBaseline (real only, n={len(df_train)}): AUC={baseline['auc_roc']:.4f}  F1={baseline['f1_minority']:.4f}")

    # ── Augmentation sweep ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Augmentation sweep...")
    print("=" * 60)

    all_rows = [{"method": "Baseline", "condition": "real_only", "alpha": 0, **baseline}]

    for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
        print(f"\n  [{gen_name}]")

        # TSTR
        if gen_name != "SMOTE":
            try:
                if gen_name == "GaussianCopula":
                    df_syn = generate_gaussian_copula(df_train, target, len(df_train), task)
                else:
                    df_syn = generate_ctgan(df_train, target, len(df_train), task)
                X_s = df_syn.drop(columns=[target]).values.astype(float)
                y_s = df_syn[target].values
                m = train_evaluate(X_s, y_s, X_te, y_te, task)
                print(f"    TSTR: AUC={m['auc_roc']:.4f}  F1={m['f1_minority']:.4f}")
                all_rows.append({"method": gen_name, "condition": "synthetic_only", "alpha": 0, **m})
            except Exception as e:
                print(f"    TSTR failed: {e}")

        # Augmentation sweep
        for alpha in ALPHAS:
            n_syn = int(len(df_train) * alpha)
            try:
                if gen_name == "SMOTE":
                    df_syn = generate_smote(X_tr, y_tr, n_syn)
                    X_aug = np.vstack([X_tr.values.astype(float), df_syn.drop(columns=[target]).values.astype(float)])
                    y_aug = np.concatenate([y_tr.values, df_syn[target].values])
                    m = train_evaluate(X_aug, y_aug, X_te, y_te, task)
                elif gen_name == "GaussianCopula":
                    df_syn = generate_gaussian_copula(df_train, target, n_syn, task)
                    X_syn = df_syn.drop(columns=[target]).values.astype(float)
                    y_syn = df_syn[target].values
                    X_aug = np.vstack([X_tr.values.astype(float), X_syn])
                    y_aug = np.concatenate([y_tr.values, y_syn])
                    m = train_evaluate(X_aug, y_aug, X_te, y_te, task)
                else:
                    df_syn = generate_ctgan(df_train, target, n_syn, task)
                    X_syn = df_syn.drop(columns=[target]).values.astype(float)
                    y_syn = df_syn[target].values
                    X_aug = np.vstack([X_tr.values.astype(float), X_syn])
                    y_aug = np.concatenate([y_tr.values, y_syn])
                    m = train_evaluate(X_aug, y_aug, X_te, y_te, task)
                print(f"    α={alpha}: AUC={m['auc_roc']:.4f}  F1={m['f1_minority']:.4f}")
                all_rows.append({"method": gen_name, "condition": "augmented", "alpha": alpha, **m})
            except Exception as e:
                print(f"    α={alpha} failed: {e}")

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(RESULTS_DIR / "metrics_hillstrom.csv", index=False)

    # ── Plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"GaussianCopula": "#2196F3", "CTGAN": "#FF5722", "SMOTE": "#4CAF50"}
    for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
        sub = df_results[(df_results["method"] == gen_name) &
                         (df_results["condition"] == "augmented")].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub["auc_roc"], marker="o", label=gen_name,
                color=colors[gen_name], linewidth=2)

    ax.axhline(baseline["auc_roc"], linestyle="--", color="black",
               linewidth=1.5, label=f"Baseline ({baseline['auc_roc']:.3f})")
    ax.set_xlabel("Synthetic fraction α")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Hillstrom Email Marketing — Conversion Propensity\nn={N_CAP}, {df[target].mean()*100:.1f}% positive rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ucurve_hillstrom.png", dpi=150)
    plt.close()
    print("\n  Saved: results/ucurve_hillstrom.png")

    # ── Summary ─────────────────────────────────────────────────
    print("\n=== RESULTS SUMMARY ===")
    print(f"Baseline (real only): AUC={baseline['auc_roc']:.4f}  F1={baseline['f1_minority']:.4f}")
    print()
    for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
        aug = df_results[(df_results["method"] == gen_name) & (df_results["condition"] == "augmented")]
        tstr = df_results[(df_results["method"] == gen_name) & (df_results["condition"] == "synthetic_only")]
        if aug.empty:
            continue
        best_idx = aug["auc_roc"].idxmax()
        best = aug.loc[best_idx]
        tstr_auc = tstr["auc_roc"].values[0] if not tstr.empty else float("nan")
        print(f"{gen_name}:")
        print(f"  TSTR:     AUC={tstr_auc:.4f}")
        print(f"  Best aug: AUC={best['auc_roc']:.4f} at α={best['alpha']}  gain={best['auc_roc']-baseline['auc_roc']:+.4f}")
