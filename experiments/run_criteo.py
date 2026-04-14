"""
Criteo Uplift Modeling Dataset — synthetic data augmentation experiment.

Real display advertising campaign data from Criteo (2014).
~13.9M rows; treatment/control split; target = conversion (~2.9% positive).
Severe class imbalance, real-world ad click/conversion signal.

We cap to N_CAP rows for tractability and predict conversion propensity
from behavioral features (f0–f11) — the standard setup for augmentation testing.

Reference: https://ailab.criteo.com/criteo-uplift-modeling-dataset/
           Diemert et al. (2018), AdKDD Workshop.
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
import matplotlib.pyplot as plt

ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP = 10000   # 13.9M rows total; cap for tractable generation


def load_criteo_uplift():
    path = DATA_DIR / "criteo_uplift.csv"
    if not path.exists():
        print("  Downloading Criteo Uplift dataset (~300MB, may take a moment)...")
        try:
            # Primary: Criteo AI Lab
            url = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
            df = pd.read_csv(url, compression="gzip")
        except Exception:
            try:
                url2 = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
                df = pd.read_csv(url2, compression="gzip")
            except Exception as e:
                raise RuntimeError(
                    f"Could not download Criteo dataset: {e}\n"
                    "Please download manually from https://ailab.criteo.com/criteo-uplift-modeling-dataset/ "
                    f"and save as {path}"
                )

        df.columns = [c.lower().strip() for c in df.columns]

        # Features: f0–f11 (anonymized behavioral features)
        # Target: conversion (binary), treatment: exposure flag
        # For augmentation experiment: predict conversion from features only
        # Drop treatment column to avoid leakage
        drop_cols = [c for c in ["treatment", "visit", "exposure"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        if "conversion" in df.columns:
            df = df.rename(columns={"conversion": "target"})
        else:
            df = df.rename(columns={df.columns[-1]: "target"})

        df["target"] = df["target"].astype(int)

        # Save full (capped to 500K to keep file manageable)
        df.sample(min(500_000, len(df)), random_state=RANDOM_STATE).to_csv(path, index=False)
        print(f"  Saved {min(500_000, len(df)):,} rows to {path}")

    df = pd.read_csv(path, nrows=N_CAP * 5)   # read a chunk, then sample
    df = df.sample(min(N_CAP * 5, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.dropna()
    print(f"  Shape after load: {df.shape}, positive rate: {df['target'].mean()*100:.1f}%")
    return df, "target", "classification", "Criteo Uplift"


if __name__ == "__main__":
    # ── Load ────────────────────────────────────────────────────
    print("=" * 60)
    df_full, target, task, name = load_criteo_uplift()
    print(f"Loaded {name}: {df_full.shape}")

    # Cap
    df = df_full.sample(N_CAP, random_state=RANDOM_STATE).reset_index(drop=True)
    pos_rate = df[target].mean()
    print(f"Capped to n={N_CAP}, positive rate: {pos_rate*100:.1f}%")

    if pos_rate < 0.01:
        print("  WARNING: Very low positive rate (<1%). SMOTE may be unstable at small k.")

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
    print(f"\nBaseline (real only, n={len(df_train)}): AUC={baseline['auc_roc']:.4f}  AP={baseline.get('avg_precision', float('nan')):.4f}")

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
                    minority_count = y_tr.sum()
                    if minority_count < 6:
                        print(f"    α={alpha}: skipped (too few minority samples for SMOTE)")
                        continue
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
    df_results.to_csv(RESULTS_DIR / "metrics_criteo.csv", index=False)

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
    ax.set_title(f"Criteo Uplift — Display Ad Conversion Propensity\nn={N_CAP}, {pos_rate*100:.1f}% positive rate (severe imbalance)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ucurve_criteo.png", dpi=150)
    plt.close()
    print("\n  Saved: results/ucurve_criteo.png")

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
