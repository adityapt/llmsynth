"""
Nomao sparse/small-n experiment.
Simulates real marketing conditions: sparse features + limited labeled data.
Tests whether synthetic augmentation recovers lost performance.
"""
import sys
sys.path.insert(0, '.')
from experiments.synthetic_data_eval import (
    run_experiment, run_low_data_experiment,
    plot_ucurve, plot_low_data,
    DATA_DIR, RESULTS_DIR, RANDOM_STATE,
    generate_ctgan, generate_gaussian_copula, generate_smote,
    train_evaluate
)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

SPARSITY = 0.70   # fraction of feature values to zero out
N_SMALL  = 500    # small dataset size
ALPHAS   = [0.1, 0.2, 0.3, 0.5, 1.0]
CV_FOLDS = 5

def load_nomao():
    path = DATA_DIR / "nomao.csv"
    df = pd.read_csv(path)
    return df, "target", "classification", "Nomao Lead (Sparse)"

def apply_sparsity(df, target, sparsity, seed=42):
    """Randomly zero out `sparsity` fraction of feature values."""
    rng = np.random.default_rng(seed)
    df_sparse = df.copy()
    feature_cols = [c for c in df.columns if c != target]
    mask = rng.random((len(df_sparse), len(feature_cols))) < sparsity
    df_sparse[feature_cols] = df_sparse[feature_cols].values * (1 - mask)
    return df_sparse

def cv_score(df_train, df_test, target, task, generators):
    """Single train/test split evaluation across generators."""
    results = []
    X_train = df_train.drop(columns=[target]).values.astype(float)
    y_train = df_train[target].values
    X_test  = df_test.drop(columns=[target]).values.astype(float)
    y_test  = df_test[target].values

    # Baseline
    m = train_evaluate(X_train, y_train, X_test, y_test, task)
    results.append({"method": "Baseline", "condition": "real_only", "alpha": 0, **m})

    for gen_name, gen_fn in generators.items():
        # TSTR
        if gen_name != "SMOTE":
            try:
                n_syn = len(df_train)
                df_syn = gen_fn(df_train, n_syn)
                X_s = df_syn.drop(columns=[target]).values.astype(float)
                y_s = df_syn[target].values
                m = train_evaluate(X_s, y_s, X_test, y_test, task)
                results.append({"method": gen_name, "condition": "synthetic_only", "alpha": 0, **m})
            except Exception as e:
                print(f"    TSTR failed ({gen_name}): {e}")

        # Augmentation sweep
        for alpha in ALPHAS:
            n_syn = int(len(df_train) * alpha)
            try:
                if gen_name == "SMOTE":
                    X_aug, y_aug = generate_smote(X_train, y_train, n_syn)
                else:
                    df_syn = gen_fn(df_train, n_syn)
                    X_syn = df_syn.drop(columns=[target]).values.astype(float)
                    y_syn = df_syn[target].values
                    X_aug = np.vstack([X_train, X_syn])
                    y_aug = np.concatenate([y_train, y_syn])
                m = train_evaluate(X_aug, y_aug, X_test, y_test, task)
                results.append({"method": gen_name, "condition": "augmented", "alpha": alpha, **m})
                print(f"      {gen_name} α={alpha}: AUC={m['auc_roc']:.4f}")
            except Exception as e:
                print(f"      {gen_name} α={alpha} failed: {e}")

    return pd.DataFrame(results)

# ── Load & prepare ──────────────────────────────────────────
df_full, target, task, name = load_nomao()
print(f"Full dataset: {df_full.shape}, positive rate: {df_full[target].mean()*100:.1f}%")

# Subsample to small n
df_small = df_full.sample(N_SMALL, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Subsampled to n={N_SMALL}")

# Apply sparsity
df_sparse = apply_sparsity(df_small, target, SPARSITY)
n_zero = (df_sparse.drop(columns=[target]) == 0).sum().sum()
total   = df_sparse.drop(columns=[target]).size
print(f"Sparsity applied: {n_zero/total*100:.1f}% of feature values zeroed")
print(f"Positive rate: {df_sparse[target].mean()*100:.1f}%")

# Train/test split (80/20)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_sparse, test_size=0.2,
                                      random_state=RANDOM_STATE,
                                      stratify=df_sparse[target])
print(f"Train: {len(df_train)}, Test: {len(df_test)}")

# Dense baseline (full features, no sparsity) for reference
df_train_dense, df_test_dense = train_test_split(df_small, test_size=0.2,
                                                   random_state=RANDOM_STATE,
                                                   stratify=df_small[target])
X_td = df_train_dense.drop(columns=[target]).values.astype(float)
y_td = df_train_dense[target].values
X_te = df_test_dense.drop(columns=[target]).values.astype(float)
y_te = df_test_dense[target].values
dense_baseline = train_evaluate(X_td, y_td, X_te, y_te, task)
print(f"\nDense baseline (n={N_SMALL}, no sparsity): AUC={dense_baseline['auc_roc']:.4f}")

# Sparse baseline (no augmentation)
X_tr = df_train.drop(columns=[target]).values.astype(float)
y_tr = df_train[target].values
X_te2 = df_test.drop(columns=[target]).values.astype(float)
y_te2 = df_test[target].values
sparse_baseline = train_evaluate(X_tr, y_tr, X_te2, y_te2, task)
print(f"Sparse baseline (n={N_SMALL}, {SPARSITY*100:.0f}% sparse): AUC={sparse_baseline['auc_roc']:.4f}")
print(f"Sparsity cost: {(sparse_baseline['auc_roc'] - dense_baseline['auc_roc'])*100:+.2f} AUC pts\n")

# Generators
generators = {
    "GaussianCopula": lambda df, n: generate_gaussian_copula(df, target, n, task),
    "CTGAN":          lambda df, n: generate_ctgan(df, target, n, task),
    "SMOTE":          None,
}

# Run
print("=" * 60)
print(f"Running augmentation sweep on sparse n={N_SMALL} dataset...")
print("=" * 60)

all_rows = []
for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
    print(f"\n  [{gen_name}]")
    for alpha in ALPHAS:
        n_syn = int(len(df_train) * alpha)
        try:
            if gen_name == "SMOTE":
                X_aug, y_aug = generate_smote(X_tr, y_tr, n_syn)
                m = train_evaluate(X_aug, y_aug, X_te2, y_te2, task)
            else:
                gen_fn = generators[gen_name]
                df_syn = gen_fn(df_train, n_syn)
                X_syn = df_syn.drop(columns=[target]).values.astype(float)
                y_syn = df_syn[target].values
                X_aug = np.vstack([X_tr, X_syn])
                y_aug = np.concatenate([y_tr, y_syn])
                m = train_evaluate(X_aug, y_aug, X_te2, y_te2, task)
            print(f"    α={alpha}: AUC={m['auc_roc']:.4f}  F1={m['f1_minority']:.4f}")
            all_rows.append({"method": gen_name, "alpha": alpha, **m})
        except Exception as e:
            print(f"    α={alpha} failed: {e}")

# TSTR
print("\n  [TSTR — synthetic only]")
for gen_name in ["GaussianCopula", "CTGAN"]:
    try:
        gen_fn = generators[gen_name]
        df_syn = gen_fn(df_train, len(df_train))
        X_syn = df_syn.drop(columns=[target]).values.astype(float)
        y_syn = df_syn[target].values
        m = train_evaluate(X_syn, y_syn, X_te2, y_te2, task)
        print(f"  {gen_name} TSTR: AUC={m['auc_roc']:.4f}")
        all_rows.append({"method": f"{gen_name}_TSTR", "alpha": 0, **m})
    except Exception as e:
        print(f"  {gen_name} TSTR failed: {e}")

df_results = pd.DataFrame(all_rows)
df_results.to_csv(RESULTS_DIR / "metrics_nomao_sparse.csv", index=False)

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"GaussianCopula": "#2196F3", "CTGAN": "#FF5722", "SMOTE": "#4CAF50"}
for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
    sub = df_results[df_results["method"] == gen_name].sort_values("alpha")
    if sub.empty:
        continue
    ax.plot(sub["alpha"], sub["auc_roc"], marker="o", label=gen_name,
            color=colors.get(gen_name), linewidth=2)

ax.axhline(dense_baseline["auc_roc"], linestyle="--", color="black",
           linewidth=1.5, label=f"Dense baseline ({dense_baseline['auc_roc']:.3f})")
ax.axhline(sparse_baseline["auc_roc"], linestyle=":", color="gray",
           linewidth=1.5, label=f"Sparse baseline ({sparse_baseline['auc_roc']:.3f})")
ax.set_xlabel("Synthetic fraction α (n_synthetic / n_real)")
ax.set_ylabel("AUC-ROC")
ax.set_title(f"Nomao Lead — Sparse Features ({int(SPARSITY*100)}% zeroed) + Small n={N_SMALL}\nDoes synthetic augmentation recover performance?")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ucurve_nomao_sparse.png", dpi=150)
plt.close()
print(f"\n  Saved: results/ucurve_nomao_sparse.png")

# ── Summary ─────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Dense baseline (no sparsity, n={N_SMALL}):  AUC = {dense_baseline['auc_roc']:.4f}")
print(f"Sparse baseline (70% missing, n={N_SMALL}): AUC = {sparse_baseline['auc_roc']:.4f}  [{(sparse_baseline['auc_roc']-dense_baseline['auc_roc'])*100:+.2f} pts]")
print()
for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
    sub = df_results[df_results["method"] == gen_name]
    if sub.empty:
        continue
    best_idx = sub["auc_roc"].idxmax()
    best = sub.loc[best_idx]
    recovery = (best["auc_roc"] - sparse_baseline["auc_roc"]) / \
               (dense_baseline["auc_roc"] - sparse_baseline["auc_roc"]) * 100
    print(f"{gen_name}: best AUC={best['auc_roc']:.4f} at α={best['alpha']}  "
          f"gain={best['auc_roc']-sparse_baseline['auc_roc']:+.4f}  "
          f"recovery={recovery:.1f}%")
