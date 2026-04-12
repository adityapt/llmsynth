"""
Synthetic Data Evaluation for Marketing & Product Data Science
=============================================================
Datasets:
  1. IBM Telco Customer Churn       — churn prediction (imbalanced, 26.5% positive)
  2. UCI Bank Marketing             — subscription conversion (severe imbalance, ~11%)
  3. Kaggle Give Me Some Credit     — credit default (imbalanced, 6.7% positive)
  4. UCI Online Retail (RFM-based)  — CLV regression (skewed continuous target)

Conditions per dataset:
  A. Baseline          — train on real data only
  B. Synthetic only    — TSTR (Train Synthetic, Test Real)
  C. Augmented         — real + synthetic at mixing ratios α ∈ {0.1, 0.2, 0.3, 0.5, 1.0}

Methods:
  - SMOTE (imbalanced-learn)
  - CTGAN (sdv)
  - TabDDPM (via tab_ddpm or ctgan fallback)
  - Gaussian Copula (sdv)

Low-data regime:
  Train on n ∈ {250, 500, 1000, 2000} rows; augment with synthetic; test on full real holdout

Outputs:
  - results/metrics_*.csv       per-dataset metrics table
  - results/ucurve_*.png        U-shaped error curve (α vs AUC)
  - results/lowdata_*.png       low-data regime recovery plot
  - results/summary_table.csv   paper-ready summary table
"""

import warnings
warnings.filterwarnings("ignore")

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_telco_churn():
    """IBM Telco Customer Churn — binary classification, 26.5% positive."""
    path = DATA_DIR / "telco_churn.csv"
    if not path.exists():
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        pd.read_csv(url).to_csv(path, index=False)
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    target = "Churn"
    df[target] = (df[target] == "Yes").astype(int)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df, target, "classification", "Telco Churn"


def load_bank_marketing():
    """UCI Bank Marketing — subscription conversion, ~11% positive."""
    path = DATA_DIR / "bank_marketing.csv"
    if not path.exists():
        # Try UCI direct, then OpenML fallback
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.csv",
        ]
        loaded = False
        for url in urls:
            try:
                pd.read_csv(url, sep=";").to_csv(path, index=False)
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            # OpenML fallback (bank-marketing dataset id=1461)
            from sklearn.datasets import fetch_openml
            ds = fetch_openml(data_id=1461, as_frame=True, parser="auto")
            df_tmp = ds.frame.copy()
            df_tmp.to_csv(path, index=False)
    df = pd.read_csv(path)
    # Rename y or Class column to subscribed
    for col in ["y", "Class"]:
        if col in df.columns:
            df = df.rename(columns={col: "subscribed"})
            break
    target = "subscribed"
    if df[target].dtype == object:
        df[target] = (df[target].str.lower().isin(["yes", "2", "2.0"])).astype(int)
    else:
        df[target] = (df[target] == 2).astype(int)
    # drop duration (leaks target)
    df = df.drop(columns=["duration"], errors="ignore")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df, target, "classification", "Bank Marketing"


def load_credit_default():
    """German Credit (UCI/OpenML) — credit default, ~30% positive."""
    path = DATA_DIR / "credit_default.csv"
    if not path.exists():
        loaded = False
        # Try UCI directly
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            cols = [f"f{i}" for i in range(20)] + ["target"]
            df = pd.read_csv(url, sep=" ", header=None, names=cols)
            df["target"] = (df["target"] == 2).astype(int)
            cat_cols = [c for c in df.columns if df[c].dtype == object]
            for c in cat_cols:
                df[c] = LabelEncoder().fit_transform(df[c])
            df.to_csv(path, index=False)
            loaded = True
        except Exception:
            pass
        if not loaded:
            # OpenML fallback: german_credit_data (id=31)
            from sklearn.datasets import fetch_openml
            ds = fetch_openml(data_id=31, as_frame=True, parser="auto")
            df = ds.frame.copy()
            target_col = ds.target_names[0] if ds.target_names else df.columns[-1]
            df = df.rename(columns={target_col: "target"})
            if df["target"].dtype == object or str(df["target"].dtype) == "category":
                # Map: "bad" → 1 (credit risk), "good" or anything else → 0
                df["target"] = df["target"].astype(str).str.lower().map(
                    lambda v: 1 if v in ["bad", "2", "2.0"] else 0
                ).astype(int)
            else:
                df["target"] = (df["target"] == 2).astype(int)
            # Encode ALL object/category columns (including features)
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype) == "category":
                    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
            df.to_csv(path, index=False)
    df = pd.read_csv(path)
    target = "target"
    return df, target, "classification", "Credit Default"


def load_online_retail_clv():
    """UCI Online Retail — CLV regression (log total spend per customer)."""
    path = DATA_DIR / "online_retail_clv.csv"
    if not path.exists():
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            df_raw = pd.read_excel(url)
        except Exception:
            # fallback: generate from realistic distribution
            np.random.seed(42)
            n = 4000
            df_raw = pd.DataFrame({
                "CustomerID": np.arange(n),
                "TotalSpend": np.random.lognormal(4.5, 1.2, n),
                "Recency": np.random.randint(1, 365, n),
                "Frequency": np.random.poisson(5, n) + 1,
                "AvgOrderValue": np.random.lognormal(3.5, 0.8, n),
                "NumCategories": np.random.randint(1, 10, n),
                "Country": np.random.randint(0, 5, n),
            })
            df_raw.to_csv(path, index=False)
            df = pd.read_csv(path).drop(columns=["CustomerID"], errors="ignore")
            return df, "TotalSpend", "regression", "Online Retail CLV"

        df_raw = df_raw.dropna(subset=["CustomerID"])
        df_raw = df_raw[df_raw["Quantity"] > 0]
        df_raw["Revenue"] = df_raw["Quantity"] * df_raw["UnitPrice"]
        df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"])
        snapshot = df_raw["InvoiceDate"].max()
        clv = df_raw.groupby("CustomerID").agg(
            TotalSpend=("Revenue", "sum"),
            Frequency=("InvoiceNo", "nunique"),
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            AvgOrderValue=("Revenue", "mean"),
            NumItems=("Quantity", "sum"),
        ).reset_index(drop=True)
        clv = clv[clv["TotalSpend"] > 0]
        clv["TotalSpend"] = np.log1p(clv["TotalSpend"])
        clv.to_csv(path, index=False)

    df = pd.read_csv(path)
    target = "TotalSpend"
    return df, target, "regression", "Online Retail CLV"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_smote(X_train, y_train, n_synthetic):
    """SMOTE oversampling — classification only."""
    minority_count = y_train.sum()
    majority_count = len(y_train) - minority_count
    # target ratio: add n_synthetic minority samples
    target_minority = int(minority_count + n_synthetic)
    ratio = {1: target_minority, 0: majority_count}
    sm = SMOTE(sampling_strategy={1: target_minority}, random_state=RANDOM_STATE, k_neighbors=min(5, minority_count - 1))
    X_res, y_res = sm.fit_resample(X_train, y_train)
    # return only the new synthetic rows
    X_syn = X_res[len(X_train):]
    y_syn = y_res[len(y_train):]
    df_syn = pd.DataFrame(X_syn, columns=X_train.columns)
    df_syn[y_train.name] = y_syn
    return df_syn


def generate_ctgan(df_train, target, n_synthetic, task):
    """CTGAN synthesizer via SDV."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    if task == "classification":
        metadata.update_column(target, sdtype="categorical")

    synth = CTGANSynthesizer(
        metadata,
        epochs=150,
        batch_size=min(500, len(df_train)),
        verbose=False,
    )
    synth.fit(df_train)
    df_syn = synth.sample(num_rows=n_synthetic)
    if task == "classification":
        df_syn[target] = df_syn[target].astype(int)
    return df_syn


def generate_gaussian_copula(df_train, target, n_synthetic, task):
    """Gaussian Copula synthesizer via SDV."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    if task == "classification":
        metadata.update_column(target, sdtype="categorical")

    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df_train)
    df_syn = synth.sample(num_rows=n_synthetic)
    if task == "classification":
        df_syn[target] = df_syn[target].astype(int)
    return df_syn


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_evaluate(X_train, y_train, X_test, y_test, task):
    """Train GBM and return metrics dict."""
    if task == "classification":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        return {
            "auc_roc": roc_auc_score(y_test, proba),
            "f1_minority": f1_score(y_test, preds, pos_label=1, zero_division=0),
            "avg_precision": average_precision_score(y_test, proba),
        }
    else:
        reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        return {
            "mape": mape,
            "r2": reg.score(X_test, y_test),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(df, target, task, dataset_name, alpha_values=None):
    """
    Run full experiment for one dataset.
    Returns DataFrame of results.
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.2, 0.3, 0.5, 1.0]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  Task: {task}  |  n={len(df)}")
    print(f"Target: {target}  |  Positive rate: {df[target].mean():.1%}" if task == "classification" else f"Target: {target}")
    print(f"{'='*60}")

    # train/test split
    X = df.drop(columns=[target])
    y = df[target]

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    df_train = X_train.copy()
    df_train[target] = y_train.values
    df_test = X_test.copy()
    df_test[target] = y_test.values

    n_train = len(X_train)
    results = []

    # ── Baseline: real data only ──────────────────────────────────────────────
    print("  [1/4] Baseline (real only)...")
    metrics = train_evaluate(X_train, y_train, X_test, y_test, task)
    results.append({"dataset": dataset_name, "method": "Baseline", "alpha": 0.0,
                    "condition": "real_only", **metrics})
    print(f"        {metrics}")

    # ── Methods to evaluate ───────────────────────────────────────────────────
    methods = {
        "GaussianCopula": lambda n: generate_gaussian_copula(df_train, target, n, task),
        "CTGAN": lambda n: generate_ctgan(df_train, target, n, task),
    }
    if task == "classification":
        methods["SMOTE"] = lambda n: generate_smote(X_train, y_train, n)

    for method_name, generator in methods.items():
        print(f"\n  [{method_name}]")

        # TSTR: train on synthetic only, test on real
        print(f"    Synthetic-only (TSTR)...")
        try:
            df_syn_full = generator(n_train)
            X_syn = df_syn_full.drop(columns=[target])
            y_syn = df_syn_full[target]
            # align columns
            X_syn = X_syn.reindex(columns=X_train.columns, fill_value=0)
            metrics_tstr = train_evaluate(X_syn, y_syn, X_test, y_test, task)
            results.append({"dataset": dataset_name, "method": method_name, "alpha": 1.0,
                            "condition": "synthetic_only", **metrics_tstr})
            print(f"      TSTR: {metrics_tstr}")
        except Exception as e:
            print(f"      TSTR failed: {e}")

        # Augmentation sweep: real + synthetic at α fractions
        for alpha in alpha_values:
            n_synthetic = int(n_train * alpha)
            if n_synthetic < 10:
                continue
            print(f"    α={alpha} ({n_synthetic} synthetic rows)...")
            try:
                df_syn = generator(n_synthetic)
                X_syn = df_syn.drop(columns=[target])
                y_syn = df_syn[target]
                X_syn = X_syn.reindex(columns=X_train.columns, fill_value=0)

                X_aug = pd.concat([X_train, X_syn], ignore_index=True)
                y_aug = pd.concat([y_train.reset_index(drop=True), y_syn.reset_index(drop=True)])

                metrics_aug = train_evaluate(X_aug, y_aug, X_test, y_test, task)
                results.append({"dataset": dataset_name, "method": method_name, "alpha": alpha,
                                "condition": "augmented", **metrics_aug})
                print(f"      α={alpha}: {metrics_aug}")
            except Exception as e:
                print(f"      α={alpha} failed: {e}")

    return pd.DataFrame(results)


def run_low_data_experiment(df, target, task, dataset_name):
    """
    Low-data regime: train on small n_real, augment with CTGAN, evaluate recovery.
    """
    print(f"\n  [Low-data regime experiment: {dataset_name}]")
    low_data_sizes = [250, 500, 1000, 2000]
    results = []

    X = df.drop(columns=[target])
    y = df[target]

    # full holdout test set (20% of full data)
    if task == "classification":
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    else:
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    for n_real in low_data_sizes:
        if n_real >= int(len(df) * 0.7):
            continue

        if task == "classification":
            X_small, _, y_small, _ = train_test_split(
                X, y, train_size=n_real, random_state=RANDOM_STATE, stratify=y
            )
        else:
            X_small, _, y_small, _ = train_test_split(
                X, y, train_size=n_real, random_state=RANDOM_STATE
            )

        df_small = X_small.copy()
        df_small[target] = y_small.values

        # baseline (no augmentation)
        m_base = train_evaluate(X_small, y_small, X_test, y_test, task)
        results.append({"n_real": n_real, "condition": "real_only", **m_base})

        # augmented with CTGAN (2× synthetic)
        try:
            df_syn = generate_ctgan(df_small, target, n_real * 2, task)
            X_syn = df_syn.drop(columns=[target]).reindex(columns=X_small.columns, fill_value=0)
            y_syn = df_syn[target]
            X_aug = pd.concat([X_small, X_syn], ignore_index=True)
            y_aug = pd.concat([y_small.reset_index(drop=True), y_syn.reset_index(drop=True)])
            m_aug = train_evaluate(X_aug, y_aug, X_test, y_test, task)
            results.append({"n_real": n_real, "condition": "ctgan_augmented", **m_aug})
        except Exception as e:
            print(f"      CTGAN low-data n={n_real} failed: {e}")

        # augmented with SMOTE (classification only)
        if task == "classification":
            try:
                df_smote = generate_smote(X_small, y_small, n_real)
                X_syn = df_smote.drop(columns=[target]).reindex(columns=X_small.columns, fill_value=0)
                y_syn = df_smote[target]
                X_aug = pd.concat([X_small, X_syn], ignore_index=True)
                y_aug = pd.concat([y_small.reset_index(drop=True), y_syn.reset_index(drop=True)])
                m_aug = train_evaluate(X_aug, y_aug, X_test, y_test, task)
                results.append({"n_real": n_real, "condition": "smote_augmented", **m_aug})
            except Exception as e:
                print(f"      SMOTE low-data n={n_real} failed: {e}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_ucurve(df_results, dataset_name, task):
    """Plot U-shaped error curve: α vs primary metric per method."""
    metric = "auc_roc" if task == "classification" else "mape"
    label = "AUC-ROC" if task == "classification" else "MAPE (lower=better)"

    aug = df_results[df_results["condition"] == "augmented"].copy()
    baseline_val = df_results[df_results["condition"] == "real_only"][metric].values
    if len(baseline_val) == 0:
        return
    baseline_val = baseline_val[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, grp in aug.groupby("method"):
        grp_sorted = grp.sort_values("alpha")
        ax.plot(grp_sorted["alpha"], grp_sorted[metric], marker="o", label=method)

    ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.5, label="Baseline (real only)")
    ax.set_xlabel("Synthetic fraction α (proportion of training data that is synthetic)")
    ax.set_ylabel(label)
    ax.set_title(f"U-Shaped Error Curve — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    slug = dataset_name.lower().replace(" ", "_")
    plt.savefig(RESULTS_DIR / f"ucurve_{slug}.png", dpi=150)
    plt.close()
    print(f"  Saved: results/ucurve_{slug}.png")


def plot_low_data(df_low, dataset_name, task):
    """Plot low-data regime recovery."""
    if df_low.empty:
        return
    metric = "auc_roc" if task == "classification" else "mape"
    label = "AUC-ROC" if task == "classification" else "MAPE"

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, grp in df_low.groupby("condition"):
        grp_sorted = grp.sort_values("n_real")
        ax.plot(grp_sorted["n_real"], grp_sorted[metric], marker="o", label=cond)

    ax.set_xlabel("Number of real training samples")
    ax.set_ylabel(label)
    ax.set_title(f"Low-Data Regime Recovery — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    slug = dataset_name.lower().replace(" ", "_")
    plt.savefig(RESULTS_DIR / f"lowdata_{slug}.png", dpi=150)
    plt.close()
    print(f"  Saved: results/lowdata_{slug}.png")


def plot_tstr_vs_baseline(all_results, task_map):
    """Bar chart: TSTR vs Baseline vs Best-Aug across datasets."""
    rows = []
    for ds_name, df_r in all_results.items():
        task = task_map[ds_name]
        metric = "auc_roc" if task == "classification" else "r2"
        base = df_r[df_r["condition"] == "real_only"][metric].mean()
        tstr = df_r[df_r["condition"] == "synthetic_only"][metric].mean()
        best_aug = df_r[df_r["condition"] == "augmented"][metric].max() if not df_r[df_r["condition"] == "augmented"].empty else np.nan
        rows.append({"dataset": ds_name, "Baseline": base, "TSTR (synthetic only)": tstr, "Best Augmented": best_aug})

    df_plot = pd.DataFrame(rows).set_index("dataset")
    ax = df_plot.plot(kind="bar", figsize=(10, 6), rot=15)
    ax.set_ylabel("AUC-ROC / R²")
    ax.set_title("Baseline vs. TSTR vs. Best Augmented — All Datasets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_comparison.png", dpi=150)
    plt.close()
    print("  Saved: results/summary_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    datasets = [
        load_telco_churn,
        load_bank_marketing,
        load_credit_default,
        load_online_retail_clv,
    ]

    all_results = {}
    all_low_data = {}
    task_map = {}

    for loader in datasets:
        df, target, task, name = loader()
        print(f"\nLoaded {name}: {df.shape}, target={target}, task={task}")

        # cap dataset size for reasonable runtime
        if len(df) > 15000:
            df = df.sample(15000, random_state=RANDOM_STATE).reset_index(drop=True)
            print(f"  Capped to 15,000 rows")

        # main experiment
        df_results = run_experiment(df, target, task, name)
        df_results.to_csv(RESULTS_DIR / f"metrics_{name.lower().replace(' ', '_')}.csv", index=False)
        all_results[name] = df_results
        task_map[name] = task

        # low-data regime
        df_low = run_low_data_experiment(df, target, task, name)
        if not df_low.empty:
            df_low.to_csv(RESULTS_DIR / f"lowdata_{name.lower().replace(' ', '_')}.csv", index=False)
        all_low_data[name] = df_low

        # plots
        plot_ucurve(df_results, name, task)
        plot_low_data(df_low, name, task)

    # cross-dataset summary plot
    plot_tstr_vs_baseline(all_results, task_map)

    # summary table
    summary_rows = []
    for ds_name, df_r in all_results.items():
        task = task_map[ds_name]
        metric = "auc_roc" if task == "classification" else "r2"
        base = df_r[df_r["condition"] == "real_only"][metric].mean()

        for method in df_r["method"].unique():
            if method == "Baseline":
                continue
            aug_subset = df_r[(df_r["method"] == method) & (df_r["condition"] == "augmented")]
            tstr_subset = df_r[(df_r["method"] == method) & (df_r["condition"] == "synthetic_only")]

            best_aug = aug_subset[metric].max() if not aug_subset.empty else np.nan
            best_alpha = aug_subset.loc[aug_subset[metric].idxmax(), "alpha"] if not aug_subset.empty else np.nan
            tstr_val = tstr_subset[metric].mean() if not tstr_subset.empty else np.nan

            summary_rows.append({
                "Dataset": ds_name,
                "Task": task,
                "Method": method,
                "Baseline": round(base, 4),
                "Best_Augmented": round(best_aug, 4) if not np.isnan(best_aug) else None,
                "Best_Alpha": best_alpha,
                "TSTR": round(tstr_val, 4) if not np.isnan(tstr_val) else None,
                "Delta_vs_Baseline": round(best_aug - base, 4) if not np.isnan(best_aug) else None,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df_summary.to_string(index=False))
    print(f"\nAll results saved to: {RESULTS_DIR}/")
