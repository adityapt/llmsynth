"""Run experiment on Nomao (OpenML id=1486) — telemarketing lead classification."""
import sys
sys.path.insert(0, '.')
from experiments.synthetic_data_eval import (
    run_experiment, run_low_data_experiment,
    plot_ucurve, plot_low_data,
    DATA_DIR, RESULTS_DIR, RANDOM_STATE
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_nomao():
    """Nomao telemarketing dataset — binary classification, ~25% positive."""
    path = DATA_DIR / "nomao.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml
        print("  Downloading Nomao from OpenML...")
        ds = fetch_openml(data_id=1486, as_frame=True, parser="auto")
        df = ds.frame.copy()
        target_col = ds.target_names[0] if ds.target_names else df.columns[-1]
        df = df.rename(columns={target_col: "target"})
        # target: '1' = real agency, '2' = not real → recode to 0/1
        df["target"] = df["target"].astype(str).str.strip()
        vals = df["target"].unique()
        print(f"  Target unique values: {vals}")
        # Map minority class to 1
        vc = df["target"].value_counts()
        minority = vc.idxmin()
        df["target"] = (df["target"] == minority).astype(int)
        # Encode any remaining object/category cols
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) == "category":
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        df.to_csv(path, index=False)
        print(f"  Saved to {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}, positive rate: {df['target'].mean()*100:.1f}%")
    return df, "target", "classification", "Nomao Lead"

# Run
print("=" * 60)
df, target, task, name = load_nomao()
print(f"\nLoaded {name}: {df.shape}, target={target}, task={task}")

# Cap for tractability
if len(df) > 10000:
    df = df.sample(10000, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Capped to 10,000 rows")

print(f"  Positive rate after cap: {df[target].mean()*100:.1f}%")

df_results = run_experiment(df, target, task, name)
df_results.to_csv(RESULTS_DIR / f"metrics_nomao_lead.csv", index=False)

df_low = run_low_data_experiment(df, target, task, name)
if not df_low.empty:
    df_low.to_csv(RESULTS_DIR / f"lowdata_nomao_lead.csv", index=False)

plot_ucurve(df_results, name, task)
plot_low_data(df_low, name, task)

# Print summary
print("\n=== RESULTS ===")
metric = "auc_roc"
base = df_results[df_results["condition"] == "real_only"][metric].mean()
print(f"Baseline (real only): {base:.4f}")
for method in ["GaussianCopula", "CTGAN", "SMOTE"]:
    sub = df_results[(df_results["method"] == method) & (df_results["condition"] == "augmented")]
    if sub.empty:
        continue
    best = sub[metric].max()
    best_alpha = sub.loc[sub[metric].idxmax(), "alpha"]
    tstr_sub = df_results[(df_results["method"] == method) & (df_results["condition"] == "synthetic_only")]
    tstr = tstr_sub[metric].mean() if not tstr_sub.empty else float("nan")
    print(f"{method}: TSTR={tstr:.4f}, Best aug={best:.4f} (α={best_alpha}), gain={best-base:+.4f}")

print("\nDone. Results in results/")
