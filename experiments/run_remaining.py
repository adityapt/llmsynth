"""Run only Credit Default and Online Retail CLV, then merge with cached results."""
import sys
sys.path.insert(0, '.')
from experiments.synthetic_data_eval import (
    load_credit_default, load_online_retail_clv,
    run_experiment, run_low_data_experiment,
    plot_ucurve, plot_low_data, plot_tstr_vs_baseline,
    RESULTS_DIR, RANDOM_STATE
)
import pandas as pd
import numpy as np

# Load cached results from first two datasets
all_results = {}
all_low_data = {}
task_map = {}

for fname, name, task in [
    ("metrics_telco_churn.csv", "Telco Churn", "classification"),
    ("metrics_bank_marketing.csv", "Bank Marketing", "classification"),
]:
    p = RESULTS_DIR / fname
    if p.exists():
        all_results[name] = pd.read_csv(p)
        task_map[name] = task
        ld_p = RESULTS_DIR / f"lowdata_{name.lower().replace(' ', '_')}.csv"
        all_low_data[name] = pd.read_csv(ld_p) if ld_p.exists() else pd.DataFrame()
        print(f"Loaded cached: {name}")

# Run remaining datasets
for loader in [load_credit_default, load_online_retail_clv]:
    df, target, task, name = loader()
    print(f"\nLoaded {name}: {df.shape}, target={target}, task={task}")

    if len(df) > 15000:
        df = df.sample(15000, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  Capped to 15,000 rows")

    df_results = run_experiment(df, target, task, name)
    df_results.to_csv(RESULTS_DIR / f"metrics_{name.lower().replace(' ', '_')}.csv", index=False)
    all_results[name] = df_results
    task_map[name] = task

    df_low = run_low_data_experiment(df, target, task, name)
    if not df_low.empty:
        df_low.to_csv(RESULTS_DIR / f"lowdata_{name.lower().replace(' ', '_')}.csv", index=False)
    all_low_data[name] = df_low

    plot_ucurve(df_results, name, task)
    plot_low_data(df_low, name, task)

# Cross-dataset summary
plot_tstr_vs_baseline(all_results, task_map)

# Summary table
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
            "dataset": ds_name, "method": method, "task": task,
            "baseline": base, "tstr": tstr_val, "best_augmented": best_aug, "best_alpha": best_alpha
        })

pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary_table.csv", index=False)
print("\nDone. Results in results/")
