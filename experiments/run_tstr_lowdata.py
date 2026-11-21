"""
CTGAN train-on-synthetic-test-on-real (TSTR) numbers (Section 3.1) and the low-data runs (Figure 4).

Both are single-seed runs (seed 42) that reuse run_experiment and run_low_data_experiment from
synthetic_data_eval.py, the functions the first-pass scripts used. Datasets and caps follow the
first pass: Telco, Bank Marketing (capped at 15,000 rows), German Credit, and Nomao (capped at 10,000 rows).
Each dataset is run in a fresh call so NumPy starts from seed 42, as in the first-pass scripts.

Outputs (results/):
  lowdata_{telco_churn,bank_marketing,credit_default,nomao_lead}.csv   low-data runs, full metrics
  tstr_summary.csv   baseline and synthetic-only (TSTR) rows for every dataset, full metrics
  metrics_{telco_churn,bank_marketing,credit_default,nomao_lead}.csv   the single-seed alpha sweep
Usage (from the repo root): python3 experiments/run_tstr_lowdata.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from experiments.synthetic_data_eval import (
    load_telco_churn, load_bank_marketing, load_credit_default,
    run_experiment, run_low_data_experiment, RESULTS_DIR, RANDOM_STATE, DATA_DIR,
)

CAPS = {"Telco Churn": None, "Bank Marketing": 15000, "German Credit": None, "Nomao Lead": 10000}


def load_nomao_dense():
    df = pd.read_csv(DATA_DIR / "nomao.csv")
    return df, "target", "classification", "Nomao Lead"


tstr_rows = []
for loader in [load_telco_churn, load_bank_marketing, load_credit_default, load_nomao_dense]:
    np.random.seed(RANDOM_STATE)
    df, target, task, name = loader()
    cap = CAPS.get(name)
    if cap and len(df) > cap:
        df = df.sample(cap, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  Capped to {cap} rows", flush=True)
    slug = name.lower().replace(" ", "_")
    if name == "German Credit":
        slug = "credit_default"
    res = run_experiment(df, target, task, name)
    res.to_csv(RESULTS_DIR / f"metrics_{slug}.csv", index=False)
    tstr_rows.append(res[res.condition.isin(["real_only", "synthetic_only"])])
    low = run_low_data_experiment(df, target, task, name)
    low.to_csv(RESULTS_DIR / f"lowdata_{slug}.csv", index=False)
    pd.concat(tstr_rows).to_csv(RESULTS_DIR / "tstr_summary.csv", index=False)

print("\nDone.", flush=True)
