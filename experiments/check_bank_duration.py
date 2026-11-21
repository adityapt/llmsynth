"""Check whether the Bank Marketing runs include the call-duration feature (V12 in the OpenML file).

The loader in synthetic_data_eval.py drops a column named 'duration', but the OpenML file used here
(data/bank_marketing.csv) names its columns V1..V16, so that drop is a no-op. This script measures
the real-only Gradient Boosting AUC with and without V12 under the same protocol as the main runs
(15,000-row sample, 80/20 stratified split, 5 seeds).
Output: results/bank_duration_check.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from experiments.synthetic_data_eval import load_bank_marketing

df, target, _, _ = load_bank_marketing()
print("columns used by the loader:", [c for c in df.columns if c != target])
rows = []
for seed in [42, 123, 7, 2024, 999]:
    d = df.sample(15_000, random_state=seed).reset_index(drop=True)
    tr, te = train_test_split(d, test_size=0.2, random_state=seed, stratify=d[target])
    for label, drop in [("with V12 (as run)", []), ("without V12", ["V12"])]:
        cols = [c for c in d.columns if c != target and c not in drop]
        m = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed).fit(tr[cols], tr[target])
        auc = roc_auc_score(te[target], m.predict_proba(te[cols])[:, 1])
        rows.append({"seed": seed, "features": label, "n_features": len(cols), "auc_roc": auc})
out = pd.DataFrame(rows); out.to_csv("results/bank_duration_check.csv", index=False)
print(out.groupby("features").auc_roc.agg(["mean", "std", "count"]).round(4))
print("V12 range:", df["V12"].min(), df["V12"].max())
