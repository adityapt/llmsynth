"""
Bar chart comparing all evaluated augmentation methods (headline generators
+ the newly-added missing baselines) on Hillstrom and Criteo, sorted by gain.
Output: results/plots/paper2/fig15_missing_baselines.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "plots" / "paper2"

def ci95(v):
    v = np.array(v)
    if len(v) < 2:
        return float(np.mean(v)), 0.0
    se = stats.sem(v)
    return float(np.mean(v)), se * stats.t.ppf(0.975, df=len(v) - 1)


def best_gain(df, method, alpha_col="alpha"):
    base = df[df.method == "Baseline"].set_index("seed")["auc_roc"]
    bm, _ = ci95(base.values)
    best, best_h = -999, 0
    for alpha in df[df.method == method][alpha_col].unique():
        vals = df[(df.method == method) & (df[alpha_col] == alpha)]["auc_roc"].values
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        gain = (m - bm) * 100
        if gain > best:
            best, best_h = gain, h * 100
    return best, best_h


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (label, ci_path, mb_path) in zip(axes, [
    ("Hillstrom", "results/ci_hillstrom.csv", "results/missing_baselines_hillstrom.csv"),
    ("Criteo", "results/ci_criteo.csv", "results/missing_baselines_criteo.csv"),
]):
    df_ci = pd.read_csv(ci_path)
    df_mb = pd.read_csv(mb_path)

    methods = []
    for m in ["CTGAN", "SMOTE", "GaussianCopula"]:
        g, h = best_gain(df_ci, m)
        methods.append((m, g, h))
    for m in ["ADASYN", "BorderlineSMOTE"]:
        g, h = best_gain(df_mb, m)
        methods.append((m, g, h))
    # RandomUnderSampler: single config, alpha column holds "1:1" string
    base = df_mb[df_mb.method == "Baseline"]["auc_roc"].values
    bm, _ = ci95(base)
    ru = df_mb[df_mb.method == "RandomUnderSampler"]["auc_roc"].values
    m, h = ci95(ru)
    methods.append(("RandomUndersampler", (m - bm) * 100, h * 100))

    methods.sort(key=lambda x: x[1], reverse=True)
    names = [m[0] for m in methods]
    gains = [m[1] for m in methods]
    errs = [m[2] for m in methods]
    colors = ["#FF5722" if n == "CTGAN" else ("#4CAF50" if n in ("SMOTE", "ADASYN") else "#9E9E9E") for n in names]

    ax.barh(names, gains, xerr=errs, color=colors, capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Gain vs baseline (AUC points)")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

fig.suptitle("Figure 15 — All evaluated methods, sorted by gain (orange=CTGAN, green=free methods matching it)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig15_missing_baselines.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig15_missing_baselines.png'}")
