"""
Combined dose-response comparison: Bank Marketing vs Nomao, side by side.
Output: results/plots/paper2/fig13_dose_response_combined.png
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
C = {"GaussianCopula": "#2196F3", "CTGAN": "#FF5722", "SMOTE": "#4CAF50"}

def ci95(v):
    v = np.array(v)
    if len(v) < 2:
        return float(np.mean(v)), 0.0
    se = stats.sem(v)
    return float(np.mean(v)), se * stats.t.ppf(0.975, df=len(v) - 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (label, path) in zip(axes, [
    ("Bank Marketing (17 features)", "results/dose_response_bank_marketing.csv"),
    ("Nomao (119 features)", "results/dose_response_nomao.csv"),
]):
    df = pd.read_csv(path)
    counts = sorted(df.minority_count.unique())
    for gen in ["GaussianCopula", "CTGAN", "SMOTE"]:
        gains, his = [], []
        for mc in counts:
            base = df[(df.minority_count == mc) & (df.method == "Baseline")].set_index("seed")["auc_roc"]
            aug = df[(df.minority_count == mc) & (df.method == gen)].set_index("seed")["auc_roc"]
            common = sorted(set(base.index) & set(aug.index))
            diffs = (aug.loc[common] - base.loc[common]).values * 100
            m, h = ci95(diffs)
            gains.append(m); his.append(h)
        ax.plot(counts, gains, marker="o", color=C[gen], linewidth=2, label=gen)
        ax.fill_between(counts, np.array(gains) - np.array(his), np.array(gains) + np.array(his),
                         color=C[gen], alpha=0.12)
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Minority-class count (fixed N=10,000)")
    ax.set_ylabel("Gain vs baseline (AUC points)")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Figure 13 — Dose-response replication: Bank Marketing vs Nomao\n"
    "Same qualitative direction (gains shrink as count rises) — different threshold and effect size (Nomao's near-ceiling baseline leaves little room to move)",
    fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig13_dose_response_combined.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig13_dose_response_combined.png'}")
