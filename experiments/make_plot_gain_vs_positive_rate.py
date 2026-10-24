"""
Combined "gain vs positive rate" view — overlays the original 6-dataset
cross-dataset CTGAN-gain regression (sparse, one point per dataset, Figure 9)
with the two dense within-dataset dose-response curves (Bank Marketing,
Nomao — Figure 13). This doubles as a diminishing-returns plot: within each
dose-response curve, marginal gain from additional real minority data shrinks
monotonically and turns negative, on a shared "gain vs positive rate" axis
that makes the cross-dataset and within-dataset evidence directly comparable.

Output: results/plots/paper2/fig14_gain_vs_positive_rate.png
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

# Cross-dataset CTGAN-SPECIFIC best-alpha gains (recomputed directly from raw
# CSVs -- NOT the §4.2 "best generator" column, which is GaussianCopula for
# Telco and Nomao; this plot needs CTGAN's own gain for every dataset for a
# fair like-for-like comparison with the CTGAN dose-response curves below)
CROSS_DATASET = [
    ("Telco Churn",     26.6,  0.06),
    ("Bank Marketing",  11.7, -0.17),
    ("German Credit",   30.0,  0.27),
    ("Nomao Lead",      28.3, -0.07),
    ("Hillstrom",        0.9,  5.75),
    ("Criteo",            0.3, 12.87),
]

fig, ax = plt.subplots(figsize=(9, 6))

# Cross-dataset sparse points
# One marker and legend entry per dataset (cross-dataset points, one per dataset)
DS_STYLE = {
    "Telco Churn":    ("#8E24AA", "D"),
    "Bank Marketing": ("#00897B", "s"),
    "German Credit":  ("#F9A825", "^"),
    "Nomao Lead":     ("#6D4C41", "v"),
    "Hillstrom":      ("#D81B60", "P"),
    "Criteo":         ("#212121", "X"),
}
for name, r, g in CROSS_DATASET:
    color, marker = DS_STYLE[name]
    ax.scatter([r], [g], s=110, color=color, marker=marker, zorder=5,
               edgecolors="white", linewidths=0.8, label=f"{name} ({r}% positive)")

# Dense within-dataset dose-response curves
for label, path, color in [
    ("Bank Marketing dose-response", "results/dose_response_bank_marketing.csv", "#FF5722"),
    ("Nomao dose-response", "results/dose_response_nomao.csv", "#2196F3"),
]:
    df = pd.read_csv(path)
    counts = sorted(df.minority_count.unique())
    rate_pts, gain_pts, his = [], [], []
    for mc in counts:
        base = df[(df.minority_count == mc) & (df.method == "Baseline")].set_index("seed")["auc_roc"]
        aug = df[(df.minority_count == mc) & (df.method == "CTGAN")].set_index("seed")["auc_roc"]
        common = sorted(set(base.index) & set(aug.index))
        diffs = (aug.loc[common] - base.loc[common]).values * 100
        m, h = ci95(diffs)
        rate_pts.append(mc / 10_000 * 100); gain_pts.append(m); his.append(h)
    ax.plot(rate_pts, gain_pts, marker="o", color=color, linewidth=2, label=label)
    ax.fill_between(rate_pts, np.array(gain_pts) - np.array(his), np.array(gain_pts) + np.array(his),
                     color=color, alpha=0.12)

ax.axhline(0, color="black", linewidth=1, linestyle=":")
ax.set_xscale("log")
ax.set_xlabel("Positive rate (%, log scale)")
ax.set_ylabel("CTGAN gain vs baseline (AUC points)")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig14_gain_vs_positive_rate.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig14_gain_vs_positive_rate.png'}")
