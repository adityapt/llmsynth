"""
Combined "gain vs positive rate" view — overlays the original 6-dataset
cross-dataset CTGAN-gain regression (sparse, one point per dataset, Figure 9)
with the two dense within-dataset dose-response curves (Bank Marketing,
Nomao — Figure 13). This doubles as a diminishing-returns plot: within each
dose-response curve, marginal gain from additional real minority data shrinks
and turns negative (the curves are not strictly monotonic at every point), on a shared "gain vs positive rate" axis
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
def _ctgan_best_gain(path, auc_col="auc_roc"):
    """Best mean seed-paired CTGAN gain over alpha, in AUC points, from a 5-seed result file."""
    d = pd.read_csv(path); d["alpha"] = d["alpha"].astype(float)
    base = d[d.method == "Baseline"].set_index("seed")[auc_col]
    gains = [((d[(d.method == "CTGAN") & (d.alpha == a)].set_index("seed")[auc_col] - base).mean() * 100)
             for a in sorted(d[d.method == "CTGAN"].alpha.unique())]
    return max(gains)

# Latest 5-seed reruns (fullmetrics_*.csv for the control datasets, ci_*_fullmetrics.csv for marketing)
CROSS_DATASET = [
    ("Telco Churn",     26.6, _ctgan_best_gain("results/fullmetrics_telco.csv")),
    ("Bank Marketing",  11.7, _ctgan_best_gain("results/fullmetrics_bank_marketing.csv")),
    ("German Credit",   30.0, _ctgan_best_gain("results/fullmetrics_german_credit.csv")),
    ("Nomao Lead",      28.3, _ctgan_best_gain("results/fullmetrics_nomao.csv")),
    ("Hillstrom",        0.9, _ctgan_best_gain("results/ci_hillstrom_fullmetrics.csv")),
    ("Criteo",            0.3, _ctgan_best_gain("results/ci_criteo_fullmetrics.csv")),
]

fig, ax = plt.subplots(figsize=(9, 6))

# Cross-dataset sparse points
# One marker and legend entry per dataset (cross-dataset points, one per dataset)
DS_STYLE = {
    "Telco Churn":    ("#8E24AA", "D"),
    "Bank Marketing": ("#FF5722", "s"),
    "German Credit":  ("#F9A825", "^"),
    "Nomao Lead":     ("#2196F3", "v"),
    "Hillstrom":      ("#D81B60", "P"),
    "Criteo":         ("#212121", "X"),
}
pts, lines = [], []
for name, r, g in CROSS_DATASET:
    color, marker = DS_STYLE[name]
    pts.append(ax.scatter([r], [g], s=110, color=color, marker=marker, zorder=5,
               edgecolors="white", linewidths=0.8, label=f"{name} ({r}% positive)"))

# Dense within-dataset dose-response curves
for label, path, color, mk in [
    ("Bank Marketing dose-response", "results/dose_response_bank_marketing.csv", "#FF5722", "s"),
    ("Nomao dose-response", "results/dose_response_nomao.csv", "#2196F3", "v"),
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
    lines.append(ax.plot(rate_pts, gain_pts, marker=mk, markersize=7, color=color, linewidth=2, label=label)[0])
    ax.fill_between(rate_pts, np.array(gain_pts) - np.array(his), np.array(gain_pts) + np.array(his),
                     color=color, alpha=0.12)

ax.axhline(0, color="black", linewidth=1, linestyle=":")
ax.set_xscale("log")
ax.set_xlabel("Positive rate (%, log scale)")
ax.set_ylabel("CTGAN gain vs baseline (AUC points)")
# Two legends: one for the cross-dataset points, one for the dose-response curves.
# Bank Marketing and Nomao use the same colour for their point and their curve.
leg1 = ax.legend(handles=pts, title="Cross-dataset points (best CTGAN gain)", fontsize=8.5,
                 title_fontsize=9, loc="upper right")
ax.add_artist(leg1)
ax.legend(handles=lines, title="Within-dataset dose-response curves", fontsize=8.5,
          title_fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig14_gain_vs_positive_rate.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig14_gain_vs_positive_rate.png'}")
