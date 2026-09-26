"""
Plots the dose-response sweep (minority count 16->1024, Bank Marketing,
fixed N=10,000, fixed holdout) — see run_dose_response_bank_marketing.py.

Output: results/plots/paper2/fig12_dose_response.png
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
OUT.mkdir(parents=True, exist_ok=True)

C = {"GaussianCopula": "#2196F3", "CTGAN": "#FF5722", "SMOTE": "#4CAF50", "Baseline": "#212121"}

def ci95(v):
    v = np.array(v)
    if len(v) < 2:
        return float(np.mean(v)), 0.0
    se = stats.sem(v)
    return float(np.mean(v)), se * stats.t.ppf(0.975, df=len(v) - 1)

df = pd.read_csv(RESULTS / "dose_response_bank_marketing.csv")
counts = sorted(df.minority_count.unique())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
base_m, base_h = [], []
for mc in counts:
    m, h = ci95(df[(df.minority_count == mc) & (df.method == "Baseline")]["auc_roc"].values)
    base_m.append(m); base_h.append(h)
ax.plot(counts, base_m, linestyle="--", color=C["Baseline"], marker="o", label="Baseline")
ax.fill_between(counts, np.array(base_m) - np.array(base_h), np.array(base_m) + np.array(base_h),
                 color=C["Baseline"], alpha=0.08)
for gen in ["GaussianCopula", "CTGAN", "SMOTE"]:
    gm, gh = [], []
    for mc in counts:
        m, h = ci95(df[(df.minority_count == mc) & (df.method == gen)]["auc_roc"].values)
        gm.append(m); gh.append(h)
    ax.plot(counts, gm, marker="o", color=C[gen], linewidth=2, label=gen)
    ax.fill_between(counts, np.array(gm) - np.array(gh), np.array(gm) + np.array(gh), color=C[gen], alpha=0.12)
ax.set_xscale("log")
ax.set_xlabel("Minority-class count (fixed N=10,000)")
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC vs minority count")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

ax = axes[1]
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
    ax.fill_between(counts, np.array(gains) - np.array(his), np.array(gains) + np.array(his), color=C[gen], alpha=0.12)
ax.axhline(0, color="black", linewidth=1, linestyle=":")
ax.set_xscale("log")
ax.set_xlabel("Minority-class count (fixed N=10,000)")
ax.set_ylabel("Gain vs baseline (AUC points)")
ax.set_title("Augmentation gain vs minority count")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

fig.suptitle(
    "Figure 12 — Dose-response: minority count vs augmentation gain (Bank Marketing, single dataset, fixed N)\n"
    "Positive gain only at count=16 (SMOTE significant); augmentation significantly HURTS at counts >=64",
    fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig12_dose_response.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig12_dose_response.png'}")
