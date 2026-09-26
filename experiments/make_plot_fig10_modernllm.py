"""
Regenerates fig10_modernllm_comparison.png (GPT-2 vs Mistral-7B vs Baseline).

This figure previously had no source script anywhere in the repository —
it was produced ad hoc and was not reproducible, and it was built from a
version of the GPT-2/Mistral-7B comparison that had a wrong-baseline bug
(see paper2-empirical.md §4.7 correction note). This script fixes both
problems: it is a real, checked-in script, and it recomputes every point
from raw per-seed CSVs with each backbone's own seed-matched baseline,
excluding seeds that failed to generate valid rows (non-null 'error').

Reads:
  results/great_german_results.csv               (GPT-2, German Credit)
  results/modernllm_german_results_parallel.csv  (Mistral-7B, German Credit)
  results/ci_great_hillstrom.csv                 (GPT-2, Hillstrom)
  results/modernllm_hillstrom_results_parallel.csv (Mistral-7B, Hillstrom)
  results/great_telco_results.csv                (GPT-2, Telco)
  results/modernllm_telco_results_parallel.csv   (Mistral-7B, Telco)

Writes:
  results/plots/paper2/fig10_modernllm_comparison.png
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
OUT     = RESULTS / "plots" / "paper2"
OUT.mkdir(parents=True, exist_ok=True)

C = {"Baseline": "#212121", "GPT-2 GReaT": "#795548", "Mistral-7B": "#E91E63"}

MIN_VALID_SEEDS = 3  # below this, a point is dropped as too thin to plot (matches paper's exclusion rule)


def ci95(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) < 2:
        return float(np.mean(v)) if len(v) else float("nan"), 0.0
    se = stats.sem(v)
    return float(np.mean(v)), se * stats.t.ppf(0.975, df=len(v) - 1)


def gpt2_series(path, method_name="GReaT"):
    """GPT-2 files use a single Baseline+method pair per (n, seed), no failures documented."""
    df = pd.read_csv(path)
    ns, means, his = [], [], []
    for n in sorted(df["n"].unique()):
        vals = df[(df["n"] == n) & (df["method"] == method_name)]["auc"].dropna().values
        if len(vals) < MIN_VALID_SEEDS:
            continue
        m, h = ci95(vals)
        ns.append(n); means.append(m); his.append(h)
    return ns, means, his


def gpt2_baseline_series(path):
    df = pd.read_csv(path)
    ns, means, his = [], [], []
    for n in sorted(df["n"].unique()):
        vals = df[(df["n"] == n) & (df["method"] == "Baseline")]["auc"].dropna().values
        if len(vals) < MIN_VALID_SEEDS:
            continue
        m, h = ci95(vals)
        ns.append(n); means.append(m); his.append(h)
    return ns, means, his


def mistral_series(path):
    """Mistral files log an 'error' string for failed seeds but still carry a stray
    AUC value in some rows — those must be excluded, not averaged in."""
    df = pd.read_csv(path).drop_duplicates(subset=["n", "seed", "method"], keep="last")
    ns, base_m, base_h, mis_m, mis_h, n_valid = [], [], [], [], [], []
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n]
        base = sub[sub["method"] == "Baseline"].set_index("seed")["auc"]
        mllm_rows = sub[sub["method"] == "ModernLLM"]
        mllm_valid = mllm_rows[mllm_rows["error"].isna()].set_index("seed")["auc"]
        common = sorted(set(base.index) & set(mllm_valid.index))
        if len(common) < MIN_VALID_SEEDS:
            continue
        bm, bh = ci95(base.loc[common].values)
        mm, mh = ci95(mllm_valid.loc[common].values)
        ns.append(n); base_m.append(bm); base_h.append(bh)
        mis_m.append(mm); mis_h.append(mh); n_valid.append(len(common))
    return ns, base_m, base_h, mis_m, mis_h, n_valid


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

datasets = [
    ("German Credit", "results/great_german_results.csv", "results/modernllm_german_results_parallel.csv"),
    ("Hillstrom",     "results/ci_great_hillstrom.csv",    "results/modernllm_hillstrom_results_parallel.csv"),
    ("Telco Churn",   "results/great_telco_results.csv",   "results/modernllm_telco_results_parallel.csv"),
]

for ax, (label, gpt2_path, mistral_path) in zip(axes, datasets):
    gn, gb_m, gb_h = gpt2_baseline_series(gpt2_path)
    _, gm_m, gm_h = gpt2_series(gpt2_path)
    mn, mb_m, mb_h, mm_m, mm_h, n_valid = mistral_series(mistral_path)

    if gn:
        ax.plot(gn, gb_m, linestyle="--", color=C["Baseline"], linewidth=1.3, label="Baseline (GPT-2 run)")
        ax.fill_between(gn, np.array(gb_m) - np.array(gb_h), np.array(gb_m) + np.array(gb_h),
                         color=C["Baseline"], alpha=0.06)
        ax.plot(gn, gm_m, marker="o", color=C["GPT-2 GReaT"], linewidth=2.0, markersize=6, label="GPT-2 GReaT")
        ax.fill_between(gn, np.array(gm_m) - np.array(gm_h), np.array(gm_m) + np.array(gm_h),
                         color=C["GPT-2 GReaT"], alpha=0.12)

    if mn:
        ax.plot(mn, mb_m, linestyle=":", color=C["Baseline"], linewidth=1.3, alpha=0.7, label="Baseline (Mistral run)")
        ax.plot(mn, mm_m, marker="^", color=C["Mistral-7B"], linewidth=2.0, markersize=6, label="Mistral-7B")
        ax.fill_between(mn, np.array(mm_m) - np.array(mm_h), np.array(mm_m) + np.array(mm_h),
                         color=C["Mistral-7B"], alpha=0.12)
        for x, y, k in zip(mn, mm_m, n_valid):
            if k < 5:
                ax.annotate(f"{k}/5", (x, y), textcoords="offset points", xytext=(0, 8),
                            fontsize=7, ha="center", color=C["Mistral-7B"])

    ax.set_xlabel("Training set size n", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Figure 10 — GPT-2 (117M) vs Mistral-7B (7B) vs Baseline\n"
    "Each backbone plotted against its own seed-matched baseline; annotated points used fewer than 5 valid seeds "
    "(generation failures excluded, not averaged in). Backbone scaling does not rescue GReaT on any dataset tested.",
    fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig10_modernllm_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'fig10_modernllm_comparison.png'}")
