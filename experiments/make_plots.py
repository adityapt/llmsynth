"""
Publication-quality comparison plots for the MarSynth paper.

Generates:
  results/plots/plot_augmentation_gain_by_imbalance.png  — gain vs positive rate across all datasets
  results/plots/plot_ucurves_grid.png                    — alpha sweep grid (all datasets, all generators)
  results/plots/plot_tstr_gap.png                        — TSTR vs real baseline comparison
  results/plots/plot_great_smalln.png                    — GReaT vs CTGAN vs GC at small-n
  results/plots/plot_ci_hillstrom.png                    — Hillstrom with 95% CI error bars
  results/plots/plot_ci_criteo.png                       — Criteo with 95% CI error bars
  results/plots/plot_imbalance_vs_gain.png               — scatter: imbalance severity vs best gain
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS  = Path("results")
PLOTS    = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GaussianCopula": "#2196F3",
    "CTGAN":          "#FF5722",
    "SMOTE":          "#4CAF50",
    "GReaT":          "#9C27B0",
    "Baseline":       "#212121",
}
STYLE = {"linewidth": 2.2, "markersize": 7}


# ─────────────────────────────────────────────────────────────────────────────
# Load all results CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load(name):
    p = RESULTS / f"metrics_{name}.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["dataset"] = name
        return df
    return None

datasets_meta = {
    "telco_churn":     {"label": "Telco Churn",     "pos_rate": 26.6, "n": 7032},
    "bank_marketing":  {"label": "Bank Marketing",  "pos_rate": 11.7, "n": 15000},
    "credit_default":  {"label": "German Credit",   "pos_rate": 30.0, "n": 1000},
    "hillstrom":       {"label": "Hillstrom Email", "pos_rate": 0.9,  "n": 10000},
    "criteo_uplift":   {"label": "Criteo Display",  "pos_rate": 0.2,  "n": 10000},
    "kdd_appetency":   {"label": "KDD Appetency",   "pos_rate": 6.7,  "n": 5000},
}

frames = {k: load(k) for k, _ in datasets_meta.items()}
frames = {k: v for k, v in frames.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# 1. U-curve grid — alpha sweep for each dataset × generator
# ─────────────────────────────────────────────────────────────────────────────

available = [k for k in datasets_meta if k in frames]
n_datasets = len(available)
ncols = 3
nrows = int(np.ceil(n_datasets / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, key in enumerate(available):
    ax   = axes[i]
    df   = frames[key]
    meta = datasets_meta[key]

    baseline_auc = df[(df["method"] == "Baseline") & (df["condition"] == "real_only")]["auc_roc"].values
    baseline_val = baseline_auc[0] if len(baseline_auc) > 0 else None

    for gen in ["GaussianCopula", "CTGAN", "SMOTE"]:
        sub = df[(df["method"] == gen) & (df["condition"] == "augmented")].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub["auc_roc"], marker="o", label=gen,
                color=COLORS[gen], **STYLE)

    if baseline_val is not None:
        ax.axhline(baseline_val, linestyle="--", color=COLORS["Baseline"],
                   linewidth=1.5, label=f"Baseline ({baseline_val:.3f})")

    ax.set_title(f"{meta['label']}\n{meta['pos_rate']}% positive, n={meta['n']:,}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Synthetic fraction α", fontsize=9)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

# hide unused axes
for j in range(len(available), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Augmentation U-Curves: AUC-ROC vs Synthetic Fraction α\n(across all datasets and generators)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(PLOTS / "plot_ucurves_grid.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: plot_ucurves_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TSTR gap bar chart — synthetic-only vs real baseline
# ─────────────────────────────────────────────────────────────────────────────

tstr_data = []
for key in available:
    df   = frames[key]
    meta = datasets_meta[key]
    baseline_auc = df[(df["method"] == "Baseline") & (df["condition"] == "real_only")]["auc_roc"]
    if baseline_auc.empty:
        continue
    base_val = baseline_auc.values[0]
    for gen in ["GaussianCopula", "CTGAN"]:
        tstr_rows = df[(df["method"] == gen) & (df["condition"] == "synthetic_only")]
        if tstr_rows.empty:
            continue
        tstr_val = tstr_rows["auc_roc"].values[0]
        tstr_data.append({
            "dataset": meta["label"],
            "pos_rate": meta["pos_rate"],
            "generator": gen,
            "baseline": base_val,
            "tstr": tstr_val,
            "gap_pct": (base_val - tstr_val) / base_val * 100,
        })

if tstr_data:
    df_tstr = pd.DataFrame(tstr_data)
    datasets_ordered = df_tstr.groupby("dataset")["gap_pct"].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(len(datasets_ordered))
    width   = 0.35

    for j, gen in enumerate(["GaussianCopula", "CTGAN"]):
        sub = df_tstr[df_tstr["generator"] == gen].set_index("dataset")
        vals = [sub.loc[d, "gap_pct"] if d in sub.index else 0 for d in datasets_ordered]
        bars = ax.bar(x + j * width, vals, width, label=gen,
                      color=COLORS[gen], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets_ordered, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("TSTR Gap (% below real baseline AUC)", fontsize=11)
    ax.set_title("TSTR Performance Gap: Synthetic-Only Training vs Real Data Baseline\n"
                 "(Higher = worse; synthetic-only is never a safe replacement)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot_tstr_gap.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: plot_tstr_gap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Augmentation gain vs class imbalance — the key insight scatter
# ─────────────────────────────────────────────────────────────────────────────

gain_data = []
for key in available:
    df   = frames[key]
    meta = datasets_meta[key]
    baseline_auc = df[(df["method"] == "Baseline") & (df["condition"] == "real_only")]["auc_roc"]
    if baseline_auc.empty:
        continue
    base_val = baseline_auc.values[0]
    for gen in ["GaussianCopula", "CTGAN", "SMOTE"]:
        aug = df[(df["method"] == gen) & (df["condition"] == "augmented")]
        if aug.empty:
            continue
        best_auc = aug["auc_roc"].max()
        gain_data.append({
            "dataset":  meta["label"],
            "pos_rate": meta["pos_rate"],
            "n":        meta["n"],
            "generator": gen,
            "best_gain_pts": best_auc - base_val,
        })

if gain_data:
    df_gain = pd.DataFrame(gain_data)
    best_per_dataset = df_gain.groupby(["dataset", "pos_rate", "n"])["best_gain_pts"].max().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        best_per_dataset["pos_rate"],
        best_per_dataset["best_gain_pts"],
        s=best_per_dataset["n"].apply(lambda x: max(50, min(500, x / 50))),
        c=best_per_dataset["best_gain_pts"],
        cmap="RdYlGn",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    plt.colorbar(scatter, ax=ax, label="Best AUC gain (pts)", shrink=0.8)

    for _, row in best_per_dataset.iterrows():
        ax.annotate(
            row["dataset"],
            xy=(row["pos_rate"], row["best_gain_pts"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="#333333",
        )

    ax.axhline(0, linestyle="--", color="#aaa", linewidth=1.2)
    ax.set_xlabel("Positive Rate (%)", fontsize=12)
    ax.set_ylabel("Best Augmentation Gain (AUC pts)", fontsize=12)
    ax.set_title("Augmentation Gain vs Class Imbalance\n"
                 "(Bubble size ∝ dataset n; color = gain magnitude)",
                 fontsize=12, fontweight="bold")
    ax.invert_xaxis()   # higher imbalance on left
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot_imbalance_vs_gain.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: plot_imbalance_vs_gain.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GReaT small-n comparison
# ─────────────────────────────────────────────────────────────────────────────

great_path = RESULTS / "metrics_great_german_credit.csv"
if great_path.exists():
    df_gr = pd.read_csv(great_path)
    # Expect columns: n, method/generator, auc_roc (or auc)
    auc_col = "auc_roc" if "auc_roc" in df_gr.columns else "auc"
    gen_col  = "method"  if "method"  in df_gr.columns else "generator"

    ns = sorted(df_gr["n"].unique()) if "n" in df_gr.columns else []
    if ns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for gen in ["Baseline", "GaussianCopula", "CTGAN", "GReaT"]:
            sub = df_gr[df_gr[gen_col] == gen].sort_values("n")
            if sub.empty:
                continue
            ls = "--" if gen == "Baseline" else "-"
            marker = None if gen == "Baseline" else "o"
            ax.plot(sub["n"], sub[auc_col], linestyle=ls, marker=marker,
                    label=gen, color=COLORS.get(gen, "#888"), **STYLE)

        ax.set_xlabel("Training set size n", fontsize=12)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title("LLM vs Statistical Generators at Extreme Small-n\n"
                     "(German Credit dataset, fixed 300-row holdout)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(ns)
        plt.tight_layout()
        plt.savefig(PLOTS / "plot_great_smalln.png", dpi=160, bbox_inches="tight")
        plt.close()
        print("Saved: plot_great_smalln.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CI plots — Hillstrom and Criteo with error bars (if CI results exist)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ci(ci_csv, raw_csv, dataset_label, out_name, headline_alpha):
    """Plot augmentation curves with 95% CI error bands."""
    if not Path(raw_csv).exists():
        print(f"  Skipping {out_name} — CI results not yet available")
        return

    df_raw = pd.read_csv(raw_csv)
    baseline_per_seed = df_raw[df_raw["method"] == "Baseline"][["seed", "auc_roc"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for gen in ["GaussianCopula", "CTGAN", "SMOTE"]:
        sub = df_raw[(df_raw["method"] == gen) & (df_raw["condition"] == "augmented")]
        if sub.empty:
            continue
        alphas = sorted(sub["alpha"].unique())
        means, lowers, uppers = [], [], []
        for alpha in alphas:
            vals = sub[sub["alpha"] == alpha]["auc_roc"].values
            if len(vals) < 2:
                continue
            from scipy import stats as sc
            m   = np.mean(vals)
            h   = sc.sem(vals) * sc.t.ppf(0.975, df=len(vals) - 1)
            means.append(m); lowers.append(m - h); uppers.append(m + h)

        if not means:
            continue
        ax.plot(alphas[:len(means)], means, marker="o", label=gen,
                color=COLORS[gen], **STYLE)
        ax.fill_between(alphas[:len(means)], lowers, uppers,
                        color=COLORS[gen], alpha=0.15)

    # Baseline band
    base_vals = baseline_per_seed["auc_roc"].values
    from scipy import stats as sc
    bm = np.mean(base_vals)
    bh = sc.sem(base_vals) * sc.t.ppf(0.975, df=len(base_vals) - 1)
    ax.axhline(bm, linestyle="--", color=COLORS["Baseline"], linewidth=1.8,
               label=f"Baseline ({bm:.3f} ± {bh:.3f})")
    ax.axhspan(bm - bh, bm + bh, color=COLORS["Baseline"], alpha=0.08)

    ax.set_xlabel("Synthetic fraction α", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(f"{dataset_label} — Augmentation Curves with 95% CI\n"
                 f"(5 independent seeds; shaded region = 95% confidence interval)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / out_name, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_name}")


plot_ci(
    RESULTS / "ci_summary.csv", RESULTS / "ci_hillstrom.csv",
    "Hillstrom Email Marketing (0.9% conversion)",
    "plot_ci_hillstrom.png", headline_alpha=0.3
)
plot_ci(
    RESULTS / "ci_summary.csv", RESULTS / "ci_criteo.csv",
    "Criteo Display Advertising (0.2% conversion)",
    "plot_ci_criteo.png", headline_alpha=0.5
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Best augmentation gain per generator per dataset — grouped bar
# ─────────────────────────────────────────────────────────────────────────────

if gain_data:
    df_gain_full = pd.DataFrame(gain_data)
    dataset_order = ["Telco Churn", "Bank Marketing", "German Credit",
                     "KDD Appetency", "Hillstrom Email", "Criteo Display"]
    dataset_order = [d for d in dataset_order if d in df_gain_full["dataset"].values]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x     = np.arange(len(dataset_order))
    width = 0.25
    gens  = ["GaussianCopula", "CTGAN", "SMOTE"]

    for j, gen in enumerate(gens):
        sub  = df_gain_full[df_gain_full["generator"] == gen].set_index("dataset")
        vals = [sub.loc[d, "best_gain_pts"] if d in sub.index else 0 for d in dataset_order]
        bars = ax.bar(x + j * width, vals, width, label=gen,
                      color=COLORS[gen], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if abs(val) > 0.002:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if val >= 0 else -0.008),
                        f"{val:+.2f}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=7.5)

    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_order, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Best AUC gain over baseline (pts)", fontsize=11)
    ax.set_title("Best Augmentation Gain per Generator per Dataset\n"
                 "(Sorted by ascending positive rate → severity of imbalance increases right)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "plot_best_gain_by_dataset.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: plot_best_gain_by_dataset.png")

print("\nAll plots saved to results/plots/")
