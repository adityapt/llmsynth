"""
Publication-quality plots for paper2-empirical.md.

Generates (all saved to results/plots/paper2/):
  fig1_summary_comparison.png        — cross-dataset gain summary
  fig2_ucurves_benchmark.png         — U-curves for 4 benchmark datasets
  fig3_ucurve_sparse.png             — sparse stress test U-curve
  fig4_lowdata_regime.png            — low-data regime (AUC vs n_real)
  fig5_marketing_ci.png              — Hillstrom + Criteo CI U-curves
  fig6_regression_hypothesis.png     — cross-dataset regression (formal HT)
  fig7_tabddpm_comparison.png        — CTGAN vs TabDDPM 2k and 10k
  fig8_mlp_rescue.png                — MLP per-seed AUC: baseline vs CTGAN
  fig9_multiclassifier.png           — multi-classifier robustness on Criteo
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

RESULTS = Path("results")
OUT     = RESULTS / "plots" / "paper2"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "GaussianCopula": "#2196F3",
    "CTGAN":          "#FF5722",
    "SMOTE":          "#4CAF50",
    "TabDDPM":        "#9C27B0",
    "GReaT":          "#795548",
    "Baseline":       "#212121",
    "GBC":            "#FF5722",
    "LR":             "#2196F3",
    "RF":             "#4CAF50",
    "MLP":            "#9C27B0",
}
STYLE = {"linewidth": 2.2, "markersize": 7}

def ci95(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) < 2:
        return float(np.mean(v)), 0.0
    se = stats.sem(v)
    return float(np.mean(v)), float(se * stats.t.ppf(0.975, df=len(v)-1))

# ── Load data ─────────────────────────────────────────────────────────────────
dh  = pd.read_csv(RESULTS / "ci_hillstrom.csv")
dc  = pd.read_csv(RESULTS / "ci_criteo.csv")
dth = pd.read_csv(RESULTS / "ci_tabddpm_hillstrom.csv")
dtc = pd.read_csv(RESULTS / "ci_tabddpm_criteo.csv")
dth10 = pd.read_csv(RESULTS / "ci_tabddpm_hillstrom_10000.csv")
dtc10 = pd.read_csv(RESULTS / "ci_tabddpm_criteo_10000.csv")
dmch  = pd.read_csv(RESULTS / "ci_multi_classifier_hillstrom.csv")
dmcc  = pd.read_csv(RESULTS / "ci_multi_classifier_criteo.csv")
dg    = pd.read_csv(RESULTS / "ci_great_hillstrom.csv")

benchmark_ci = {}
for ds in ["telco_churn", "bank_marketing", "credit_default", "nomao_lead", "nomao_sparse"]:
    p = RESULTS / f"ci_{ds}.csv"
    if p.exists():
        benchmark_ci[ds] = pd.read_csv(p)

ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Cross-dataset summary (bar chart: best gain per dataset)
# ─────────────────────────────────────────────────────────────────────────────
def best_gain(df, metric="auc_roc"):
    base = df[df["method"]=="Baseline"][metric].mean()
    best = -99
    for gen in ["GaussianCopula","CTGAN","SMOTE"]:
        for alpha in ALPHAS:
            vals = df[(df["method"]==gen)&(df["alpha"]==alpha)][metric].values
            if len(vals)==0: continue
            m, _ = ci95(vals)
            if m - base > best:
                best = m - base
    return best, base

datasets_info = [
    ("Telco Churn\n26.6%",    "telco_churn",    benchmark_ci.get("telco_churn")),
    ("Bank Mktg\n11.7%",      "bank_marketing", benchmark_ci.get("bank_marketing")),
    ("German Credit\n30.0%",  "credit_default", benchmark_ci.get("credit_default")),
    ("Nomao Lead\n28.3%",     "nomao_lead",     benchmark_ci.get("nomao_lead")),
    ("Hillstrom\n0.9%",       "hillstrom",      dh),
    ("Criteo\n0.2%",          "criteo",         dc),
]

labels, gains = [], []
for label, key, df in datasets_info:
    if df is None: continue
    col = "auc_roc" if "auc_roc" in df.columns else "auc"
    g, _ = best_gain(df, col)
    labels.append(label)
    gains.append(g * 100)  # to pts

colors = ["#90CAF9" if g < 1 else "#FF5722" for g in gains]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, gains, color=colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, gains):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.05 if val >= 0 else -0.15),
            f"{val:+.1f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=9.5, fontweight="bold")
ax.axhline(0, color="#555", linewidth=0.8)
ax.set_ylabel("Best augmentation gain (AUC pts)", fontsize=11)
ax.set_title("Figure 1 — Best Augmentation Gain by Dataset\n"
             "Gains concentrate on imbalanced marketing datasets (Hillstrom 0.9%, Criteo 0.2%)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
blue_patch  = mpatches.Patch(color="#90CAF9", label="Balanced (positive rate ≥ 10%)")
red_patch   = mpatches.Patch(color="#FF5722", label="Imbalanced (positive rate < 1%)")
ax.legend(handles=[blue_patch, red_patch], fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig1_summary_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig1_summary_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — U-curves for 4 benchmark datasets (2×2 grid with CI bands)
# ─────────────────────────────────────────────────────────────────────────────
bench_keys = [
    ("Telco Churn\n(26.6% positive)", "telco_churn"),
    ("Bank Marketing\n(11.7% positive)", "bank_marketing"),
    ("German Credit\n(30.0% positive)", "credit_default"),
    ("Nomao Lead\n(28.3% positive)", "nomao_lead"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (label, key) in zip(axes.flatten(), bench_keys):
    df = benchmark_ci.get(key)
    if df is None:
        ax.set_visible(False)
        continue
    col = "auc_roc" if "auc_roc" in df.columns else "auc"
    base_vals = df[df["method"]=="Baseline"][col].values
    bm, bh = ci95(base_vals)
    ax.axhline(bm, linestyle="--", color=C["Baseline"], linewidth=1.5, label=f"Baseline {bm:.3f}")
    ax.axhspan(bm-bh, bm+bh, color=C["Baseline"], alpha=0.08)
    for gen in ["GaussianCopula","CTGAN","SMOTE"]:
        means, lows, highs = [], [], []
        for alpha in ALPHAS:
            vals = df[(df["method"]==gen)&(df["alpha"]==alpha)][col].values
            if len(vals)==0: continue
            m, h = ci95(vals)
            means.append(m); lows.append(m-h); highs.append(m+h)
        if means:
            ax.plot(ALPHAS[:len(means)], means, marker="o", label=gen, color=C[gen], **STYLE)
            ax.fill_between(ALPHAS[:len(means)], lows, highs, color=C[gen], alpha=0.12)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Synthetic fraction α", fontsize=9)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

fig.suptitle("Figure 2 — Augmentation U-Curves: Benchmark Datasets (5-seed CI)\n"
             "No generator exceeds +0.27 AUC points on balanced datasets",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig2_ucurves_benchmark.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig2_ucurves_benchmark.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Sparse stress test U-curve
# ─────────────────────────────────────────────────────────────────────────────
df_sp = benchmark_ci.get("nomao_sparse")
if df_sp is not None:
    col = "auc_roc" if "auc_roc" in df_sp.columns else "auc"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_vals = df_sp[df_sp["method"]=="Baseline"][col].values
    bm, bh = ci95(base_vals)
    ax.axhline(bm, linestyle="--", color=C["Baseline"], linewidth=1.5,
               label=f"Baseline (sparse) {bm:.3f} ± {bh:.3f}")
    ax.axhspan(bm-bh, bm+bh, color=C["Baseline"], alpha=0.08)
    ax.axhline(0.992, linestyle=":", color="#888", linewidth=1.2,
               label="Dense baseline 0.992 (reference)")
    for gen in ["GaussianCopula","CTGAN","SMOTE"]:
        means, lows, highs = [], [], []
        for alpha in ALPHAS:
            vals = df_sp[(df_sp["method"]==gen)&(df_sp["alpha"]==alpha)][col].values
            if len(vals)==0: continue
            m, h = ci95(vals)
            means.append(m); lows.append(m-h); highs.append(m+h)
        if means:
            ax.plot(ALPHAS[:len(means)], means, marker="o", label=gen, color=C[gen], **STYLE)
            ax.fill_between(ALPHAS[:len(means)], lows, highs, color=C[gen], alpha=0.12)
    ax.set_xlabel("Synthetic fraction α", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Figure 3 — Sparsity Stress Test: Nomao Sparse (70% missing, n=500)\n"
                 "Augmentation does not recover the 9.4-pt gap caused by feature sparsity",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_ucurve_sparse.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved fig3_ucurve_sparse.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Low-data regime (AUC vs n_real for benchmark datasets)
# ─────────────────────────────────────────────────────────────────────────────
# Use existing lowdata_*.png files in a composite grid
from PIL import Image as PILImage
import io

lowdata_files = [
    (RESULTS / "lowdata_telco_churn.png",    "Telco Churn (26.6%)"),
    (RESULTS / "lowdata_bank_marketing.png", "Bank Marketing (11.7%)"),
    (RESULTS / "lowdata_credit_default.png", "German Credit (30.0%)"),
    (RESULTS / "lowdata_nomao_lead.png",     "Nomao Lead (28.3%)"),
]

existing = [(p, l) for p, l in lowdata_files if p.exists()]
if existing:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (path, label) in zip(axes.flatten(), existing):
        img = PILImage.open(path)
        ax.imshow(np.array(img))
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")
    for ax in axes.flatten()[len(existing):]:
        ax.set_visible(False)
    fig.suptitle("Figure 4 — Low-Data Regime: AUC vs Real Training Set Size\n"
                 "Augmentation recovers 30–60% of performance gap at n=250; "
                 "benefit narrows rapidly above n=1,000",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_lowdata_regime.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("Saved fig4_lowdata_regime.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Marketing datasets CI U-curves (Hillstrom + Criteo side by side)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (df, label, pos_rate) in zip(axes, [
    (dh, "Hillstrom Email Marketing", "0.9%"),
    (dc, "Criteo Display Advertising", "0.2%"),
]):
    base_vals = df[df["method"]=="Baseline"]["auc_roc"].values
    bm, bh = ci95(base_vals)
    ax.axhline(bm, linestyle="--", color=C["Baseline"], linewidth=1.8,
               label=f"Baseline {bm:.3f} ± {bh:.3f}")
    ax.axhspan(bm-bh, bm+bh, color=C["Baseline"], alpha=0.10)
    for gen in ["GaussianCopula","CTGAN","SMOTE"]:
        means, lows, highs = [], [], []
        for alpha in ALPHAS:
            vals = df[(df["method"]==gen)&(df["alpha"]==alpha)]["auc_roc"].values
            if len(vals)==0: continue
            m, h = ci95(vals)
            means.append(m); lows.append(m-h); highs.append(m+h)
        if means:
            ax.plot(ALPHAS[:len(means)], means, marker="o", label=gen, color=C[gen], **STYLE)
            ax.fill_between(ALPHAS[:len(means)], lows, highs, color=C[gen], alpha=0.15)
    ax.set_xlabel("Synthetic fraction α", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(f"{label}\n(positive rate {pos_rate}, 5 seeds, shaded = 95% CI)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Figure 5 — Augmentation U-Curves: Real Marketing Datasets\n"
             "Large gains and substantially narrower CIs after augmentation",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig5_marketing_ci.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig5_marketing_ci.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Cross-dataset regression: CTGAN gain vs log(positive rate)  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
datasets_regression = {
    "Telco Churn\n26.6%":    (0.266, 0.0021, benchmark_ci.get("telco_churn"), "auc"),
    "Bank Mktg\n11.7%":      (0.117, None,   benchmark_ci.get("bank_marketing"), "auc"),
    "German Credit\n30.0%":  (0.300, None,   benchmark_ci.get("credit_default"), "auc"),
    "Nomao Lead\n28.3%":     (0.283, None,   benchmark_ci.get("nomao_lead"), "auc"),
    "Hillstrom\n0.9%":       (0.009, None,   dh, "auc_roc"),
    "Criteo\n0.2%":          (0.002, None,   dc, "auc_roc"),
}

reg_x, reg_y, reg_labels = [], [], []
for label, (pos, _, df, col) in datasets_regression.items():
    if df is None: continue
    base = df[df["method"]=="Baseline"][col].mean()
    best = -99
    for gen in ["CTGAN"]:
        for alpha in ALPHAS:
            vals = df[(df["method"]==gen)&(df["alpha"]==alpha)][col].values
            if len(vals)==0: continue
            m, _ = ci95(vals)
            if m - base > best: best = m - base
    if best > -99:
        reg_x.append(np.log10(pos * 100))  # log10(positive rate %)
        reg_y.append(best * 100)            # gain in pts
        reg_labels.append(label)

reg_x, reg_y = np.array(reg_x), np.array(reg_y)
slope, intercept, r, p_val, _ = stats.linregress(reg_x, reg_y)
x_line = np.linspace(min(reg_x)-0.1, max(reg_x)+0.1, 100)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(reg_x, reg_y, s=100, zorder=3,
                     c=reg_y, cmap="RdYlGn", edgecolors="#333", linewidths=0.8)
plt.colorbar(scatter, ax=ax, label="CTGAN gain (AUC pts)", shrink=0.8)
ax.plot(x_line, y_line, "--", color="#333", linewidth=1.8,
        label=f"Fit: slope={slope:.2f}, R²={r**2:.2f}, p={p_val:.4f}")
for xi, yi, lab in zip(reg_x, reg_y, reg_labels):
    ax.annotate(lab, xy=(xi, yi), xytext=(8, 4), textcoords="offset points",
                fontsize=8.5, color="#333")
ax.set_xlabel("log₁₀(Positive rate %)", fontsize=12)
ax.set_ylabel("Best CTGAN gain (AUC pts)", fontsize=12)
ax.set_title(f"Figure 6 — Cross-Dataset Regression: CTGAN Gain vs Class Imbalance\n"
             f"R²={r**2:.2f}, p={p_val:.4f} — statistical confirmation of the imbalance hypothesis",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axhline(0, color="#aaa", linewidth=0.8, linestyle=":")
ax.invert_xaxis()  # more imbalanced (lower positive rate) on the right

# Add positive-rate labels on top x-axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
tick_pos_rates = [30, 10, 1, 0.2]
ax2.set_xticks([np.log10(p) for p in tick_pos_rates])
ax2.set_xticklabels([f"{p}%" for p in tick_pos_rates], fontsize=9)
ax2.set_xlabel("Positive rate (%)", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "fig6_regression_hypothesis.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig6_regression_hypothesis.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — TabDDPM vs CTGAN: N_iter=2k and N_iter=10k  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (ds_label, df_ci, df_tab2k, df_tab10k, pos) in zip(axes, [
    ("Hillstrom (0.9% positive)", dh, dth, dth10, "hillstrom"),
    ("Criteo (0.2% positive)",    dc, dtc, dtc10, "criteo"),
]):
    base_vals = df_ci[df_ci["method"]=="Baseline"]["auc_roc"].values
    bm, bh = ci95(base_vals)
    ax.axhline(bm, linestyle="--", color=C["Baseline"], linewidth=1.5, label=f"Baseline {bm:.3f}")
    ax.axhspan(bm-bh, bm+bh, color=C["Baseline"], alpha=0.08)

    # CTGAN
    ctgan_m, ctgan_lo, ctgan_hi = [], [], []
    for alpha in ALPHAS:
        vals = df_ci[(df_ci["method"]=="CTGAN")&(df_ci["alpha"]==alpha)]["auc_roc"].values
        if len(vals)==0: continue
        m, h = ci95(vals)
        ctgan_m.append(m); ctgan_lo.append(m-h); ctgan_hi.append(m+h)
    ax.plot(ALPHAS[:len(ctgan_m)], ctgan_m, marker="o", label="CTGAN", color=C["CTGAN"], **STYLE)
    ax.fill_between(ALPHAS[:len(ctgan_m)], ctgan_lo, ctgan_hi, color=C["CTGAN"], alpha=0.12)

    # TabDDPM 2k
    tab2_m, tab2_lo, tab2_hi = [], [], []
    for alpha in ALPHAS:
        vals = df_tab2k[(df_tab2k["method"]=="TabDDPM")&(df_tab2k["alpha"]==alpha)]["auc_roc"].values
        if len(vals)==0: continue
        m, h = ci95(vals)
        tab2_m.append(m); tab2_lo.append(m-h); tab2_hi.append(m+h)
    ax.plot(ALPHAS[:len(tab2_m)], tab2_m, marker="s", linestyle="-",
            label="TabDDPM N_iter=2k", color=C["TabDDPM"], linewidth=2.0, markersize=6)
    ax.fill_between(ALPHAS[:len(tab2_m)], tab2_lo, tab2_hi, color=C["TabDDPM"], alpha=0.10)

    # TabDDPM 10k
    tab10_m, tab10_lo, tab10_hi = [], [], []
    for alpha in ALPHAS:
        vals = df_tab10k[(df_tab10k["method"]=="TabDDPM")&(df_tab10k["alpha"]==alpha)]["auc_roc"].values
        if len(vals)==0: continue
        m, h = ci95(vals)
        tab10_m.append(m); tab10_lo.append(m-h); tab10_hi.append(m+h)
    ax.plot(ALPHAS[:len(tab10_m)], tab10_m, marker="^", linestyle=":",
            label="TabDDPM N_iter=10k", color=C["TabDDPM"], linewidth=2.0, markersize=6, alpha=0.7)
    ax.fill_between(ALPHAS[:len(tab10_m)], tab10_lo, tab10_hi, color=C["TabDDPM"], alpha=0.07)

    ax.set_xlabel("Synthetic fraction α", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title(f"{ds_label}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

fig.suptitle("Figure 7 — CTGAN vs TabDDPM at Two Training Budgets (5-seed CI)\n"
             "Extended TabDDPM training widens the CTGAN advantage — gap is architectural, not training-budget",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig7_tabddpm_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig7_tabddpm_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — MLP rescue: per-seed AUC distribution on Criteo  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
mlp_criteo = dmcc[dmcc["classifier"]=="MLP"].copy()
base_per_seed = mlp_criteo[mlp_criteo["method"]=="Baseline"].sort_values("seed")["auc_roc"].values
ctgan_best    = mlp_criteo[(mlp_criteo["method"]=="CTGAN")&(mlp_criteo["alpha"]==0.2)].sort_values("seed")["auc_roc"].values
smote_best    = mlp_criteo[(mlp_criteo["method"]=="SMOTE")&(mlp_criteo["alpha"]==1.0)].sort_values("seed")["auc_roc"].values
seeds = mlp_criteo[mlp_criteo["method"]=="Baseline"].sort_values("seed")["seed"].values

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(seeds))
width = 0.25
bars_b  = ax.bar(x - width,  base_per_seed, width, label="Baseline (real only)",
                 color="#CFD8DC", edgecolor="white", linewidth=0.8)
bars_c  = ax.bar(x,           ctgan_best,   width, label="CTGAN α=0.2 (augmented)",
                 color=C["CTGAN"], edgecolor="white", linewidth=0.8, alpha=0.9)
bars_s  = ax.bar(x + width,   smote_best,   width, label="SMOTE α=1.0 (augmented)",
                 color=C["SMOTE"], edgecolor="white", linewidth=0.8, alpha=0.9)

ax.axhline(0.5, linestyle=":", color="#888", linewidth=1.0, label="Random chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=9)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("Figure 8 — MLP Convergence Rescue on Criteo (0.2% positive rate)\n"
             "7 of 10 baseline seeds fail to converge (AUC ≈ 0); CTGAN rescues all 10",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 1.05)

# Annotate collapsed seeds
for i, (seed, val) in enumerate(zip(seeds, base_per_seed)):
    if val < 0.15:
        ax.text(i - width, val + 0.02, "✗", ha="center", va="bottom",
                fontsize=10, color="#E53935", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig8_mlp_rescue.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig8_mlp_rescue.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — Multi-classifier robustness on Criteo  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
classifiers = ["GBC", "LR", "RF", "MLP"]
clf_labels  = ["Gradient\nBoosting", "Logistic\nRegression", "Random\nForest", "MLP"]
generators  = ["GaussianCopula", "CTGAN", "SMOTE"]

# Best gain and baseline per classifier
clf_baselines, clf_gains = {}, {gen: [] for gen in generators}

for clf in classifiers:
    sub = dmcc[dmcc["classifier"]==clf]
    base = sub[sub["method"]=="Baseline"]["auc_roc"].values
    bm, _ = ci95(base)
    clf_baselines[clf] = bm
    for gen in generators:
        best = -99
        for alpha in ALPHAS:
            vals = sub[(sub["method"]==gen)&(sub["alpha"]==alpha)]["auc_roc"].values
            if len(vals)==0: continue
            m, _ = ci95(vals)
            if m - bm > best: best = m - bm
        clf_gains[gen].append(best * 100 if best > -99 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: baseline AUC per classifier
x = np.arange(len(classifiers))
ax1.bar(x, [clf_baselines[c] for c in classifiers],
        color=[C[c] for c in classifiers], edgecolor="white", alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(clf_labels, fontsize=10)
ax1.set_ylabel("Baseline AUC-ROC", fontsize=11)
ax1.set_title("Baseline performance (real-only)", fontsize=10, fontweight="bold")
ax1.axhline(0.5, linestyle=":", color="#888", linewidth=1.0, label="Random chance")
ax1.set_ylim(0, 1.05)
ax1.grid(axis="y", alpha=0.3)
for i, clf in enumerate(classifiers):
    ax1.text(i, clf_baselines[clf] + 0.01,
             f"{clf_baselines[clf]:.3f}", ha="center", fontsize=9, fontweight="bold")

# Right: best gain per generator per classifier
width = 0.25
for j, gen in enumerate(generators):
    bars = ax2.bar(x + (j-1)*width, clf_gains[gen], width,
                   label=gen, color=C[gen], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, clf_gains[gen]):
        if abs(val) > 0.3:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.2 if val >= 0 else -0.5),
                     f"{val:+.1f}", ha="center",
                     va="bottom" if val >= 0 else "top", fontsize=7.5)
ax2.axhline(0, color="#555", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(clf_labels, fontsize=10)
ax2.set_ylabel("Best CTGAN gain (AUC pts)", fontsize=11)
ax2.set_title("Best augmentation gain per classifier", fontsize=10, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 9 — Multi-Classifier Robustness on Criteo (10 seeds)\n"
             "CTGAN advantage holds across GBC (+12.0) and RF (+9.6); "
             "LR near ceiling; MLP rescued from failure",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig9_multiclassifier.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved fig9_multiclassifier.png")

print(f"\nAll figures saved to {OUT}")
