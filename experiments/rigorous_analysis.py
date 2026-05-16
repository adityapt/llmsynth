"""
Phase 1 rigorous re-analysis of §6.6 GReaT-vs-Baseline experiments.

Implements the pre-registered analysis plan in `papers/improvement_plan.md`:
  - Paired t-CIs alongside independent CIs
  - Cohen's d_z paired effect sizes
  - Benjamini-Hochberg FDR correction across the 16-test family
  - Jackknife sensitivity (drop-one-seed) on headline findings
  - TOST equivalence test at |Δ| < 0.5 AUC pts for "tied" claims

Outputs:
  - results/rigorous_analysis.csv    (tidy per-cell stats, all 18 cells)
  - results/rigorous_analysis.md     (human-readable summary)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sc
from statsmodels.stats.multitest import multipletests

RESULTS = Path("results")

DATASETS = [
    ("German Credit", "great_german_results.csv"),
    ("Telco Churn",   "great_telco_results.csv"),
    ("Hillstrom",     "ci_great_hillstrom.csv"),
]

EQUIV_BOUND_AUC = 0.005   # |Δ| < 0.5 AUC pts is "equivalent"
ALPHA = 0.05
FDR_Q = 0.10


def paired_stats(baseline, great):
    """Paired stats on matched-seed AUCs. Returns dict of per-cell metrics."""
    delta = great - baseline
    n = len(delta)
    mean_d = delta.mean()
    sd_d   = delta.std(ddof=1)
    sem_d  = sd_d / np.sqrt(n)
    t_crit = sc.t.ppf(0.975, df=n-1)
    paired_ci = t_crit * sem_d

    # Paired t-test vs H0: mean_d = 0
    if sem_d > 0:
        t_stat = mean_d / sem_d
        p_paired = 2 * (1 - sc.t.cdf(abs(t_stat), df=n-1))
    else:
        t_stat, p_paired = np.nan, np.nan

    # Cohen's d_z (paired effect size)
    d_z = mean_d / sd_d if sd_d > 0 else np.nan

    # TOST equivalence test (two one-sided tests at ±EQUIV_BOUND_AUC)
    # H0: |mean_d| > EQUIV_BOUND_AUC; reject if both one-sided tests reject.
    if sem_d > 0:
        t_lower = (mean_d - (-EQUIV_BOUND_AUC)) / sem_d
        t_upper = ((+EQUIV_BOUND_AUC) - mean_d) / sem_d
        p_lower = 1 - sc.t.cdf(t_lower, df=n-1)
        p_upper = 1 - sc.t.cdf(t_upper, df=n-1)
        p_tost = max(p_lower, p_upper)  # TOST p = max of the two one-sided p's
    else:
        p_tost = np.nan

    # Independent CIs for each method (for backward comparison with paper tables)
    bl_ci = t_crit * baseline.std(ddof=1) / np.sqrt(len(baseline))
    gr_ci = t_crit * great.std(ddof=1)    / np.sqrt(len(great))

    # Win rate
    wins = int((delta > 0).sum())

    # Jackknife: drop each seed, recompute mean_d. Report range.
    jack_means = []
    for i in range(n):
        d_drop = np.delete(delta, i)
        jack_means.append(d_drop.mean())
    jack_min, jack_max = min(jack_means), max(jack_means)
    # Does dropping any single seed flip the sign?
    sign_stable = (jack_min * jack_max) > 0  # both same sign

    return {
        "n_seeds": n,
        "baseline_mean": baseline.mean(),
        "baseline_ci95_indep": bl_ci,
        "great_mean": great.mean(),
        "great_ci95_indep": gr_ci,
        "delta_mean": mean_d,
        "delta_ci95_paired": paired_ci,
        "delta_sd": sd_d,
        "cohen_dz": d_z,
        "t_stat": t_stat,
        "p_paired": p_paired,
        "wins": wins,
        "p_tost_equiv_0.5pts": p_tost,
        "jack_delta_min": jack_min,
        "jack_delta_max": jack_max,
        "jack_sign_stable": sign_stable,
    }


def analyze_dataset(name, fname):
    """Load CSV, compute per-cell paired stats."""
    df = pd.read_csv(RESULTS / fname)
    df = df[df["method"].isin(["Baseline", "GReaT"])].copy()
    rows = []
    for n in sorted(df["n"].unique()):
        bl = df[(df["n"] == n) & (df["method"] == "Baseline")].sort_values("seed")
        gr = df[(df["n"] == n) & (df["method"] == "GReaT")].sort_values("seed")
        # Ensure aligned seeds
        if not (bl["seed"].values == gr["seed"].values).all():
            raise ValueError(f"Seed mismatch in {name} n={n}")
        stats = paired_stats(bl["auc"].values, gr["auc"].values)
        stats.update({"dataset": name, "n": n})
        rows.append(stats)
    return pd.DataFrame(rows)


def main():
    all_stats = pd.concat(
        [analyze_dataset(name, fname) for name, fname in DATASETS],
        ignore_index=True
    )

    # Apply BH-FDR across the family of paired tests
    # Family size: German has 4 n's, Telco has 6, Hillstrom has 6 = 16 paired tests
    valid = all_stats["p_paired"].notna()
    reject, p_fdr, _, _ = multipletests(
        all_stats.loc[valid, "p_paired"].values,
        alpha=FDR_Q,
        method="fdr_bh",
    )
    all_stats.loc[valid, "p_fdr_bh"] = p_fdr
    all_stats.loc[valid, "fdr_reject_q0.10"] = reject

    # Apply Bonferroni for reference (more conservative)
    n_tests = valid.sum()
    all_stats.loc[valid, "p_bonferroni"] = np.minimum(
        all_stats.loc[valid, "p_paired"] * n_tests, 1.0
    )

    # Reorder columns
    cols = ["dataset", "n", "n_seeds",
            "baseline_mean", "baseline_ci95_indep",
            "great_mean", "great_ci95_indep",
            "delta_mean", "delta_ci95_paired", "delta_sd",
            "cohen_dz", "wins",
            "t_stat", "p_paired", "p_fdr_bh", "p_bonferroni",
            "fdr_reject_q0.10", "p_tost_equiv_0.5pts",
            "jack_delta_min", "jack_delta_max", "jack_sign_stable"]
    all_stats = all_stats[cols]

    # Round for display
    out = all_stats.copy()
    for c in ["baseline_mean", "baseline_ci95_indep",
              "great_mean", "great_ci95_indep",
              "delta_mean", "delta_ci95_paired", "delta_sd",
              "cohen_dz", "t_stat",
              "jack_delta_min", "jack_delta_max"]:
        out[c] = out[c].round(4)
    for c in ["p_paired", "p_fdr_bh", "p_bonferroni", "p_tost_equiv_0.5pts"]:
        out[c] = out[c].round(4)

    # Save tidy CSV
    out.to_csv(RESULTS / "rigorous_analysis.csv", index=False)
    print(f"Saved: {RESULTS / 'rigorous_analysis.csv'}")

    # Produce human-readable markdown summary
    lines = []
    lines.append("# Phase 1 Rigorous Re-Analysis — §6.6 GReaT Findings\n")
    lines.append("**Pre-registered analysis plan:** `papers/improvement_plan.md` (2026-05-16)\n")
    lines.append(f"**Family size for FDR:** {n_tests} paired tests across "
                 f"{out['dataset'].nunique()} datasets × variable n.")
    lines.append(f"**FDR target:** Benjamini-Hochberg q={FDR_Q}.")
    lines.append(f"**Equivalence bound (TOST):** |Δ| < {EQUIV_BOUND_AUC*100:.1f} AUC pts.\n")

    lines.append("## Headline finding decision rule\n")
    telco_50 = out[(out["dataset"] == "Telco Churn") & (out["n"] == 50)].iloc[0]
    lines.append(
        f"- **Telco n=50:** Δ={telco_50['delta_mean']*100:+.2f} pts, "
        f"paired t={telco_50['t_stat']:+.2f}, "
        f"raw p={telco_50['p_paired']:.4f}, "
        f"BH-FDR p={telco_50['p_fdr_bh']:.4f}, "
        f"Bonferroni p={telco_50['p_bonferroni']:.4f}, "
        f"d_z={telco_50['cohen_dz']:.2f}, "
        f"wins={int(telco_50['wins'])}/{int(telco_50['n_seeds'])}, "
        f"jack-stable={telco_50['jack_sign_stable']}"
    )
    fdr_decision = (
        "✓ Headline finding survives BH-FDR. Phase 3 (seed expansion) is OPTIONAL."
        if telco_50.get("fdr_reject_q0.10", False)
        else "✗ Headline finding does NOT survive BH-FDR. Phase 3 (seed expansion) is RECOMMENDED."
    )
    lines.append(f"\n**Decision:** {fdr_decision}\n")

    lines.append("## Full per-cell results\n")
    # Markdown table
    md_cols = ["dataset", "n", "delta_mean", "delta_ci95_paired",
               "cohen_dz", "wins", "p_paired", "p_fdr_bh",
               "p_tost_equiv_0.5pts", "jack_sign_stable"]
    md = out[md_cols].copy()
    md.columns = ["dataset", "n", "Δ (mean)", "Δ ±CI (paired)",
                  "d_z", "wins/5", "p (paired)", "p (FDR)",
                  "p (TOST equiv)", "jack-stable"]
    md["wins/5"] = md["wins/5"].astype(str) + "/5"
    # Manual markdown table (tabulate version mismatch on local env)
    header = "| " + " | ".join(md.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(md.columns)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in md.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- **Δ (mean) > 0** = GReaT helps (mean AUC gain in proportion).")
    lines.append("- **Δ ±CI (paired)** = 95% CI on the paired difference. If CI excludes zero, "
                 "the paired test rejects H0:Δ=0 at α=0.05 (unadjusted).")
    lines.append("- **d_z** = paired effect size (Cohen's d_z). |d_z| > 0.8 = large; "
                 "0.5–0.8 = medium; 0.2–0.5 = small.")
    lines.append("- **p (paired)** = raw paired t-test p-value (matched-seed comparison).")
    lines.append("- **p (FDR)** = BH-FDR-adjusted p across the 16-test family. "
                 f"Reject if p (FDR) < {FDR_Q}.")
    lines.append("- **p (TOST equiv)** = two-one-sided-test p for equivalence within "
                 f"±{EQUIV_BOUND_AUC*100:.1f} AUC pts. "
                 f"If p (TOST) < 0.05, statistically confirmed as 'tied'.")
    lines.append("- **jack-stable** = does dropping any single seed leave the sign of Δ unchanged?")
    lines.append("")

    (RESULTS / "rigorous_analysis.md").write_text("\n".join(lines))
    print(f"Saved: {RESULTS / 'rigorous_analysis.md'}")


if __name__ == "__main__":
    main()
