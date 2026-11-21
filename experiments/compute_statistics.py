"""Statistical tests for the Electronics manuscript, computed from the result files.

Methods (Section 2.2.2 of the manuscript):
  - every p-value is a two-sided paired t-test over seeds, run as a one-sample t-test on the per-seed differences
  - Benjamini-Hochberg false discovery rate correction at q = 0.10, in two separate families:
      Table 3 (24 tests: AUC and F1 for each of 12 dataset-by-generator rows) and the 14 main comparisons of Section 3.5
  - Cohen's d_z = mean(difference) / sd(difference), with ddof=1
  - 95% confidence intervals from the t-distribution
  - power of the paired t-test at n=5 seeds, from the noncentral t distribution

Outputs, in the results directory:
  stats_table3.csv, fdr_family_14.csv, stats_free_baselines.csv, stats_dose_response.csv, stats_power.csv
Usage (from the repo root): python3 experiments/compute_statistics.py [results_dir]
`verify_manuscript_numbers.py` recomputes the same quantities independently and compares them with the text.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
Q = 0.10


def paired(x, y):
    """Mean difference (x - y), d_z and two-sided paired t-test p-value, over the seeds both have."""
    i = sorted(set(x.index) & set(y.index))
    d = (x.loc[i] - y.loc[i]).values.astype(float)
    return d.mean(), d.mean() / d.std(ddof=1), stats.ttest_1samp(d, 0).pvalue, len(d)


def ser(df, method, alpha, col="auc_roc"):
    return df[(df.method == method) & (df.alpha.astype(float) == float(alpha))].set_index("seed")[col]


def best_alpha(df, method, col="auc_roc"):
    g = df[df.method == method].copy()
    g["alpha"] = g.alpha.astype(float)
    return g.groupby("alpha")[col].mean().idxmax()


# ------------------------------------------------------------ Table 3: four control datasets
CTRL = {"Telco": "fullmetrics_telco.csv", "Bank Marketing": "fullmetrics_bank_marketing.csv",
        "German Credit": "fullmetrics_german_credit.csv", "Nomao": "fullmetrics_nomao.csv"}
rows = []
for name, f in CTRL.items():
    d = pd.read_csv(R / f)
    d["alpha"] = d.alpha.astype(float)
    base = d[d.method == "Baseline"].set_index("seed")
    for m in ["GaussianCopula", "CTGAN", "SMOTE"]:
        a = best_alpha(d, m)
        g = d[(d.method == m) & (d.alpha == a)].set_index("seed")
        c = sorted(set(g.index) & set(base.index))
        dauc = (g.loc[c].auc_roc - base.loc[c].auc_roc) * 100
        df1 = g.loc[c].f1_minority - base.loc[c].f1_minority
        rows.append(dict(dataset=name, generator=m, best_alpha=a, n_seeds=len(c),
                         auc_gain_points=dauc.mean(), p_auc=stats.ttest_1samp(dauc, 0).pvalue,
                         f1_gain=df1.mean(), p_f1=stats.ttest_1samp(df1, 0).pvalue,
                         precision_gain=(g.loc[c].precision - base.loc[c].precision).mean(),
                         recall_gain=(g.loc[c].recall - base.loc[c].recall).mean(),
                         accuracy_gain_points=(g.loc[c].accuracy - base.loc[c].accuracy).mean() * 100))
t3 = pd.DataFrame(rows)
pvals = np.concatenate([t3[["p_auc", "p_f1"]].values.ravel()])
rej, padj, _, _ = multipletests(pvals, method="fdr_bh", alpha=Q)
t3["p_auc_fdr"], t3["p_f1_fdr"] = padj[0::2], padj[1::2]
t3["auc_significant_fdr"], t3["f1_significant_fdr"] = rej[0::2], rej[1::2]
t3.to_csv(R / "stats_table3.csv", index=False)

# ------------------------------------------------------------ the 14 main comparisons (Section 3.5)
fam, free = [], []
for ds in ["hillstrom", "criteo"]:
    fm = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv")
    fm["alpha"] = fm.alpha.astype(float)
    base = ser(fm, "Baseline", 0.0)
    a_ct, a_sm = best_alpha(fm, "CTGAN"), best_alpha(fm, "SMOTE")
    fam += [(f"CTGAN vs Baseline {ds}", *paired(ser(fm, "CTGAN", a_ct), base)[:3]),
            (f"SMOTE vs Baseline {ds}", *paired(ser(fm, "SMOTE", a_sm), base)[:3])]
    mc = pd.read_csv(R / f"ci_multi_classifier_{ds}.csv")
    mc = mc[mc.classifier == "GBC"].copy()
    mc["alpha"] = mc.alpha.astype(float)
    b10 = ser(mc, "Baseline", 0.0)
    for nm in ["CTGAN", "SMOTE"]:
        a = max(sorted(set(mc[mc.method == nm].alpha)), key=lambda a: (ser(mc, nm, a) - b10).mean())
        fam.append((f"{nm} vs Baseline {ds} 10-seed", *paired(ser(mc, nm, a), b10)[:3]))
    t = pd.read_csv(R / f"ci_tabddpm_{ds}.csv")
    t["alpha"] = t.alpha.astype(float)
    at = max(sorted(set(t[t.method == "TabDDPM"].alpha)), key=lambda a: (ser(t, "TabDDPM", a) - base).mean())
    fam.append((f"CTGAN vs TabDDPM 2k {ds}", *paired(ser(fm, "CTGAN", a_ct), ser(t, "TabDDPM", at))[:3]))
    bal = pd.read_csv(R / f"ci_balanced_{ds}.csv")
    fam.append((f"CTGAN vs Balanced {ds}", *paired(ser(fm, "CTGAN", a_ct), bal[bal.method == "Balanced"].set_index("seed").auc_roc)[:3]))
    # free baselines against CTGAN (not part of the 14-test family)
    mb = pd.read_csv(R / f"missing_baselines_{ds}.csv")
    mb["alpha"] = pd.to_numeric(mb.alpha, errors="coerce")  # random undersampling is labelled "1:1"
    mb = mb[mb.alpha.notna()]
    for nm, label in [("ADASYN", "ADASYN"), ("BorderlineSMOTE", "Borderline-SMOTE")]:  # (label in the file, name in the paper)
        if (mb.method == nm).any():
            a = best_alpha(mb, nm)
            dm, dz, p, n = paired(ser(mb, nm, a), ser(fm, "CTGAN", a_ct))
            free.append(dict(dataset=ds, method=label, best_alpha=a, vs="CTGAN", diff_points=dm * 100, dz=dz, p=p, n_seeds=n))
    dm, dz, p, n = paired(ser(fm, "SMOTE", a_sm), ser(fm, "CTGAN", a_ct))
    free.append(dict(dataset=ds, method="SMOTE", best_alpha=a_sm, vs="CTGAN", diff_points=dm * 100, dz=dz, p=p, n_seeds=n))
gr = pd.read_csv(R / "ci_great_hillstrom.csv")
for n in (50, 2000):
    x = gr[(gr.n == n) & (gr.method == "GReaT")].set_index("seed").auc
    b = gr[(gr.n == n) & (gr.method == "Baseline")].set_index("seed").auc
    fam.append((f"GReaT vs Baseline n={n}", *paired(x, b)[:3]))
f14 = pd.DataFrame(fam, columns=["test", "delta", "dz", "p"])
f14["delta"] = f14.delta * 100
f14["p_fdr"] = multipletests(f14.p, method="fdr_bh", alpha=Q)[1]
f14["significant_fdr"] = f14.p_fdr < Q
f14.to_csv(R / "fdr_family_14.csv", index=False)
fb = pd.DataFrame(free)
# Smallest difference a paired t-test at n seeds detects with 80% power: d_z at 80% power (below) times the SD of the paired differences.
fb["sd_diff_points"] = (fb.diff_points / fb.dz).abs()
fb["mdd80_points"] = np.nan  # filled in after the power curve is computed
fb.to_csv(R / "stats_free_baselines.csv", index=False)

# ------------------------------------------------------------ dose-response (raw paired t-tests, not in a corrected family)
dr = []
for ds, f in [("Bank Marketing", "dose_response_bank_marketing.csv"), ("Nomao", "dose_response_nomao.csv")]:
    d = pd.read_csv(R / f)
    for mc_ in sorted(d.minority_count.unique()):
        base = d[(d.minority_count == mc_) & (d.method == "Baseline")].set_index("seed").auc_roc
        for m in ["GaussianCopula", "CTGAN", "SMOTE"]:
            x = d[(d.minority_count == mc_) & (d.method == m)].set_index("seed").auc_roc
            if len(x) > 1:
                dm, dz, p, n = paired(x, base)
                dr.append(dict(dataset=ds, minority_count=mc_, method=m, gain_points=dm * 100, dz=dz, p=p, n_seeds=n))
pd.DataFrame(dr).to_csv(R / "stats_dose_response.csv", index=False)

# Bank Marketing dose-response at alpha=0.2 (run_dose_response_bank_alpha.py), including ADASYN
alpha_file = R / "dose_response_bank_marketing_alpha02.csv"
if alpha_file.exists():
    d = pd.read_csv(alpha_file)
    dr2 = []
    for mc_ in sorted(d.minority_count.unique()):
        base = d[(d.minority_count == mc_) & (d.method == "Baseline")].set_index("seed").auc_roc
        for m in ["GaussianCopula", "CTGAN", "SMOTE", "ADASYN"]:
            x = d[(d.minority_count == mc_) & (d.method == m)].set_index("seed").auc_roc
            if len(x) > 1:
                dm, dz, p, n = paired(x, base)
                dr2.append(dict(dataset="Bank Marketing", alpha=0.2, minority_count=mc_, method=m, gain_points=dm * 100, dz=dz, p=p, n_seeds=n))
    pd.DataFrame(dr2).to_csv(R / "stats_dose_response_alpha02.csv", index=False)

# ------------------------------------------------------------ power of the paired t-test at 5 seeds
def power(d, n=5, alpha=0.05):
    df_, nc = n - 1, d * np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, df_)
    return 1 - stats.nct.cdf(t, df_, nc) + stats.nct.cdf(-t, df_, nc)


dgrid = np.round(np.arange(0.5, 3.01, 0.1), 1)
pw = pd.DataFrame({"d_z": dgrid, "power_n5": [power(d) for d in dgrid]})
pw.to_csv(R / "stats_power.csv", index=False)
d80 = float(np.interp(0.8, pw.power_n5.values, pw.d_z.values))
fb["mdd80_points"] = d80 * fb.sd_diff_points
fb.to_csv(R / "stats_free_baselines.csv", index=False)
print(f"Table 3: {len(t3)} rows; 14-test family: {len(f14)} tests; free baselines: {len(free)}; dose-response: {len(dr)}")
print(f"Power 0.80 at d_z about {d80:.2f} (n=5, two-sided alpha 0.05)")
print("FDR-significant in Table 3:", [(r.dataset, r.generator, k) for r in t3.itertuples()
                                       for k in ("auc", "f1") if getattr(r, f"{k}_significant_fdr")])
print("FDR-significant in the 14-test family:", list(f14[f14.significant_fdr].test))
