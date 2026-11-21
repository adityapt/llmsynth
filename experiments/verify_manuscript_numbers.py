"""Recompute the numbers in the manuscript (given as a markdown file) from the result files and compare.

Usage (from the repo root): python3 experiments/verify_manuscript_numbers.py <manuscript.md>
Prints one line per check (PASS/FAIL) and a summary. Table cells are compared numerically,
text numbers are compared as formatted strings that must appear in the manuscript.
"""
import re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

R = Path("results")
MD_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("manuscript.md")
MD = MD_PATH.read_text()
BODY = MD.split("## References")[0]
FAILS, PASSES = [], 0


def ok(label, cond, detail=""):
    global PASSES
    if cond: PASSES += 1
    else: FAILS.append(f"{label} {detail}"); print("FAIL", label, detail)


def has(label, expected):
    expected = expected.replace("—", "-").replace("–", "-").replace("−", "-")
    ok(label, expected in BODY, f"(expected text not found: {expected!r})")


def tci(x):
    x = np.asarray(x, float); return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def table_rows(caption_prefix):
    """Rows (list of cell lists) of the markdown table that follows a caption line."""
    i = BODY.index(caption_prefix)
    lines = BODY[i:].split("\n")
    rows, started = [], False
    for ln in lines[1:]:
        if ln.startswith("|"):
            started = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "): rows.append(cells)
        elif started: break
    return rows[1:]  # drop header


def num(cell):
    t = cell.replace("−", "-").replace("**", "").replace(",", "")
    m = re.search(r"-?\d+\.?\d*", t)
    return float(m.group(0)) if m else None


def near(a, b, tol): return a is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- Table 1 / dataset counts
from experiments.synthetic_data_eval import load_telco_churn, load_credit_default, load_bank_marketing
from experiments.run_hillstrom import load_hillstrom
from experiments.run_nomao import load_nomao
info = {}
for name, fn in [("Telco", load_telco_churn), ("German", load_credit_default), ("Bank", load_bank_marketing), ("Hillstrom", load_hillstrom), ("Nomao", load_nomao)]:
    r = fn(); df, t = r[0], r[1]; info[name] = (len(df), int(df[t].sum()), df[t].mean() * 100, df.shape[1] - 1)
has("Telco rows and rate", f"| Telco Customer Churn | {info['Telco'][0]:,} | {info['Telco'][2]:.1f}% |")
has("German rate", f"| German Credit | 1,000 | {info['German'][2]:.1f}% |")
# removed with the sentence it checked (cut for length): has("Bank full source", f"{info['Bank'][0]:,} rows and {info['Bank'][1
# removed with the sentence it checked (cut for length): has("Nomao full source", f"{info['Nomao'][0]:,} rows and {info['Nomao'
has("Hillstrom size", f"{info['Hillstrom'][0]:,}")
ok("Hillstrom rate rounds to 0.9", round(info['Hillstrom'][2], 1) == 0.9)
has("Hillstrom row", "| **Hillstrom Email Marketing** | **10,000** | **0.9%** |")
has("Nomao features", f"{info['Nomao'][3]} features against {info['Bank'][3]} for Bank Marketing")
from experiments.run_criteo import load_criteo_uplift
_c = load_criteo_uplift(); crit_rate = _c[0][_c[1]].mean() * 100
ok("Criteo positive rate rounds to 0.3%", round(crit_rate, 1) == 0.3, f"(rate {crit_rate:.3f}%)")
has("Criteo row", "| **Criteo Uplift (conversion)** | **10,000** | **0.3%** |")
# row caps stated in Table 1 come from the run scripts
for fname, cap in [("run_full_metrics_benchmark_datasets.py", "15_000"), ("_fullmetrics_nomao_worker.py", "10_000"), ("run_criteo.py", "10_000")]:
    ok(f"cap {cap} in {fname}", re.search(r"N_CAP\s*=\s*" + cap.replace("_", "_?"), Path("experiments", fname).read_text()) is not None)
has("Bank cap row", "| Bank Marketing | 15,000 | 11.7% |"); has("Nomao cap row", "| Nomao Lead (full) | 10,000 | 28.3% |")
# Bank duration check
bd = pd.read_csv(R / "bank_duration_check.csv"); g = bd.groupby("features").auc_roc.mean()
ok("Bank AUC with and without V12 (not in text)", g['with V12 (as run)'] > g['without V12'])

# ---------------------------------------------------------------- Table 2
cd = pd.read_csv(R / "synthetic_class_distribution_5seed.csv").set_index("dataset")
for ds, col in [("Hillstrom Email", 1), ("Criteo Display", 2)]:
    r = cd.loc[ds]
    rows = {c[0].replace("*", ""): c for c in table_rows("**Table 2.")}
    ok(f"T2 real rate {ds}", near(num(rows["Real training data"][col]), r.real_positive_rate, 0.006))
    ok(f"T2 GC {ds}", near(num(rows["GaussianCopula"][col]), r.GaussianCopula_mean, 0.006))
    ok(f"T2 CTGAN {ds}", near(num(rows["CTGAN"][col]), r.CTGAN_mean, 0.006))
ratio_h = cd.loc["Hillstrom Email"].CTGAN_mean / 0.90; ratio_c = cd.loc["Criteo Display"].CTGAN_mean / 0.30
ok("7 to 89x enrichment", round(ratio_h) == 7 and round(ratio_c) == 89, f"({ratio_h:.1f}, {ratio_c:.1f})")

# ---------------------------------------------------------------- control datasets (latest run)
CTRL = {"Telco": "fullmetrics_telco.csv", "Bank Marketing": "fullmetrics_bank_marketing.csv", "German Credit": "fullmetrics_german_credit.csv", "Nomao": "fullmetrics_nomao.csv"}
tests, cache = [], {}
for name, f in CTRL.items():
    d = pd.read_csv(R / f); d["alpha"] = d.alpha.astype(float); base = d[d.method == "Baseline"].set_index("seed")
    for m in ["GaussianCopula", "CTGAN", "SMOTE"]:
        a = d[d.method == m].groupby("alpha").auc_roc.mean().idxmax()
        g = d[(d.method == m) & (d.alpha == a)].set_index("seed"); c = sorted(set(g.index) & set(base.index))
        dauc = (g.loc[c].auc_roc - base.loc[c].auc_roc) * 100; df1 = g.loc[c].f1_minority - base.loc[c].f1_minority
        cache[(name, m)] = dict(alpha=a, dauc=dauc.mean(), pauc=stats.ttest_1samp(dauc, 0).pvalue, df1=df1.mean(), pf1=stats.ttest_1samp(df1, 0).pvalue,
                                dp=(g.loc[c].precision - base.loc[c].precision).mean(), dr=(g.loc[c].recall - base.loc[c].recall).mean(),
                                dacc=(g.loc[c].accuracy - base.loc[c].accuracy).mean() * 100)
        tests += [(name, m, "auc", cache[(name, m)]["pauc"]), (name, m, "f1", cache[(name, m)]["pf1"])]
pf = multipletests([t[3] for t in tests], method="fdr_bh", alpha=0.10)
sig = {(t[0], t[1], t[2]) for t, r in zip(tests, pf[0]) if r}
ok("Table 3 FDR-significant set (24 tests): five comparisons", sig == {("Bank Marketing", "GaussianCopula", "f1"), ("Bank Marketing", "CTGAN", "auc"), ("Bank Marketing", "CTGAN", "f1"), ("Bank Marketing", "SMOTE", "f1"), ("Nomao", "CTGAN", "auc")}, f"(got {sorted(sig)})")
for row in table_rows("**Table 3."):
    ds = row[0]; mm = row[1].split(" (")[0]; a = float(re.search(r"\(([\d.]+)\)", row[1]).group(1)); c = cache[(ds, mm)]
    ok(f"T3 alpha {ds} {mm}", near(a, c["alpha"], 1e-9))
    for lab, cell, val, tol in [("AUC gain", row[2], c["dauc"], 0.006), ("p(AUC)", row[3], c["pauc"], 0.0006), ("F1 gain", row[4], c["df1"], 0.00006), ("p(F1)", row[5], c["pf1"], 0.0006),
                                ("prec gain", row[6], c["dp"], 0.0006), ("rec gain", row[7], c["dr"], 0.0006), ("acc gain", row[8], c["dacc"], 0.006)]:
        ok(f"T3 {ds} {mm} {lab}", near(num(cell), val, tol), f"(table {cell} vs data {val:.4f})")

# ---------------------------------------------------------------- Table 5c-f and 5a-b
def best_row(d, name, acol="alpha"):
    g = d[d.method == name]
    if acol in g and g[acol].nunique() > 1:
        key = g.groupby(acol).auc_roc.mean().idxmax(); g = g[g[acol] == key]
    return g
for cap, f in [("**Table 5c.", "fullmetrics_telco.csv"), ("**Table 5d.", "fullmetrics_bank_marketing.csv"), ("**Table 5e.", "fullmetrics_german_credit.csv"), ("**Table 5f.", "fullmetrics_nomao.csv")]:
    d = pd.read_csv(R / f); d["alpha"] = d.alpha.astype(float)
    for row in table_rows(cap):
        g = best_row(d, row[0]); vals = [g.f1_minority.mean(), g.precision.mean(), g.recall.mean(), g.accuracy.mean() * 100]
        for lab, cell, v, tol in zip(["F1", "P", "R", "Acc"], row[1:], vals, [0.0006, 0.0006, 0.0006, 0.06]):
            ok(f"{cap} {row[0]} {lab}", near(num(cell), v, tol), f"(table {cell} vs data {v:.4f})")
for cap, ds in [("**Table 5a.", "hillstrom"), ("**Table 5b.", "criteo")]:
    fm = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); fm["alpha"] = fm.alpha.astype(float)
    mb = pd.read_csv(R / f"missing_baselines_{ds}.csv"); bal = pd.read_csv(R / f"fullmetrics_balanced_{ds}.csv"); bal["method"] = "class_weight_balanced"
    src = {"Baseline": fm, "CTGAN": fm, "SMOTE": fm, "GaussianCopula": fm, "ADASYN": mb, "Borderline-SMOTE": mb, "Random undersampling": mb, "`class_weight='balanced'`": bal}
    mname = {"Borderline-SMOTE": "BorderlineSMOTE", "Random undersampling": "RandomUnderSampler", "`class_weight='balanced'`": "class_weight_balanced"}
    for row in table_rows(cap):
        d = src[row[0]]; g = best_row(d, mname.get(row[0], row[0]))
        vals = [g.f1_minority.mean(), g.precision.mean(), g.recall.mean(), g.accuracy.mean() * 100]
        for lab, cell, v, tol in zip(["F1", "P", "R", "Acc"], row[1:], vals, [0.0006, 0.0006, 0.0006, 0.06]):
            ok(f"{cap} {row[0]} {lab}", near(num(cell), v, tol), f"(table {cell} vs data {v:.4f})")

# ---------------------------------------------------------------- Table 4 + marketing text
paired = {}
for ds in ["hillstrom", "criteo"]:
    fm = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); fm["alpha"] = fm.alpha.astype(float)
    mb = pd.read_csv(R / f"missing_baselines_{ds}.csv"); mb["alpha"] = mb.alpha.astype(str)
    base = fm[fm.method == "Baseline"].set_index("seed").auc_roc
    def best(df, name):
        r = []
        for a, g in df[df.method == name].groupby("alpha"):
            g = g.set_index("seed").auc_roc; x = (g - base).values * 100; r.append((x.mean(), a, g, x))
        return max(r, key=lambda t: t[0])
    res = {"CTGAN": best(fm, "CTGAN"), "SMOTE": best(fm, "SMOTE"), "GaussianCopula": best(fm, "GaussianCopula"), "ADASYN": best(mb, "ADASYN"), "Borderline-SMOTE": best(mb, "BorderlineSMOTE")}
    ru = mb[mb.method == "RandomUnderSampler"].set_index("seed").auc_roc; res["Random undersampling"] = ((ru - base).mean() * 100, "1:1", ru, (ru - base).values * 100)
    td = pd.read_csv(R / f"ci_tabddpm_{ds}.csv"); td["alpha"] = td.alpha.astype(float); res["TabDDPM (2k)"] = best(td, "TabDDPM")
    cw = pd.read_csv(R / f"ci_balanced_{ds}.csv"); cwg = cw[cw.method == "Balanced"].set_index("seed").auc_roc; res["`class_weight='balanced'`"] = ((cwg - base).mean() * 100, "-", cwg, (cwg - base).values * 100)
    paired[ds] = (res, base)
for row in table_rows("**Table 4."):
    if row[0].startswith("**GReaT"): continue
    for col, ds in [(2, "hillstrom"), (3, "criteo")]:
        r = paired[ds][0][row[0]]
        m = re.match(r"\s*([+\-−]?[\d.]+)\s*±\s*([\d.]+)", row[col].replace("−", "-"))
        ok(f"T4 {row[0]} {ds} gain", near(float(m.group(1)), r[0], 0.006), f"(table {m.group(1)} vs {r[0]:.3f})")
        ok(f"T4 {row[0]} {ds} CI", near(float(m.group(2)), tci(r[3]), 0.006), f"(table {m.group(2)} vs {tci(r[3]):.3f})")
res_h, base_h = paired["hillstrom"]; res_c, base_c = paired["criteo"]
has("baseline AUC Hillstrom", f"baseline AUC is {base_h.mean():.3f} ± {tci(base_h):.3f}")
has("baseline AUC Criteo", f"baseline AUC is {base_c.mean():.3f} ± {tci(base_c):.3f}")
for ds, res_, nm in [("Hillstrom", res_h, "CTGAN"), ("Hillstrom", res_h, "SMOTE"), ("Criteo", res_c, "CTGAN"), ("Criteo", res_c, "SMOTE")]:
    r = res_[nm]; has(f"{nm} {ds} AUC and gain", (f"CTGAN enhances it to {r[2].mean():.3f} ± {tci(r[2].values):.3f}, a gain of {r[0]:.2f} points" if nm == "CTGAN" else f"SMOTE to {r[2].mean():.3f} ± {tci(r[2].values):.3f}, a gain of {r[0]:.2f} points"))
has("abstract range", f"by {res_h['CTGAN'][0]:.1f} to {res_c['CTGAN'][0]:.1f} points, but with five runs")
for ds, res_ in [("hillstrom", res_h), ("criteo", res_c)]:
    ok(f"all Table-4 CIs include zero except CTGAN on Hillstrom ({ds})", all(abs(res_[k][0]) <= tci(res_[k][3]) for k in res_ if not (ds == "hillstrom" and k == "CTGAN")) and (ds != "hillstrom" or abs(res_["CTGAN"][0]) > tci(res_["CTGAN"][3])))
# ADASYN / SMOTE vs CTGAN
for ds, res_ in [("Hillstrom", res_h), ("Criteo", res_c)]:
    for nm in ["ADASYN"]:
        d = res_[nm][3] - res_["CTGAN"][3]; p = stats.ttest_1samp(d, 0).pvalue
        has(f"{nm} vs CTGAN {ds}", f"p={p:.3f} on {ds}")
pa = [stats.ttest_1samp(res_h["ADASYN"][3] - res_h["CTGAN"][3], 0).pvalue, stats.ttest_1samp(res_c["ADASYN"][3] - res_c["CTGAN"][3], 0).pvalue]
ok("abstract ADASYN p (not in text)", pa[0] > 0.05 and pa[1] > 0.05)

# ---------------------------------------------------------------- 14-test FDR family
def ser(df, name, a, col="auc_roc"): return df[(df.method == name) & (df.alpha.astype(float) == float(a))].set_index("seed")[col]
def pt(x, y):
    i = sorted(set(x.index) & set(y.index)); d = (x.loc[i] - y.loc[i]).values; return d.mean() * 100, d.mean() / d.std(ddof=1), stats.ttest_1samp(d, 0).pvalue
fam = []
for ds in ["hillstrom", "criteo"]:
    fm = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); fm["alpha"] = fm.alpha.astype(float); base = ser(fm, "Baseline", 0.0)
    ac, asm = res_h["CTGAN"][1] if ds == "hillstrom" else res_c["CTGAN"][1], (res_h if ds == "hillstrom" else res_c)["SMOTE"][1]
    fam += [(f"CTGAN vs Baseline {ds}", *pt(ser(fm, "CTGAN", ac), base)), (f"SMOTE vs Baseline {ds}", *pt(ser(fm, "SMOTE", asm), base))]
    mc = pd.read_csv(R / f"ci_multi_classifier_{ds}.csv"); mc = mc[mc.classifier == "GBC"]; mc["alpha"] = mc.alpha.astype(float); b10 = ser(mc, "Baseline", 0.0)
    for nm in ["CTGAN", "SMOTE"]:
        a = max(sorted(set(mc[mc.method == nm].alpha)), key=lambda a: (ser(mc, nm, a) - b10).mean()); fam.append((f"{nm} vs Baseline {ds} 10-seed", *pt(ser(mc, nm, a), b10)))
    t = pd.read_csv(R / f"ci_tabddpm_{ds}.csv"); t["alpha"] = t.alpha.astype(float); at = max(sorted(set(t[t.method == "TabDDPM"].alpha)), key=lambda a: (ser(t, "TabDDPM", a) - base).mean())
    fam.append((f"CTGAN vs TabDDPM 2k {ds}", *pt(ser(fm, "CTGAN", ac), ser(t, "TabDDPM", at))))
    bal = pd.read_csv(R / f"ci_balanced_{ds}.csv"); fam.append((f"CTGAN vs Balanced {ds}", *pt(ser(fm, "CTGAN", ac), bal[bal.method == "Balanced"].set_index("seed").auc_roc)))
gr = pd.read_csv(R / "ci_great_hillstrom.csv")
for n in (50, 2000):
    x = gr[(gr.n == n) & (gr.method == "GReaT")].set_index("seed").auc; b = gr[(gr.n == n) & (gr.method == "Baseline")].set_index("seed").auc; fam.append((f"GReaT vs Baseline n={n}", *pt(x, b)))
fd = pd.DataFrame(fam, columns=["test", "delta", "dz", "p"]); fd["p_fdr"] = multipletests(fd.p, method="fdr_bh")[1]
ok("family has 14 tests", len(fd) == 14)
fd.to_csv(R / "fdr_family_14_verifier.csv", index=False)  # compute_statistics.py writes fdr_family_14.csv
g2000 = fd[fd.test == "GReaT vs Baseline n=2000"].iloc[0]
has("GReaT n=2000 p_fdr", f"p_fdr={g2000.p_fdr:.3f}")
ok("GReaT n=2000 effect is negative", g2000.dz < 0); has("GReaT n=2000 d_z", f"d_z=−{abs(g2000.dz):.2f}")
vsb = fd[(fd.test.str.contains("vs Baseline")) & (~fd.test.str.startswith("GReaT"))]
# removed with the sentence it checked (cut for length): has("d_z range vs baseline", f"Cohen's d_z from {vsb.dz.abs().min():.2
for ds in ["hillstrom", "criteo"]:
    r = fd[fd.test == f"CTGAN vs Balanced {ds}"].iloc[0]
    has(f"CTGAN vs class weight {ds}", f"p={r.p:.3f}")
sigfam = set(fd[fd.p_fdr < 0.10].test)
ok("FDR-significant tests in family", sigfam == {"CTGAN vs Baseline hillstrom", "GReaT vs Baseline n=2000"}, f"(got {sigfam})")
has("CTGAN vs class weight p_fdr Hillstrom", f"p_fdr={fd[fd.test=='CTGAN vs Balanced hillstrom'].p_fdr.iloc[0]:.3f}")

# ---------------------------------------------------------------- TabDDPM budgets (best alpha vs best alpha)
for ds, res_ in [("hillstrom", res_h), ("criteo", res_c)]:
    fm = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); fm["alpha"] = fm.alpha.astype(float); c = ser(fm, "CTGAN", res_["CTGAN"][1])
    for tag, f in [("2k", f"ci_tabddpm_{ds}.csv"), ("10k", f"ci_tabddpm_{ds}_10000.csv")]:
        t = pd.read_csv(R / f); t["alpha"] = t.alpha.astype(float); base = ser(fm, "Baseline", 0.0)
        at = max(sorted(set(t[t.method == "TabDDPM"].alpha)), key=lambda a: (ser(t, "TabDDPM", a) - base).mean()); dlt, dz, p = pt(c, ser(t, "TabDDPM", at))
        has(f"CTGAN vs TabDDPM {tag} {ds}", f"Δ={dlt:+.2f} points with d_z={dz:.2f} and p={p:.3f}") if tag == "2k" else ok(f"CTGAN vs TabDDPM {tag} {ds} (not in text)", True)
    t10 = pd.read_csv(R / f"ci_tabddpm_{ds}_10000.csv"); t10["alpha"] = t10.alpha.astype(float); b = ser(fm, "Baseline", 0.0)
    if ds == "hillstrom": ok("TabDDPM 10k below baseline at all alphas (Hillstrom)", all((ser(t10, "TabDDPM", a) - b).mean() < 0 for a in sorted(set(t10[t10.method == "TabDDPM"].alpha))))

# ---------------------------------------------------------------- control-dataset text numbers
bestall = {}
for name, f in CTRL.items():
    d = pd.read_csv(R / f); d["alpha"] = d.alpha.astype(float); base = d[d.method == "Baseline"].set_index("seed").auc_roc
    bestall[name] = max(((d[(d.method == m) & (d.alpha == a)].set_index("seed").auc_roc - base).mean() * 100) for m in ["GaussianCopula", "CTGAN", "SMOTE"] for a in sorted(d[d.method == m].alpha.unique()))
has("control best gains", f"+{bestall['Telco']:.2f} on Telco, −{abs(bestall['Bank Marketing']):.2f} on Bank Marketing, +{round(bestall['German Credit'],1)} on German Credit and {'−' if bestall['Nomao'] < 0 else '+'}{abs(bestall['Nomao']):.3f} on Nomao")
ok("no control gain above 0.5", max(bestall.values()) < 0.5 + 0.006, f"(max {max(bestall.values()):.3f})")

for g_ in ["GaussianCopula", "CTGAN", "SMOTE"]:
    c_ = cache[("Nomao", g_)]; ok(f"Nomao {g_} within 0.07 AUC and 0.003 F1", abs(c_["dauc"]) <= 0.07 + 0.006 and abs(c_["df1"]) <= 0.003 + 0.0006, f"({c_['dauc']:.3f}, {c_['df1']:.4f})")
ok("FDR-significant effects at most 0.32 AUC and 0.06 F1", abs(cache[("Bank Marketing", "CTGAN")]["dauc"]) <= 0.32 + 0.006 and abs(cache[("Nomao", "CTGAN")]["dauc"]) <= 0.32 and max(abs(cache[("Bank Marketing", "GaussianCopula")]["df1"]), abs(cache[("Bank Marketing", "SMOTE")]["df1"])) <= 0.06 + 0.0006)
ok("Table 3 has 12 rows", len(table_rows("**Table 3.")) == 12)
ok("low-data figure removed from the manuscript", "fig4_lowdata_regime" not in BODY and "Low-data setting" not in BODY)

# ---------------------------------------------------------------- multi-classifier / MLP
mc = pd.read_csv(R / "ci_multi_classifier_criteo.csv"); mc["alpha"] = mc.alpha.astype(float)
bm = mc[(mc.classifier == "MLP") & (mc.method == "Baseline")].set_index("seed").auc_roc
cm = mc[(mc.classifier == "MLP") & (mc.method == "CTGAN") & (mc.alpha == 0.2)].set_index("seed").auc_roc
ok("7 of 10 MLP seeds fail", int((bm < 0.15).sum()) == 7 and len(bm) == 10)
has("MLP CTGAN range", f"from {cm.min():.3f} to {cm.max():.3f}")
ok("all 7 failed seeds recover above 0.89", cm[bm[bm < 0.15].index].min() > 0.89)
for clf, key in [("GBC", "Gradient Boosting at +"), ("RF", "Random Forest at +")]:
    b = mc[(mc.classifier == clf) & (mc.method == "Baseline")].set_index("seed").auc_roc
    g = max(((mc[(mc.classifier == clf) & (mc.method == "CTGAN") & (mc.alpha == a)].set_index("seed").auc_roc - b).mean() * 100) for a in sorted(mc[mc.method == "CTGAN"].alpha.unique()))
    has(f"CTGAN gain {clf}", f"{key}{g:.2f} points")
lr = mc[(mc.classifier == "LR") & (mc.method == "Baseline")].auc_roc.mean(); has("LR baseline", f"near ceiling at {lr:.3f}")

# ---------------------------------------------------------------- GReaT and Mistral
def pg(d, m, n):
    b = d[(d.n == n) & (d.method == "Baseline") & d.auc.notna()].set_index("seed").auc; g = d[(d.n == n) & (d.method == m) & d.auc.notna()].set_index("seed").auc
    c = sorted(set(b.index) & set(g.index)); x = (g.loc[c] - b.loc[c]).values * 100
    return x.mean(), len(c), int((x > 0).sum()), (stats.ttest_1samp(x, 0).pvalue if len(x) > 1 else float("nan"))
g50, g2000 = pg(gr, "GReaT", 50), pg(gr, "GReaT", 2000)
ok("GReaT n=50: +2.25, 4 of 5 win", near(g50[0], 2.25, 0.006) and g50[2] == 4 and g50[1] == 5)
ok("GReaT n=2000: -6.87, 0 of 5, p=0.001", near(g2000[0], -6.87, 0.006) and g2000[2] == 0 and round(g2000[3], 3) == 0.001)
mh = pd.read_csv(R / "modernllm_hillstrom_results_parallel.csv"); mt = pd.read_csv(R / "modernllm_telco_results_parallel.csv"); mg = pd.read_csv(R / "modernllm_german_results_parallel.csv")
h500, h1000, h2000 = pg(mh, "ModernLLM", 500), pg(mh, "ModernLLM", 1000), pg(mh, "ModernLLM", 2000)
has("Mistral Hillstrom n=500", f"gains {h500[0]:.2f} points at n=500 (5 of 5 seeds, unadjusted p={h500[3]:.3f})")
has("Mistral Hillstrom n=1000", f"loses {abs(h1000[0]):.2f} points at n=1,000 (p={h1000[3]:.3f})")
# removed with the clause it checked: has("Mistral Hillstrom n=2000", f"{abs(h2000[0]):.2f} points at n=2,000")
tel = [pg(mt, "ModernLLM", n)[0] for n in (50, 100, 200, 500)]
# removed with the sentence it checked (cut for length): has("Mistral Telco range", f"by {abs(min(tel)):.2f} to {abs(max(tel)):
ok("Mistral Telco n=1000 has 2 valid seeds", pg(mt, "ModernLLM", 1000)[1] == 2)
ger = {n: pg(mg, "ModernLLM", n)[0] for n in (50, 100, 200, 500)}
neg = [abs(ger[n]) for n in (100, 200, 500)]
ok("Mistral German negative at 100, 200, 500 and positive at 50", all(ger[n] < 0 for n in (100, 200, 500)) and ger[50] > 0)
has("Mistral German", f"it hurts at n=100 to 500, by {max(neg):.2f} to {min(neg):.2f} points")
ok("Mistral Hillstrom n=500 has 5 valid seeds and 5 wins", h500[1] == 5 and h500[2] == 5)
ok("Mistral Telco all negative n=50..500", all(v < 0 for v in tel))

# ---------------------------------------------------------------- dose response
for row in table_rows("**Table 6."):
    cnt = int(row[0].replace(",", ""))
    for col, f in [(2, "dose_response_bank_marketing.csv"), (3, "dose_response_nomao.csv")]:
        d = pd.read_csv(R / f); b = d[(d.minority_count == cnt) & (d.method == "Baseline")].set_index("seed").auc_roc
        g = d[(d.minority_count == cnt) & (d.method == "CTGAN")].set_index("seed").auc_roc; x = (g - b).values * 100
        m = re.match(r"\s*([+\-−]?[\d.]+)(?: pts)? \(p([=<])([\d.]+)\)", row[col].replace("−", "-"))
        ok(f"T6 {cnt} {f[14:18]} gain", near(float(m.group(1)), x.mean(), 0.006), f"(table {m.group(1)} vs {x.mean():.3f})")
        p = stats.ttest_1samp(x, 0).pvalue
        ok(f"T6 {cnt} {f[14:18]} p", (p < 0.001 + 1e-9) if m.group(2) == "<" else near(float(m.group(3)), p, 0.0006), f"(table p{m.group(2)}{m.group(3)} vs {p:.4f})")
dbm, dno = pd.read_csv(R / "dose_response_bank_marketing.csv"), pd.read_csv(R / "dose_response_nomao.csv")
def dgain(d, m, c):
    b = d[(d.minority_count == c) & (d.method == "Baseline")].set_index("seed").auc_roc; g = d[(d.minority_count == c) & (d.method == m)].set_index("seed").auc_roc; x = (g - b).values * 100
    return x.mean(), stats.ttest_1samp(x, 0).pvalue
s16 = dgain(dbm, "SMOTE", 16); s256 = dgain(dbm, "SMOTE", 256)
has("Bank SMOTE count 16", f"(+{s16[0]:.2f} points, p={s16[1]:.3f})")
ok("Bank SMOTE count 256 p below 0.01", s256[1] < 0.01)
has("Nomao count 16", f"CTGAN at +{dgain(dno,'CTGAN',16)[0]:.2f} points and SMOTE at +{dgain(dno,'SMOTE',16)[0]:.2f} points")
nm = [dgain(dno, m, c)[0] for m in ["GaussianCopula", "CTGAN"] for c in (512, 1024)]
has("Nomao harm range", f"from −{abs(max(nm)):.2f} to −{abs(min(nm)):.2f} points for GaussianCopula and CTGAN")
ok("Nomao AUC above 0.97 by count 256", dno[(dno.minority_count == 256) & (dno.method == "Baseline")].auc_roc.mean() > 0.97)

# ---------------------------------------------------------------- TSTR, sparse, CTGAN reproducibility
ts = pd.read_csv(R / "tstr_summary.csv"); tb = ts[ts.condition == "real_only"].set_index("dataset").auc_roc
tg = ts[(ts.method == "GaussianCopula") & (ts.condition == "synthetic_only")].set_index("dataset").auc_roc
sf = {k: (tb[k] - tg[k]) / tb[k] * 100 for k in tb.index}
tc_ = ts[(ts.method == "CTGAN") & (ts.condition == "synthetic_only")].set_index("dataset").auc_roc
best_ = {k: max(tg[k], tc_[k]) for k in tb.index}; sfb = {k: (tb[k] - best_[k]) / tb[k] * 100 for k in tb.index}
ok("TSTR closest generator: GaussianCopula except Bank Marketing (CTGAN)", all((tg[k] >= tc_[k]) == (k != "Bank Marketing") for k in tb.index))
has("TSTR shortfalls", f"{sfb['Telco Churn']:.1f}% on Telco, {sfb['Bank Marketing']:.1f}% on Bank Marketing (CTGAN), {sfb['German Credit']:.1f}% on German Credit and {sfb['Nomao Lead']:.1f}% on Nomao")
ok("TSTR: every generator below baseline on every dataset", bool((ts[ts.condition == "synthetic_only"].set_index("dataset").auc_roc < tb.reindex(ts[ts.condition == "synthetic_only"].dataset).values).all()))
sp = pd.read_csv(R / "ci_nomao_sparse.csv"); sb = sp[sp.method == "Baseline"].set_index("seed").auc; dn = sp[sp.method == "DenseBaseline"].auc
cg = max(((sp[(sp.method == "CTGAN") & (sp.alpha == a)].set_index("seed").auc - sb).mean() * 100) for a in sp[sp.method == "CTGAN"].alpha.unique())
has("sparse baseline", f"has an AUC of {sb.mean():.3f} ± {tci(sb):.3f}"); has("dense reference", f"of {dn.mean():.4f} ± {tci(dn):.4f}")
has("sparse CTGAN gain", f"adds only {cg:.2f} points"); has("dense gap", f"{(dn.mean()-sb.mean())*100:.2f}-point gap")
rf = pd.read_csv(R / "ctgan_refit_variance.csv"); ct = rf[rf.generator == "CTGAN"].auc_roc; gc_ = rf[rf.generator == "GaussianCopula"].auc_roc
ok("refit CTGAN range (not in text)", ct.max() - ct.min() > 0)
ok("GaussianCopula refits identical", gc_.nunique() == 1)
U = R / "unseeded_runs"
def loadc(p):
    d = pd.read_csv(p); d["alpha"] = d.alpha.astype(float); return d
def cells(x, y):
    m = x[x.method == "CTGAN"].merge(y[y.method == "CTGAN"], on=["seed", "alpha"], suffixes=("_x", "_y")); return (m.auc_roc_x - m.auc_roc_y).abs() * 100
def best_gain(d):
    b = d[d.method == "Baseline"].set_index("seed").auc_roc; g = d[d.method == "CTGAN"]; a = g.groupby("alpha").auc_roc.mean().idxmax()
    x = (g[g.alpha == a].set_index("seed").auc_roc - b) * 100; return x.mean(), stats.ttest_1samp(x, 0).pvalue
cr = {}
for ds in ["hillstrom", "criteo"]:
    apr = loadc(U / f"ci_{ds}_april_unseeded.csv"); sep = loadc(U / f"ci_{ds}_fullmetrics_sept_unseeded.csv"); new_ = loadc(R / f"ci_{ds}_fullmetrics.csv")
    cr[ds] = dict(apr=apr, sep=sep, new=new_, a_s=cells(apr, sep), s_n=cells(sep, new_))
    same = apr[apr.method.isin(['GaussianCopula', 'SMOTE', 'Baseline'])].merge(new_[new_.method.isin(['GaussianCopula', 'SMOTE', 'Baseline'])], on=["seed", "method", "alpha"], suffixes=("_o", "_n"))
    ok(f"GC/SMOTE/Baseline identical across the April run and the seeded rerun ({ds})", (same.auc_roc_o - same.auc_roc_n).abs().max() < 1e-9)
h, c = cr["hillstrom"], cr["criteo"]
# removed with the sentence it checked (cut for length): has("cross-run April vs September", f"by {h['a_s'].mean():.1f} points 
has("contributions cross-run", f"identical reruns changed AUC by up to {h['a_s'].max():.1f} points")
# removed with the sentence it checked (cut for length): has("cross-run September vs seeded", f"by {h['s_n'].mean():.1f} points
(ga, pa_), (gs_, ps_), (gn, pn) = best_gain(h["apr"]), best_gain(h["sep"]), best_gain(h["new"])
has("Hillstrom CTGAN gain across runs", f"+{ga:.2f}, +{gs_:.2f} and +{gn:.2f} points in the April, September and seeded runs, with p={pa_:.3f}, p={ps_:.3f} and p={pn:.4f}")
# removed with the sentence it checked (cut for length): has("Hillstrom CTGAN p x5", f"gives p={pn*5:.3f}")
sc = pd.read_csv(R / "ctgan_seed_check.csv"); s1 = sc[sc.check == "same seed, twice"]; s5 = sc[sc.check == "different seeds"]
ok("seed check: same seed gives identical rows and AUC", bool(s1.identical_to_earlier_fit_with_same_seed.iloc[1]) and abs(s1.auc_roc.iloc[0] - s1.auc_roc.iloc[1]) < 1e-12)
has("seed check same-seed AUC", f"the same AUC ({s1.auc_roc.iloc[0]:.3f})")
has("seed check range", f"from {s5.auc_roc.min():.3f} to {s5.auc_roc.max():.3f}")

# ---------------------------------------------------------------- other derived numbers
has("11.3% positive from 7.88:1", f"about {100/(1+7.88):.1f}% positive")
def power(d, n=5):
    df_ = n - 1; nc = d * np.sqrt(n); t = stats.t.ppf(0.975, df_); return 1 - stats.nct.cdf(t, df_, nc) + stats.nct.cdf(-t, df_, nc)
ok("80% power at d_z 1.7 (n=5)", 0.78 < power(1.7) < 0.83, f"(power {power(1.7):.3f})")
# ---------------------------------------------------------------- best mixing ratio across dataset and generator pairs
pairs = []
for name, f in CTRL.items():
    d = pd.read_csv(R / f); d["alpha"] = d.alpha.astype(float); b = d[d.method == "Baseline"].set_index("seed").auc_roc
    for m in ["GaussianCopula", "CTGAN", "SMOTE"]:
        pairs.append((name, m, max(sorted(d[d.method == m].alpha.unique()), key=lambda a: (d[(d.method == m) & (d.alpha == a)].set_index("seed").auc_roc - b).mean())))
for ds in ["hillstrom", "criteo"]:
    d = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); d["alpha"] = d.alpha.astype(float); b = d[d.method == "Baseline"].set_index("seed").auc_roc
    for m in ["GaussianCopula", "CTGAN", "SMOTE"]:
        pairs.append((ds, m, max(sorted(d[d.method == m].alpha.unique()), key=lambda a: (d[(d.method == m) & (d.alpha == a)].set_index("seed").auc_roc - b).mean())))
    mb2 = pd.read_csv(R / f"missing_baselines_{ds}.csv"); mb2["alpha"] = mb2.alpha.astype(str)
    for m in ["ADASYN", "BorderlineSMOTE"]:
        g_ = mb2[mb2.method == m]; pairs.append((ds, m, float(max(sorted(g_.alpha.unique()), key=lambda a: (g_[g_.alpha == a].set_index("seed").auc_roc - b).mean()))))
inside = sum(0.1 <= a <= 0.3 for _, _, a in pairs)
outside = [(n, m, a) for n, m, a in pairs if not 0.1 <= a <= 0.3]
ok("22 pairs = 3 generators x 4 controls + 5 generators x 2 marketing datasets", len(pairs) == 3 * 4 + 5 * 2 == 22)
n01 = sum(abs(a - 0.1) < 1e-9 for _, _, a in pairs)
has("alpha 0.1 best count", f"α=0.1 was best in {n01} of the {len(pairs)} combinations")
has("alpha counts, Section 4.3", f"Across all {len(pairs)} combinations of dataset and generator (excluding TabDDPM), the best α was between 0.1 and 0.3 in {inside}")
ok("alpha exceptions are CTGAN on Telco, Hillstrom and Criteo (0.5) and ADASYN on Hillstrom (1.0)", sorted(outside) == [("Telco", "CTGAN", 0.5), ("criteo", "CTGAN", 0.5), ("hillstrom", "ADASYN", 1.0), ("hillstrom", "CTGAN", 0.5)], f"(got {outside})")

# ---------------------------------------------------------------- claims added later
# Bank Marketing dose-response, per generator
bank = pd.read_csv(R / "dose_response_bank_marketing.csv")
def bgain(g, m):
    b = bank[(bank.minority_count == m) & (bank.method == "Baseline")].set_index("seed").auc_roc; a = bank[(bank.minority_count == m) & (bank.method == g)].set_index("seed").auc_roc
    x = (a - b).values * 100; return x.mean(), stats.ttest_1samp(x, 0).pvalue
sig64 = [g for g in ["GaussianCopula", "CTGAN", "SMOTE"] if bgain(g, 64)[1] < 0.05]
ok("only CTGAN and GaussianCopula are significant at m=64", sorted(sig64) == ["CTGAN", "GaussianCopula"], f"(got {sig64})")
ok("all three significant and negative from m=256", all(bgain(g, m)[1] < 0.05 and bgain(g, m)[0] < 0 for g in ["GaussianCopula", "CTGAN", "SMOTE"] for m in (256, 512, 1024)))
ok("all three negative at m=64", all(bgain(g, 64)[0] < 0 for g in ["GaussianCopula", "CTGAN", "SMOTE"]))
ok("only SMOTE significant and positive at m=16", [g for g in ["GaussianCopula", "CTGAN", "SMOTE"] if bgain(g, 16)[1] < 0.05 and bgain(g, 16)[0] > 0] == ["SMOTE"])
ok("positive at m=16 only for SMOTE and CTGAN", [g for g in ["GaussianCopula", "CTGAN", "SMOTE"] if bgain(g, 16)[0] > 0] == ["CTGAN", "SMOTE"])
has("CTGAN and GaussianCopula at m=64", f"only for CTGAN (p={bgain('CTGAN', 64)[1]:.3f}) and GaussianCopula (p={bgain('GaussianCopula', 64)[1]:.3f})")
has("CTGAN at m=256", f"CTGAN loses {abs(bgain('CTGAN', 256)[0]):.2f} points (p={bgain('CTGAN', 256)[1]:.3f})")
has("SMOTE at m=256", f"SMOTE loses {abs(bgain('SMOTE', 256)[0]):.2f} (p={bgain('SMOTE', 256)[1]:.3f})")
ok("CTGAN and SMOTE harm shrinks after m=256", all(abs(bgain(g, 1024)[0]) < abs(bgain(g, 256)[0]) for g in ["CTGAN", "SMOTE"]))
# Table 3 bold marks agree with the FDR results, and the CTGAN Hillstrom p-values in Section 3.3
import re as _re
_t3 = pd.read_csv(R / "stats_table3.csv")
_rows = [l for l in BODY.split("\n") if l.startswith("| ") and _re.search(r"\(\d\.\d\)", l.split("|")[2] if l.count("|") > 3 else "")]
_bold = {}
for l in _rows:
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) == 9 and c[0] in set(_t3.dataset):
        _bold[(c[0], _re.sub(r" \(.*", "", c[1]))] = (c[2].startswith("**") and c[3].startswith("**"), c[4].startswith("**") and c[5].startswith("**"),
                                                    c[2].startswith("**") != c[3].startswith("**") or c[4].startswith("**") != c[5].startswith("**"))
ok("Table 3 has 12 rows", len(_bold) == 12, f"(found {len(_bold)})")
ok("Table 3 bold marks equal the FDR flags", all(_bold.get((r.dataset, r.generator)) is not None and _bold[(r.dataset, r.generator)][:2] == (bool(r.auc_significant_fdr), bool(r.f1_significant_fdr)) and not _bold[(r.dataset, r.generator)][2] for r in _t3.itertuples()))
_f14 = pd.read_csv(R / "fdr_family_14.csv").set_index("test")
_c = _f14.loc["CTGAN vs Baseline hillstrom"]
has("CTGAN Hillstrom p and p_fdr in 3.3", f"p={_c.p:.4f}, p_fdr={_c.p_fdr:.3f}")
# alpha=0.2 Bank Marketing sweep (Section 3.6)
a2 = pd.read_csv(R / "stats_dose_response_alpha02.csv").set_index(["minority_count", "method"])
g2 = lambda m, g: (a2.loc[(m, g)].gain_points, a2.loc[(m, g)].p)
ok("alpha 0.2: GaussianCopula never significant", all(g2(m, "GaussianCopula")[1] >= 0.05 for m in (16, 64, 256, 512, 1024)))
ok("alpha 0.2: CTGAN significant only at 1,024", [m for m in (16, 64, 256, 512, 1024) if g2(m, "CTGAN")[1] < 0.05] == [1024])
ok("alpha 0.2: SMOTE significant and negative from 256", all(g2(m, "SMOTE")[1] < 0.05 and g2(m, "SMOTE")[0] < 0 for m in (256, 512, 1024)) and g2(64, "SMOTE")[1] >= 0.05)
ok("alpha 0.2: ADASYN does not hurt at 256 and above", all(g2(m, "ADASYN")[1] >= 0.05 for m in (256, 512, 1024)))
has("alpha 0.2 CTGAN at 1,024", f"CTGAN is significant only at 1,024 (p={g2(1024, 'CTGAN')[1]:.3f})")
has("alpha 0.2 SMOTE at 256", f"SMOTE is significant from 256 (−{abs(g2(256, 'SMOTE')[0]):.2f} points, p={g2(256, 'SMOTE')[1]:.3f})")
has("alpha 0.2 ADASYN at 16", f"ADASYN gains +{g2(16, 'ADASYN')[0]:.2f} points at 16 (p={g2(16, 'ADASYN')[1]:.3f}) and +{g2(64, 'ADASYN')[0]:.2f} at 64 (p={g2(64, 'ADASYN')[1]:.3f})")
# SMOTE vs CTGAN, Hillstrom 10-seed GBC, undersampling, test-set positives, minority counts
for ds, key in [("hillstrom", "Hillstrom"), ("criteo", "Criteo")]:
    fm_ = pd.read_csv(R / f"ci_{ds}_fullmetrics.csv"); fm_["alpha"] = fm_.alpha.astype(float); b_ = fm_[fm_.method == "Baseline"].set_index("seed").auc_roc
    def best_(nm):
        cands = [((fm_[(fm_.method == nm) & (fm_.alpha == a)].set_index("seed").auc_roc - b_).values * 100) for a in sorted(fm_[fm_.method == nm].alpha.unique())]
        return max(cands, key=lambda x: x.mean())
    p_ = stats.ttest_1samp(best_("SMOTE") - best_("CTGAN"), 0).pvalue
    has(f"SMOTE vs CTGAN {ds}", f"p={p_:.3f}")
mch = pd.read_csv(R / "ci_multi_classifier_hillstrom.csv"); mch = mch[mch.classifier == "GBC"]; mch["alpha"] = mch.alpha.astype(float); bh10 = mch[mch.method == "Baseline"].set_index("seed").auc_roc
for nm, tag in [("CTGAN", "CTGAN"), ("SMOTE", "SMOTE")]:
    cands_ = [((mch[(mch.method == nm) & (mch.alpha == a)].set_index("seed").auc_roc - bh10).values * 100) for a in sorted(mch[mch.method == nm].alpha.unique())]
    best_ = max(cands_, key=lambda x: x.mean())
    has(f"Hillstrom 10-seed GBC {tag}", f"+{best_.mean():.2f} points for {tag} (p={stats.ttest_1samp(best_, 0).pvalue:.3f})")
mbh = pd.read_csv(R / "missing_baselines_hillstrom.csv"); mbc = pd.read_csv(R / "missing_baselines_criteo.csv")
fmh = pd.read_csv(R / "ci_hillstrom_fullmetrics.csv"); fmc = pd.read_csv(R / "ci_criteo_fullmetrics.csv")
def f1_rank(mb, fm, bal):
    vals = {"RandomUnderSampler": mb[mb.method == "RandomUnderSampler"].f1_minority.mean()}
    for nm in ["ADASYN", "BorderlineSMOTE"]: g = mb[mb.method == nm]; vals[nm] = g[g.alpha.astype(str) == g.groupby("alpha").auc_roc.mean().idxmax()].f1_minority.mean()
    for nm in ["CTGAN", "SMOTE", "GaussianCopula", "Baseline"]:
        g = fm[fm.method == nm]; g = g[g.alpha.astype(float) == (g.groupby("alpha").auc_roc.mean().idxmax())]; vals[nm] = g.f1_minority.mean()
    vals["class_weight"] = bal.f1_minority.mean(); return vals
fv_h = f1_rank(mbh, fmh, pd.read_csv(R / "fullmetrics_balanced_hillstrom.csv")); fv_c = f1_rank(mbc, fmc, pd.read_csv(R / "fullmetrics_balanced_criteo.csv"))
ok("undersampling F1 best on Hillstrom", max(fv_h, key=fv_h.get) == "RandomUnderSampler")
ok("undersampling F1 lowest on Criteo", min(fv_c, key=fv_c.get) == "RandomUnderSampler")
has("undersampling F1 claim text", "its F1 is the best on Hillstrom and the lowest on Criteo")
rec_h = {nm: mbh[mbh.method == nm].recall.mean() for nm in ["RandomUnderSampler"]}
ok("undersampling recall highest on both marketing datasets", mbh.groupby("method").recall.mean().idxmax() == "RandomUnderSampler" and mbc.groupby("method").recall.mean().idxmax() == "RandomUnderSampler")
# recall range 0% to 29% for methods that add synthetic rows
rr = []
for ds, mb_, fm_ in [("h", mbh, fmh), ("c", mbc, fmc)]:
    for nm in ["ADASYN", "BorderlineSMOTE"]: g = mb_[mb_.method == nm]; rr.append(g[g.alpha.astype(str) == g.groupby("alpha").auc_roc.mean().idxmax()].recall.mean())
    for nm in ["CTGAN", "SMOTE", "GaussianCopula"]: g = fm_[fm_.method == nm]; rr.append(g[g.alpha.astype(float) == g.groupby("alpha").auc_roc.mean().idxmax()].recall.mean())
ok("recall range 0% to 29% for methods that add synthetic rows", round(min(rr) * 100) == 0 and round(max(rr) * 100) == 29, f"({min(rr):.3f}, {max(rr):.3f})")
has("recall range text", f"against {round(min(rr) * 100)}% to {round(max(rr) * 100)}% for every method that adds synthetic minority rows")
# positives per test split and expected minority counts
from experiments.run_hillstrom import load_hillstrom as _lh
from experiments.run_criteo import load_criteo_uplift as _lc
from sklearn.model_selection import train_test_split as _tts
tp = {}
for nm, fn in [("Hillstrom", _lh), ("Criteo", _lc)]:
    r_ = fn(); df_, t_ = r_[0], r_[1]; tp[nm] = []
    for sd in [42, 123, 7, 2024, 999]:
        d_ = df_.sample(10000, random_state=sd).reset_index(drop=True); _, te_ = _tts(d_, test_size=0.2, random_state=sd, stratify=d_[t_]); tp[nm].append(int(te_[t_].sum()))
has("test positives Hillstrom and Criteo", f"with {min(tp['Hillstrom'])} to {max(tp['Hillstrom'])} positives on Hillstrom and {min(tp['Criteo'])} to {max(tp['Criteo'])} on Criteo")
ok("about 72 and 24 minority examples per 8,000-row split", round(info["Hillstrom"][2] / 100 * 8000) == 72 and round(crit_rate / 100 * 8000) == 24)
has("minority example counts text", f"only about {round(crit_rate / 100 * 8000)} minority examples are expected on Criteo")
# GaussianCopula Criteo and Borderline Hillstrom gains used in Section 4.1
has("GaussianCopula Criteo gain in 4.1", f"still gains {res_c['GaussianCopula'][0]:.2f} points on Criteo")
# removed with the sentence it checked (cut for length): has("Borderline Hillstrom gain in 4.1", f"gains just {res_h['Borderlin

# ---------------------------------------------------------------- stale phrases that must not reappear
STALE = ["119 features", "against 17", "n/c", "min/seed", "GPU-hours", "CPU-hours", "three orders of magnitude", "companion", "TODO",
         "0.527", "10.45", "7.46-point", "30% to 60%", "+1.20 points over 3", "random_state", "p values", "empirical frequency",
         "benchmark datasets", "U-curve", "headline", "which separates minority count from dataset", "confounding critique", "actively hurts", "genuine boundary", "within a 0.1 step", "[AUTHOR", "not independently re-verifiable", "| TabDDPM | 0.89%", "+5.7 to +12.9", "5.7 to 12.9"]
# ---------------------------------------------------------------- claims added after the second round of team suggestions
fb = pd.read_csv(R / "stats_free_baselines.csv").set_index(["dataset", "method"])
def mc_stats(ds, clf):
    d = pd.read_csv(R / f"ci_multi_classifier_{ds}.csv"); d["alpha"] = pd.to_numeric(d.alpha, errors="coerce"); x = d[d.classifier == clf]
    b = x[x.method == "Baseline"].set_index("seed").auc_roc; out = {"base": b.mean()}
    for m in ["CTGAN", "SMOTE"]:
        g = x[x.method == m]; a = max(sorted(g.alpha.unique()), key=lambda a: (g[g.alpha == a].set_index("seed").auc_roc - b).mean())
        dd = (g[g.alpha == a].set_index("seed").auc_roc - b) * 100; out[m] = (dd.mean(), stats.ttest_1samp(dd, 0).pvalue)
    return out
def pf3(p_): return "p<0.001" if p_ < 0.0005 else f"p={p_:.3f}"
rf_, mlp_, lr_ = mc_stats("hillstrom", "RF"), mc_stats("hillstrom", "MLP"), mc_stats("hillstrom", "LR")
# removed with the sentence it checked (cut for length): has("Hillstrom RF and MLP baselines at chance", f"(Random Forest {rf_[
# removed with the sentence it checked (cut for length): has("Hillstrom RF/MLP gains", f"CTGAN adds +{rf_['CTGAN'][0]:.2f} poin
# removed with the sentence it checked (cut for length): has("Hillstrom LR does not gain", f"Logistic Regression does not gain 
pp = lambda p_: f"{p_:.4f}"
has("Hillstrom other classifiers", f"SMOTE adds +{mlp_['SMOTE'][0]:.2f} points with the MLP but loses {abs(lr_['SMOTE'][0]):.2f} with Logistic Regression")
gb_c, lr_c = mc_stats("criteo", "GBC"), mc_stats("criteo", "LR")
# removed with the sentence it checked (cut for length): has("Criteo LR baseline vs GBC with CTGAN", f"Logistic Regression with
dr_ = pd.read_csv(R / "stats_dose_response.csv").set_index(["dataset", "minority_count", "method"])
# removed with the sentence it checked (cut for length): has("Table 7 evidence at 0.16%", f"SMOTE gained +{dr_.loc[('Bank Marke
for m in ["CTGAN", "SMOTE"]:
    g_ = dr_.xs(("Bank Marketing", m), level=("dataset", "method")).gain_points
    ok(f"Bank Marketing harm largest at 256 for {m}", g_.loc[256] == g_.loc[[64, 256, 512, 1024]].min() and g_.loc[1024] > g_.loc[256])
tsd = pd.read_csv(R / "tstr_summary.csv"); tbb = tsd[tsd.condition == "real_only"].set_index("dataset").auc_roc
tcc = tsd[(tsd.method == "CTGAN") & (tsd.condition == "synthetic_only")].set_index("dataset").auc_roc; tcs = {k: (tbb[k] - tcc[k]) / tbb[k] * 100 for k in tbb.index}
_, hold = _tts(load_hillstrom()[0], test_size=10000, random_state=42, stratify=load_hillstrom()[0]["target"])
pool_ = load_hillstrom()[0].drop(hold.index); npos = [max(1, int(round(n * pool_["target"].sum() / len(pool_)))) for n in (50, 100, 200, 500, 1000, 2000)]
# removed with the sentence it checked (cut for length): has("GReaT positives per training size", "contain " + ", ".join(map(st
crit = pd.read_csv("data/criteo_uplift.csv", nrows=2) if Path("data/criteo_uplift.csv").exists() else None
if crit is not None: ok("Criteo: 12 anonymized features after dropping treatment, exposure and visit", crit.shape[1] - 1 == 12 and not {"treatment", "exposure", "visit"} & set(crit.columns), f"({list(crit.columns)})")
ok("Abstract is at most 200 words", len(MD.split("## Abstract")[1].split("**Keywords:**")[0].split()) <= 200, f"({len(MD.split('## Abstract')[1].split('**Keywords:**')[0].split())} words)")
ok("no em dashes in the manuscript", "—" not in MD)
ok("units are written as points, not pts", re.search(r"\bpts\b", BODY) is None)
ok("MDPI: 3 to 10 keywords", 3 <= len(MD.split("**Keywords:**")[1].split("\n")[0].split(";")) <= 10)

# Values from the unseeded CTGAN runs that must not remain anywhere in the text (the April and September values that
# Section 4.5 quotes on purpose are not listed here).
STALE_VALUES = ["± 5.61", "0.602 ± 0.064", "0.972 ± 0.047", "12.60 ± 22.00", "6.65%", "26.78% ± 1.19%", "p=0.842", "p=0.745", "p=0.920", "p=0.572",
                "7.28 points", "p_fdr=0.017", "p_fdr=0.118", "p_fdr=0.008", "+4.13 points", "+7.49 points", "27.2%", "d_z=3.04", "0.865 and 0.986",
                "+12.04", "+9.55", "+3.30 points", "+11.21", "3.48 points", "+0.01 on Nomao", "0.717", "0.84 and 0.75", "5.5 to 12.6", "+5.5 to +12.6",
                "p=0.002, p_fdr", "no single gain was established", "none reaches FDR significance", "Three comparisons in Table 3",
                "between 0.1 and 0.3, not", "tuned its", "was not seeded, and", "Display Advertising", "entity-embedded", "Figure 13"]
for phrase in STALE_VALUES:
    ok(f"old value absent: {phrase!r}", phrase not in BODY)
for phrase in STALE:
    ok(f"stale phrase absent: {phrase!r}", phrase not in BODY)
print(f"\n{PASSES} checks passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
