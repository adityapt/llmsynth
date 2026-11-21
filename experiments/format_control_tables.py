"""Format Table 3 and Tables 5c to 5f of the Electronics manuscript from the result files.

Usage (from the repo root): python3 experiments/format_control_tables.py [results_dir]
Prints markdown for the Table 3 rows and for Tables 5c to 5f, so the manuscript tables are produced from the
data and not typed by hand. FDR significance for the bold marks in Table 3 comes from compute_statistics.py
(stats_table3.csv) when it exists in the results directory, and is recomputed otherwise.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
MINUS = "−"
CTRL = [("Telco", "Telco Churn", "fullmetrics_telco.csv", "5c"), ("Bank Marketing", "Bank Marketing", "fullmetrics_bank_marketing.csv", "5d"),
        ("German Credit", "German Credit", "fullmetrics_german_credit.csv", "5e"), ("Nomao", "Nomao", "fullmetrics_nomao.csv", "5f")]
GENS = ["GaussianCopula", "CTGAN", "SMOTE"]


def sgn(x, nd, suffix=""):
    t = f"{abs(x):.{nd}f}"
    if float(t) == 0:
        return f"{MINUS if x < 0 else '+'}{t}{suffix}"
    return f"{'+' if x > 0 else MINUS}{t}{suffix}"


def pfmt(p):
    return "<0.001" if p < 0.0005 else f"{p:.3f}"


def load(f):
    d = pd.read_csv(R / f)
    d["alpha"] = d.alpha.astype(float)
    return d


def best_alpha(d, m):
    return d[d.method == m].groupby("alpha").auc_roc.mean().idxmax()


rows = []
for short, _, f, _ in CTRL:
    d = load(f)
    base = d[d.method == "Baseline"].set_index("seed")
    for m in GENS:
        a = best_alpha(d, m)
        g = d[(d.method == m) & (d.alpha == a)].set_index("seed")
        c = sorted(set(g.index) & set(base.index))
        dauc = (g.loc[c].auc_roc - base.loc[c].auc_roc) * 100
        df1 = g.loc[c].f1_minority - base.loc[c].f1_minority
        rows.append(dict(ds=short, m=m, a=a, dauc=dauc.mean(), pauc=stats.ttest_1samp(dauc, 0).pvalue, df1=df1.mean(), pf1=stats.ttest_1samp(df1, 0).pvalue,
                         dp=(g.loc[c].precision - base.loc[c].precision).mean(), dr=(g.loc[c].recall - base.loc[c].recall).mean(),
                         dacc=(g.loc[c].accuracy - base.loc[c].accuracy).mean() * 100))
pv = [x for r in rows for x in (r["pauc"], r["pf1"])]
rej = multipletests(pv, method="fdr_bh", alpha=0.10)[0]
print("TABLE 3 ROWS")
for i, r in enumerate(rows):
    sa, sf = rej[2 * i], rej[2 * i + 1]
    b = lambda txt, on: f"**{txt}**" if on else txt
    print(f"| {r['ds']} | {r['m']} ({r['a']:.1f}) | {b(sgn(r['dauc'], 2), sa)} | {b(pfmt(r['pauc']), sa)} | {b(sgn(r['df1'], 4), sf)} | {b(pfmt(r['pf1']), sf)} | "
          f"{sgn(r['dp'], 3)} | {sgn(r['dr'], 3)} | {sgn(r['dacc'], 2)} |")
print("\nFDR-significant:", [(r['ds'], r['m'], k) for i, r in enumerate(rows) for k, v in (("auc", rej[2 * i]), ("f1", rej[2 * i + 1])) if v])
for short, label, f, tag in CTRL:
    d = load(f)
    out = []
    for m in ["Baseline", "CTGAN", "SMOTE", "GaussianCopula"]:
        if m == "Baseline":
            g = d[d.method == "Baseline"]
        else:
            g = d[(d.method == m) & (d.alpha == best_alpha(d, m))]
        out.append((m, g.f1_minority.mean(), g.precision.mean(), g.recall.mean(), g.accuracy.mean()))
    mx = [max(round(o[k], 3) for o in out) for k in (1, 2, 3)]
    print(f"\nTABLE {tag}. {label}\n\n| Method | F1 | Precision | Recall | Accuracy |\n|---|---|---|---|---|")
    for m, f1, p, r_, acc in out:
        cells = [f"{v:.3f}" for v in (f1, p, r_)]
        cells = [f"**{c}**" if round(v, 3) == mx[k] else c for k, (c, v) in enumerate(zip(cells, (f1, p, r_)))]
        print(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {acc * 100:.1f}% |")
