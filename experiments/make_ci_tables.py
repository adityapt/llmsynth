"""Build the 95% confidence-interval versions of Tables 3, 5a-5f and 6.

Reads only files in results/ and writes CSVs to results/tables/.
Run from anywhere:  python3 experiments/make_ci_tables.py

Conventions
  * "x +/- h" is the mean and the 95% half-width t(0.975, n-1) * SD / sqrt(n) over seeds
    (n = 5 seeds, so t = 2.776), computed on paired differences where a gain is shown.
  * Rounding is half-up (Decimal), matching the manuscript tables.
  * Table 3 asterisks mark Benjamini-Hochberg FDR significance at q = 0.10, read from
    results/stats_table3.csv (written by compute_statistics.py). Precision, recall and
    accuracy gains are descriptive and are not tested.
  * Table 5 asterisks mark the highest mean in a column (on the rounded value); this is
    not a significance test. "+/- 0.000" means all seeds gave the same value.
  * Table 6 p-values are paired t-tests over seeds, unadjusted.
"""
import csv, os, warnings
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results") + os.sep
OUT = os.path.join(R, "tables") + os.sep
os.makedirs(OUT, exist_ok=True)
def q(x, dec):  # half-up rounding, as the paper does
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    return f"{d:f}"
def sgn(x, dec):
    s = q(x, dec); return s if s.startswith("-") else "+" + s
def hw(x):
    x = np.asarray(x, float); return stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
def pv(x): return stats.ttest_1samp(x, 0).pvalue
def best_alpha(df, m):
    b = df[df.method=="Baseline"].set_index("seed").auc_roc; best=ba=None
    for a, g in df[df.method==m].groupby("alpha"):
        d = (g.set_index("seed").auc_roc - b.reindex(g.set_index("seed").index)).mean()
        if best is None or d > best: best, ba = d, a
    return ba
def write(name, header, rows):
    with open(OUT + name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)")
# ---------------- Table 3
_st = pd.read_csv(R + "stats_table3.csv")
_key = {"Telco":"telco","Bank Marketing":"bank_marketing","German Credit":"german_credit","Nomao":"nomao"}
sig_auc = {(_key[r.dataset], r.generator) for r in _st.itertuples() if r.auc_significant_fdr}
sig_f1  = {(_key[r.dataset], r.generator) for r in _st.itertuples() if r.f1_significant_fdr}
names = {"telco":"Telco","bank_marketing":"Bank Marketing","german_credit":"German Credit","nomao":"Nomao"}
rows3 = []
for ds in names:
    df = pd.read_csv(R+f"fullmetrics_{ds}.csv")
    for m in ["GaussianCopula","CTGAN","SMOTE"]:
        a = best_alpha(df, m); s = df[(df.method==m)&(df.alpha==a)].set_index("seed"); b = df[df.method=="Baseline"].set_index("seed").reindex(s.index)
        da, f1, pr, rc, ac = (s.auc_roc-b.auc_roc)*100, s.f1_minority-b.f1_minority, s.precision-b.precision, s.recall-b.recall, (s.accuracy-b.accuracy)*100
        sa = "*" if (ds, m) in sig_auc else ""; sf = "*" if (ds, m) in sig_f1 else ""
        rows3.append([names[ds], f"{m} ({a})", f"{sgn(da.mean(),2)}{sa} ± {q(hw(da),2)}", f"{q(pv(da),3)}{sa}", f"{sgn(f1.mean(),4)}{sf} ± {q(hw(f1),4)}", f"{q(pv(f1),3)}{sf}",
                      f"{sgn(pr.mean(),3)} ± {q(hw(pr),3)}", f"{sgn(rc.mean(),3)} ± {q(hw(rc),3)}", f"{sgn(ac.mean(),2)} ± {q(hw(ac),2)}"])
write("table3_with_ci.csv", ["Dataset","Method (α*)","AUC gain (points)","p(AUC)","F1 gain","p(F1)","Precision gain","Recall gain","Accuracy gain (points)"], rows3)
# ---------------- Tables 5a-5f
def t5_rows(sources):  # list of (label, DataFrame indexed by seed)
    stats_ = []
    for label, s in sources:
        stats_.append((label, {c: (s[c].mean(), hw(s[c])) for c in ["f1_minority","precision","recall"]} | {"acc": (s.accuracy.mean()*100, hw(s.accuracy*100))}))
    best = {c: max(round(float(Decimal(repr(float(v[c][0]))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)), 3) for _, v in stats_) for c in ["f1_minority","precision","recall"]}
    out = []
    for label, v in stats_:
        cells = []
        for c in ["f1_minority","precision","recall"]:
            star = "*" if float(q(v[c][0], 3)) == best[c] else ""
            cells.append(f"{q(v[c][0],3)}{star} ± {q(v[c][1],3)}")
        out.append([label] + cells + [f"{q(v['acc'][0],1)}% ± {q(v['acc'][1],1)}"])
    return out
hdr5 = ["Method","F1","Precision","Recall","Accuracy"]
for ds, tag, title in [("hillstrom","5a","hillstrom"),("criteo","5b","criteo")]:
    ci = pd.read_csv(R+f"ci_{ds}_fullmetrics.csv"); mb = pd.read_csv(R+f"missing_baselines_{ds}.csv"); cw = pd.read_csv(R+f"fullmetrics_balanced_{ds}.csv")
    def at(df, key): return df[(df.method==key)&(df.alpha==best_alpha(df, key))].set_index("seed")
    src = [("Baseline", ci[ci.method=="Baseline"].set_index("seed")), ("CTGAN", at(ci,"CTGAN")), ("ADASYN", at(mb,"ADASYN")), ("SMOTE", at(ci,"SMOTE")),
           ("Borderline-SMOTE", at(mb,"BorderlineSMOTE")), ("GaussianCopula", at(ci,"GaussianCopula")), ("Random undersampling", mb[mb.method=="RandomUnderSampler"].set_index("seed")),
           ("class_weight='balanced'", cw.set_index("seed"))]
    write(f"table{tag}_{title}_with_ci.csv", hdr5, t5_rows(src))
for tag, ds, title in [("5c","telco","telco"),("5d","bank_marketing","bank_marketing"),("5e","german_credit","german_credit"),("5f","nomao","nomao")]:
    df = pd.read_csv(R+f"fullmetrics_{ds}.csv")
    def at(key): return df[(df.method==key)&(df.alpha==best_alpha(df, key))].set_index("seed")
    src = [("Baseline", df[df.method=="Baseline"].set_index("seed")), ("CTGAN", at("CTGAN")), ("SMOTE", at("SMOTE")), ("GaussianCopula", at("GaussianCopula"))]
    write(f"table{tag}_{title}_with_ci.csv", hdr5, t5_rows(src))
# ---------------- Table 6
bank = pd.read_csv(R+"dose_response_bank_marketing.csv"); nomao = pd.read_csv(R+"dose_response_nomao.csv")
def cell(df, m):
    x = df[df.minority_count==m]; b = x[x.method=="Baseline"].set_index("seed").auc_roc; c = x[x.method=="CTGAN"].set_index("seed").auc_roc; d = (c-b.reindex(c.index))*100
    p = pv(d); ps = "p<0.001" if p < .001 else f"p={q(p,3)}"
    return f"{sgn(d.mean(),2)} ± {q(hw(d),2)} ({ps})"
rows6 = [[f"{m:,}", rate, cell(bank, m), cell(nomao, m)] for m, rate in [(16,"0.16%"),(64,"0.64%"),(256,"2.56%"),(512,"5.12%"),(1024,"10.24%")]]
write("table6_with_ci.csv", ["Minority count","Positive rate","Bank Marketing gain (points)","Nomao gain (points)"], rows6)
