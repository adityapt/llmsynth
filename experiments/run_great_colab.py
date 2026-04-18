# ── Cell 1: Install dependencies ──────────────────────────────────────────────
# !pip install be_great -q

# ── Cell 2: Upload data ───────────────────────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()   # upload credit_default.csv

# ── Cell 3: Run experiment ────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from be_great import GReaT

TARGET    = "target"
SEEDS     = [42, 123, 7, 2024, 999]
SMALL_NS  = [50, 100, 200, 500]
HOLDOUT_N = 200

def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr)-1))

df = pd.read_csv("credit_default.csv")
print(f"Loaded: {df.shape}, positive rate: {df[TARGET].mean()*100:.1f}%")

_, df_holdout = train_test_split(df, test_size=HOLDOUT_N, random_state=42,
                                  stratify=df[TARGET])
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values
df_pool = df.drop(df_holdout.index).reset_index(drop=True)

rows = []

for n_train in SMALL_NS:
    print(f"\n=== n={n_train} ===")
    for seed in SEEDS:
        np.random.seed(seed)
        df_tr = df_pool.sample(min(n_train, len(df_pool)),
                               random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET]).values.astype(float)
        y_tr = df_tr[TARGET].values

        # Baseline
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         random_state=seed)
        clf.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
        rows.append({"n": n_train, "seed": seed, "method": "Baseline", "auc": base_auc})
        print(f"  seed={seed}  Baseline={base_auc:.4f}", end="", flush=True)

        # GReaT
        try:
            epochs = 100 if n_train <= 100 else 50
            model = GReaT(llm="gpt2", batch_size=32, epochs=epochs, fp16=True)
            model.fit(df_tr)
            df_syn = model.sample(n_train, guided_sampling=True, max_length=2000)
            if len(df_syn) == 0:
                raise ValueError("Empty sample")
            df_syn.columns = df_tr.columns
            df_syn[TARGET] = pd.to_numeric(df_syn[TARGET], errors="coerce").fillna(0).astype(int)
            X_aug = np.vstack([X_tr, df_syn.drop(columns=[TARGET]).values.astype(float)])
            y_aug = np.concatenate([y_tr, df_syn[TARGET].values])
            clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
            clf2.fit(X_aug, y_aug)
            auc = roc_auc_score(y_ho, clf2.predict_proba(X_ho)[:, 1])
            rows.append({"n": n_train, "seed": seed, "method": "GReaT", "auc": auc})
            print(f"  GReaT={auc:.4f}", flush=True)
        except Exception as e:
            rows.append({"n": n_train, "seed": seed, "method": "GReaT", "auc": float("nan")})
            print(f"  GReaT=FAIL({e})", flush=True)

        # Save after every seed
        pd.DataFrame(rows).to_csv("great_results.csv", index=False)

print("\n\nSummary (mean ± 95% CI):")
df_out = pd.DataFrame(rows)
print(f"{'n':>5}  {'Method':<10}  {'AUC':>10}  {'±CI95':>8}  {'vs Baseline':>12}")
print("-"*52)
for n in SMALL_NS:
    sub = df_out[df_out["n"] == n]
    if sub.empty: continue
    bm, _ = ci95(sub[sub["method"]=="Baseline"]["auc"].values)
    for meth in ["Baseline", "GReaT"]:
        vals = sub[sub["method"]==meth]["auc"].values
        if len(vals) == 0: continue
        m, h = ci95(vals)
        gain = m - bm if meth != "Baseline" else 0.0
        print(f"{n:>5}  {meth:<10}  {m:>10.4f}  {h:>8.4f}  {gain:>+12.4f}")

# ── Cell 4: Download results ──────────────────────────────────────────────────
# from google.colab import files
# files.download("great_results.csv")
