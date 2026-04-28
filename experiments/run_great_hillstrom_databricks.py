%python
# GReaT on Hillstrom — tests whether semantic feature names unlock LLM priors
# Hillstrom features: recency, history, mens, womens, zip_code, newbie, channel
# Positive rate: 0.9% — extreme imbalance + semantic features = best GReaT candidate

import warnings, os, shutil
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
HOLDOUT_N = 500
DATA_PATH = "/Workspace/Users/adityapu@zillowgroup.com/Temp/hillstrom.csv"
OUT_PATH  = "/Workspace/Users/adityapu@zillowgroup.com/Temp/great_hillstrom_results.csv"

def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr)-1))

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}, positive rate: {df[TARGET].mean()*100:.2f}%", flush=True)
print(f"Features: {list(df.columns)}", flush=True)

_, df_holdout = train_test_split(df, test_size=HOLDOUT_N, random_state=42,
                                  stratify=df[TARGET])
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values
df_pool = df.drop(df_holdout.index).reset_index(drop=True)

# Resume if interrupted
if os.path.exists(OUT_PATH):
    rows = pd.read_csv(OUT_PATH).to_dict("records")
    done = {(r["n"], r["seed"]) for r in rows if r["method"] == "Baseline"}
    print(f"Resuming — {len(done)} pairs already done", flush=True)
else:
    rows = []
    done = set()

for n_train in SMALL_NS:
    print(f"\n=== n={n_train} ===", flush=True)
    for seed in SEEDS:
        if (n_train, seed) in done:
            print(f"  seed={seed} already done, skipping", flush=True)
            continue

        np.random.seed(seed)
        df_tr = df_pool.sample(min(n_train, len(df_pool)),
                               random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET]).values.astype(float)
        y_tr = df_tr[TARGET].values

        # Baseline
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
        rows.append({"n": n_train, "seed": seed, "method": "Baseline",
                     "auc": base_auc, "error": ""})
        print(f"  seed={seed}  Baseline={base_auc:.4f}", end="", flush=True)

        # GReaT
        try:
            epochs = 100 if n_train <= 100 else 50
            ckpt_dir = "/tmp/great_ckpt_hillstrom"
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            model = GReaT(llm="gpt2", batch_size=32, epochs=epochs, fp16=True,
                          experiment_dir=ckpt_dir,
                          logging_steps=1, logging_strategy="epoch")
            print(f"  [GReaT] fitting...", end="", flush=True)
            model.fit(df_tr)
            print(f"  sampling...", end="", flush=True)
            df_syn = model.sample(n_train, guided_sampling=True, max_length=2000)

            if len(df_syn) == 0:
                raise ValueError("Empty sample returned")
            df_syn.columns = df_tr.columns
            df_syn[TARGET] = pd.to_numeric(df_syn[TARGET],
                                           errors="coerce").fillna(0).astype(int)
            X_aug = np.vstack([X_tr,
                               df_syn.drop(columns=[TARGET]).values.astype(float)])
            y_aug = np.concatenate([y_tr, df_syn[TARGET].values])
            clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                              random_state=seed)
            clf2.fit(X_aug, y_aug)
            auc = roc_auc_score(y_ho, clf2.predict_proba(X_ho)[:, 1])
            rows.append({"n": n_train, "seed": seed, "method": "GReaT",
                         "auc": auc, "error": ""})
            print(f"  GReaT={auc:.4f}", flush=True)
        except Exception as e:
            err = str(e)[:200]
            rows.append({"n": n_train, "seed": seed, "method": "GReaT",
                         "auc": float("nan"), "error": err})
            print(f"  GReaT=FAIL({err})", flush=True)

        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"  [saved]", flush=True)

print("\n\nSummary (mean ± 95% CI):", flush=True)
df_out = pd.DataFrame(rows)
print(f"{'n':>5}  {'Method':<10}  {'AUC':>10}  {'±CI95':>8}  {'vs Baseline':>12}")
print("-"*52)
for n in SMALL_NS:
    sub = df_out[df_out["n"] == n]
    if sub.empty:
        continue
    bm, _ = ci95(sub[sub["method"] == "Baseline"]["auc"].values)
    for meth in ["Baseline", "GReaT"]:
        vals = sub[sub["method"] == meth]["auc"].values
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        gain = m - bm if meth != "Baseline" else 0.0
        print(f"{n:>5}  {meth:<10}  {m:>10.4f}  {h:>8.4f}  {gain:>+12.4f}")
