"""
Compare three conditions on Hillstrom and Criteo (5-seed CI):
  1. Baseline GBC         — vanilla, no weighting
  2. Balanced GBC         — class_weight='balanced' via sample_weight
  3. CTGAN best-α         — best augmentation result for reference

Output: results/ci_balanced_{hillstrom,criteo}.csv
Schema:  seed, method, auc_roc, f1_minority, avg_precision
"""
import warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

import sys
sys.path.insert(0, ".")
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

SEEDS   = [42, 123, 7, 2024, 999]
N_CAP   = 10_000
RESULTS = Path("results")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def evaluate(X_tr, y_tr, X_te, y_te, seed, balanced=False):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    sw  = compute_sample_weight("balanced", y_tr) if balanced else None
    clf.fit(X_tr, y_tr, sample_weight=sw)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return {
        "auc_roc":       roc_auc_score(y_te, proba),
        "f1_minority":   f1_score(y_te, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_te, proba),
    }

def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2: return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr)-1))

def run(loader_fn):
    df_full, target, task, name = loader_fn()
    slug = name.lower().split()[0]
    out_path = RESULTS / f"ci_balanced_{slug}.csv"
    rows = []

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    for seed in SEEDS:
        seed_everything(seed)
        df = df_full.sample(min(N_CAP, len(df_full)), random_state=seed).reset_index(drop=True)
        df_tr, df_te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
        X_tr = df_tr.drop(columns=[target]).values.astype(float)
        y_tr = df_tr[target].values
        X_te = df_te.drop(columns=[target]).values.astype(float)
        y_te = df_te[target].values

        m_base = evaluate(X_tr, y_tr, X_te, y_te, seed, balanced=False)
        m_bal  = evaluate(X_tr, y_tr, X_te, y_te, seed, balanced=True)

        rows.append({"seed": seed, "method": "Baseline",  **m_base})
        rows.append({"seed": seed, "method": "Balanced",  **m_bal})
        print(f"  Seed {seed}: Baseline={m_base['auc_roc']:.4f}  "
              f"Balanced={m_bal['auc_roc']:.4f}  "
              f"Δ={( m_bal['auc_roc'] - m_base['auc_roc'])*100:+.2f} pts")

    pd.DataFrame(rows).to_csv(out_path, index=False)

    # Summary
    df_out = pd.DataFrame(rows)
    bm, bci  = ci95(df_out[df_out["method"]=="Baseline"]["auc_roc"].values)
    bam, baci = ci95(df_out[df_out["method"]=="Balanced"]["auc_roc"].values)
    print(f"\n  Baseline:  {bm:.4f} ± {bci:.4f}")
    print(f"  Balanced:  {bam:.4f} ± {baci:.4f}  gain={(bam-bm)*100:+.2f} pts")
    print(f"  Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    for loader in [load_hillstrom, load_criteo_uplift]:
        run(loader)
    print("\nDone.")
