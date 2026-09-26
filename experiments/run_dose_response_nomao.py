"""
Dose-response replication on Nomao (second dataset, chosen for headroom and
domain diversity — see run_dose_response_bank_marketing.py for the original
design and rationale). Nomao's full source has 34,465 rows / 9,844 positives
(28.6%); the rest of this paper uses only a 10,000-row/28.3% subsample, so
there is ample headroom for a large fixed holdout plus minority draws up to
1,024+ from the remaining pool. Domain (lead generation) and dimensionality
(119 features) are both very different from Bank Marketing (finance,
17 features) — a genuinely different test bed, not just a second finance
dataset.

Same design as the Bank Marketing sweep: fixed N=10,000, fixed 3,000-row
holdout (stratified at Nomao's natural ~28.6% rate, drawn once), minority
count varied at {16, 64, 256, 512, 1024}, GaussianCopula/CTGAN/SMOTE only,
5 seeds, resumable.

Output: results/dose_response_nomao.csv
"""
import sys, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from experiments.synthetic_data_eval import RESULTS_DIR, DATA_DIR, generate_ctgan, generate_gaussian_copula, generate_smote

SEEDS = [42, 123, 7, 2024, 999]
MINORITY_COUNTS = [16, 64, 256, 512, 1024]
N_TOTAL = 10_000
HOLDOUT_SIZE = 3_000
TARGET = "target"
OUT = RESULTS_DIR / "dose_response_nomao.csv"

df = pd.read_csv(DATA_DIR / "nomao.csv")
for c in df.columns:
    if c != TARGET and df[c].dtype == object:
        df[c] = df[c].astype("category").cat.codes
df = df.fillna(df.median(numeric_only=True))

print(f"Full pool: {df.shape}, positive rate {df[TARGET].mean()*100:.2f}%", flush=True)

_, df_holdout = train_test_split(df, test_size=HOLDOUT_SIZE, random_state=42, stratify=df[TARGET])
df_pool = df.drop(df_holdout.index).reset_index(drop=True)
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values
print(f"Holdout: {len(df_holdout)} rows, {y_ho.sum()} positives (fixed across all conditions)", flush=True)

pos_pool = df_pool[df_pool[TARGET] == 1].reset_index(drop=True)
neg_pool = df_pool[df_pool[TARGET] == 0].reset_index(drop=True)
print(f"Remaining pool: {len(pos_pool)} positives, {len(neg_pool)} negatives available for training draws", flush=True)


def metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
    }


rows = []
if OUT.exists():
    rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["minority_count"], r["seed"], r["method"]) for r in rows}
    print(f"Resuming — {len(done)} (count,seed,method) combos already done", flush=True)
else:
    done = set()

for m_count in MINORITY_COUNTS:
    n_neg = N_TOTAL - m_count
    pos_rate = m_count / N_TOTAL * 100
    print(f"\n=== minority_count={m_count} (positive rate {pos_rate:.2f}%) ===", flush=True)

    for seed in SEEDS:
        df_tr = pd.concat([
            pos_pool.sample(m_count, random_state=seed),
            neg_pool.sample(n_neg, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET])
        y_tr = df_tr[TARGET]

        if (m_count, seed, "Baseline") not in done:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
            clf.fit(X_tr.values.astype(float), y_tr.values)
            m = metrics(y_ho, clf.predict_proba(X_ho)[:, 1], clf.predict(X_ho))
            print(f"  seed={seed} Baseline: AUC={m['auc_roc']:.4f}", flush=True)
            rows.append({"minority_count": m_count, "positive_rate_pct": round(pos_rate, 3),
                         "seed": seed, "method": "Baseline", **m})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            del clf; gc.collect()

        for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
            if (m_count, seed, gen_name) in done:
                continue
            try:
                n_syn = len(df_tr)  # alpha=1.0
                if gen_name == "SMOTE":
                    df_syn = generate_smote(X_tr, y_tr, n_syn)
                else:
                    gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
                    df_syn = gen_fn(df_tr, TARGET, n_syn, "classification")
                X_aug = np.vstack([X_tr.values.astype(float), df_syn.drop(columns=[TARGET]).values.astype(float)])
                y_aug = np.concatenate([y_tr.values, df_syn[TARGET].values])
                clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
                clf2.fit(X_aug, y_aug)
                m = metrics(y_ho, clf2.predict_proba(X_ho)[:, 1], clf2.predict(X_ho))
                print(f"  seed={seed} {gen_name}: AUC={m['auc_roc']:.4f}", flush=True)
                rows.append({"minority_count": m_count, "positive_rate_pct": round(pos_rate, 3),
                             "seed": seed, "method": gen_name, **m})
                del clf2, df_syn, X_aug, y_aug
            except Exception as e:
                print(f"  seed={seed} {gen_name} FAILED: {e}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            gc.collect()

print(f"\nDone. Saved: {OUT}", flush=True)
