"""CTGAN seeding check (Section 4.5 of the Electronics manuscript).

Same setup as run_ctgan_refit_variance.py (Hillstrom, 10,000 rows, seed 42 split, alpha=0.2), but with
PyTorch seeded in generate_ctgan. Two checks:
  1. the same seed twice gives identical synthetic rows and the same downstream AUC (reproducibility);
  2. five different seeds give the spread of downstream AUC that remains once fits are reproducible.
Output: results/ctgan_seed_check.csv
Usage (from the repo root): python3 experiments/run_ctgan_seed_check.py
"""
import sys, random, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from experiments.synthetic_data_eval import generate_ctgan, RESULTS_DIR
from experiments.run_hillstrom import load_hillstrom

SEED, ALPHA = 42, 0.2
df_full, target, task, name = load_hillstrom()
df = df_full.sample(10_000, random_state=SEED).reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[target])
X_tr, y_tr = df_train.drop(columns=[target]).values.astype(float), df_train[target].values
X_te, y_te = df_test.drop(columns=[target]).values.astype(float), df_test[target].values
n_syn = int(len(df_train) * ALPHA)
rows, first = [], {}
for check, seeds in [("same seed, twice", [1, 1]), ("different seeds", [1, 2, 3, 4, 5])]:
    for k, torch_seed in enumerate(seeds):
        np.random.seed(SEED); random.seed(SEED)
        syn = generate_ctgan(df_train, target, n_syn, task, seed=torch_seed)
        identical = bool(torch_seed in first and first[torch_seed].equals(syn))
        first.setdefault(torch_seed, syn)
        X = np.vstack([X_tr, syn.drop(columns=[target]).values.astype(float)])
        y = np.concatenate([y_tr, syn[target].values])
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED).fit(X, y)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        rows.append({"check": check, "fit": k + 1, "torch_seed": torch_seed, "auc_roc": auc,
                     "identical_to_earlier_fit_with_same_seed": identical})
        print(f"{check} fit {k+1} (seed {torch_seed}): AUC={auc:.4f} identical={identical}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "ctgan_seed_check.csv", index=False)
        gc.collect()
