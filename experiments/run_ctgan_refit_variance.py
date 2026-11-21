"""CTGAN fit-to-fit variance test (Section 4.5 of the Electronics manuscript).

Holds data, train/test split, NumPy seed and downstream classifier seed fixed, refits CTGAN
several times (PyTorch is never seeded, as in the main experiments), and records the downstream
AUC of Gradient Boosting trained on real + synthetic rows. GaussianCopula is refit as a control.
Output: results/ctgan_refit_variance.csv
Usage (from repo root): python3 experiments/run_ctgan_refit_variance.py
"""
import sys, random, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from experiments.synthetic_data_eval import generate_ctgan, generate_gaussian_copula, RESULTS_DIR
from experiments.run_hillstrom import load_hillstrom

SEED, ALPHA, N_FITS = 42, 0.2, 5
df_full, target, task, name = load_hillstrom()
df = df_full.sample(10_000, random_state=SEED).reset_index(drop=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[target])
X_tr, y_tr = df_train.drop(columns=[target]).values.astype(float), df_train[target].values
X_te, y_te = df_test.drop(columns=[target]).values.astype(float), df_test[target].values
n_syn = int(len(df_train) * ALPHA)
rows = []
for gen_name, fn, n in [("CTGAN", generate_ctgan, N_FITS), ("GaussianCopula", generate_gaussian_copula, 3)]:
    for k in range(n):
        np.random.seed(SEED); random.seed(SEED)          # NumPy/Python seeded, PyTorch not (seed=-1 switches off the PyTorch seeding in generate_ctgan)
        syn = fn(df_train, target, n_syn, task, seed=-1) if gen_name == "CTGAN" else fn(df_train, target, n_syn, task)
        X = np.vstack([X_tr, syn.drop(columns=[target]).values.astype(float)])
        y = np.concatenate([y_tr, syn[target].values])
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED).fit(X, y)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        rows.append({"dataset": name, "generator": gen_name, "alpha": ALPHA, "fit": k + 1, "auc_roc": auc})
        print(f"{gen_name} fit {k+1}: AUC={auc:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "ctgan_refit_variance.csv", index=False)
        gc.collect()
