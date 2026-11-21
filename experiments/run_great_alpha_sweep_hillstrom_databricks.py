%python
# Phase 5 α-sweep — GReaT on Hillstrom Email Marketing at small-n shoulder.
# Pre-registered in papers/improvement_plan.md (Phase 5).
#
# For each (n, seed): fit GReaT once, sample n synthetic rows, evaluate at each α
# by subsampling first int(α·n) rows. Same seed → same training subset → same
# GReaT fit → matched-design comparison across α values.
#
# Output: {WORK_DIR}/great_alpha_sweep_hillstrom_results.csv
# Schema: n, seed, method, alpha, auc, error
#         Baseline rows have alpha=0.0.

import warnings, os, shutil, random
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from be_great import GReaT
from transformers import set_seed as _hf_set_seed


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch / CUDA / transformers / cuDNN for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

TARGET    = "target"
SEEDS     = [42, 123, 7, 2024, 999]
SMALL_NS  = [50, 100, 200]
ALPHAS    = [0.1, 0.2, 0.3, 0.5, 1.0]
HOLDOUT_N = 10000

# Set LLMSYNTH_WORK_DIR env var (or replace the default below) before running.
WORK_DIR  = os.environ.get("LLMSYNTH_WORK_DIR", "./work")
DATA_PATH = f"{WORK_DIR}/hillstrom.csv"
OUT_PATH  = f"{WORK_DIR}/great_alpha_sweep_hillstrom_results.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}, positive rate: {df[TARGET].mean()*100:.2f}%", flush=True)
print(f"Features: {list(df.columns)}", flush=True)

_, df_holdout = train_test_split(df, test_size=HOLDOUT_N, random_state=42,
                                  stratify=df[TARGET])
X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
y_ho = df_holdout[TARGET].values
df_pool = df.drop(df_holdout.index).reset_index(drop=True)
pos_pool = df_pool[df_pool[TARGET] == 1]
neg_pool = df_pool[df_pool[TARGET] == 0]


# Resume if interrupted
if os.path.exists(OUT_PATH):
    rows = pd.read_csv(OUT_PATH).to_dict("records")
    done = {(int(r["n"]), int(r["seed"])) for r in rows if r["method"] == "Baseline"}
    print(f"Resuming — {len(done)} (n, seed) pairs already done", flush=True)
else:
    rows = []
    done = set()

for n_train in SMALL_NS:
    print(f"\n=== n={n_train} ===", flush=True)
    for seed in SEEDS:
        if (n_train, seed) in done:
            print(f"  seed={seed} already done, skipping", flush=True)
            continue

        seed_everything(seed)
        # Stratified small-n sampling (preserves 0.9% positive rate)
        n_pos = max(1, int(round(n_train * len(pos_pool) / len(df_pool))))
        n_neg = n_train - n_pos
        df_tr = pd.concat([
            pos_pool.sample(n_pos, random_state=seed),
            neg_pool.sample(n_neg, random_state=seed)
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET]).values.astype(float)
        y_tr = df_tr[TARGET].values

        # Baseline — record once per (n, seed)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
        rows.append({"n": n_train, "seed": seed, "method": "Baseline",
                     "alpha": 0.0, "auc": base_auc, "error": ""})
        print(f"  seed={seed}  Baseline={base_auc:.4f}", flush=True)

        # GReaT fit + max-α synthetic sampling
        try:
            epochs = 100 if n_train <= 100 else 50
            ckpt_dir = "/tmp/great_alpha_ckpt_hillstrom"
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            model = GReaT(llm="gpt2", batch_size=32, epochs=epochs, fp16=True,
                          experiment_dir=ckpt_dir,
                          logging_steps=1, logging_strategy="epoch")
            print(f"    [GReaT] fitting...", end="", flush=True)
            model.fit(df_tr)
            print(f"  sampling...", end="", flush=True)
            df_syn = model.sample(n_train, guided_sampling=True, max_length=2000)
            if len(df_syn) == 0:
                raise ValueError("Empty sample returned")
            df_syn.columns = df_tr.columns
            df_syn[TARGET] = pd.to_numeric(df_syn[TARGET],
                                           errors="coerce").fillna(0).astype(int)

            # α-sweep
            for alpha in ALPHAS:
                n_syn = int(round(alpha * n_train))
                if n_syn == 0:
                    rows.append({"n": n_train, "seed": seed, "method": "GReaT",
                                 "alpha": alpha, "auc": base_auc,
                                 "error": "alpha*n=0; no synthetic added"})
                    continue
                df_syn_sub = df_syn.head(n_syn)
                X_aug = np.vstack([X_tr,
                                   df_syn_sub.drop(columns=[TARGET]).values.astype(float)])
                y_aug = np.concatenate([y_tr, df_syn_sub[TARGET].values])
                clf_aug = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                     random_state=seed)
                clf_aug.fit(X_aug, y_aug)
                auc = roc_auc_score(y_ho, clf_aug.predict_proba(X_ho)[:, 1])
                rows.append({"n": n_train, "seed": seed, "method": "GReaT",
                             "alpha": alpha, "auc": auc, "error": ""})
                print(f"    α={alpha}  n_syn={n_syn}  GReaT_AUC={auc:.4f}", flush=True)
        except Exception as e:
            err = str(e)[:200]
            for alpha in ALPHAS:
                rows.append({"n": n_train, "seed": seed, "method": "GReaT",
                             "alpha": alpha, "auc": float("nan"), "error": err})
            print(f"    GReaT failed: {err}", flush=True)

        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"    [saved {OUT_PATH}]", flush=True)

print(f"\nDone. Total rows: {len(rows)}", flush=True)
