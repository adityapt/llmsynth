%python
# Modern LLM-based tabular synthesis on Hillstrom — direct comparison to GReaT/GPT-2
#
# Identical protocol to run_great_hillstrom_databricks.py:
#   - Same SEEDS = [42, 123, 7, 2024, 999]
#   - Same SMALL_NS = [50, 100, 200, 500, 1000, 2000]
#   - Same 10K holdout, stratified small-n sampling, GBC downstream
#   - Same be-great GReaT class, guided_sampling=True
#
# Only change: llm="gpt2" (117M, 2019) → LLM_MODEL (7B+, 2024)
#
# Default model: mistralai/Mistral-7B-v0.1
#   - 7B parameters (~60× larger than GPT-2)
#   - Freely available on HuggingFace
#   - Requires ~14GB GPU RAM in fp16 → needs A10G or better
#   - Fallback: EleutherAI/gpt-j-6b (~12GB, works on most A10G)
#
# SETUP (run once):
#   %pip install be_great transformers accelerate bitsandbytes
#   dbutils.library.restartPython()
#
# OUTPUT: {WORK_DIR}/great_modernllm_hillstrom_results.csv
#   Same schema as great_hillstrom_results.csv → directly comparable

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

# ── Config ────────────────────────────────────────────────────────────────────
# Change LLM_MODEL to switch models. Options:
#   "mistralai/Mistral-7B-v0.1"       — recommended, 7B, ~14GB fp16
#   "EleutherAI/gpt-j-6b"             — 6B, ~12GB fp16, good fallback
#   "meta-llama/Llama-3.2-3B"         — 3B, needs HF token + Llama licence
#   "EleutherAI/gpt-neo-2.7B"         — 2.7B, ~6GB, works on T4
LLM_MODEL  = os.environ.get("LLMSYNTH_LLM_MODEL", "mistralai/Mistral-7B-v0.1")

TARGET     = "target"
SEEDS      = [42, 123, 7, 2024, 999]
SMALL_NS   = [50, 100, 200, 500, 1000, 2000]
HOLDOUT_N  = 10000
BATCH_SIZE = 4        # smaller than GReaT/GPT-2 (32) — 7B models need more VRAM
EPOCHS_SMALL = 100    # n <= 100
EPOCHS_LARGE =  50    # n > 100

WORK_DIR  = os.environ.get("LLMSYNTH_WORK_DIR", "/Workspace/Users/<your-username>/Temp")
DATA_PATH = f"{WORK_DIR}/hillstrom.csv"
OUT_PATH  = f"{WORK_DIR}/great_modernllm_hillstrom_results.csv"

print(f"Model: {LLM_MODEL}", flush=True)
print(f"GPU:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr)-1))


# ── Load data ─────────────────────────────────────────────────────────────────
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

# ── Resume support ────────────────────────────────────────────────────────────
if os.path.exists(OUT_PATH):
    rows = pd.read_csv(OUT_PATH).to_dict("records")
    done = {(r["n"], r["seed"]) for r in rows if r["method"] == "Baseline"}
    print(f"Resuming — {len(done)} (n, seed) pairs already done", flush=True)
else:
    rows = []
    done = set()

# ── Main loop ─────────────────────────────────────────────────────────────────
for n_train in SMALL_NS:
    print(f"\n=== n={n_train} ===", flush=True)
    for seed in SEEDS:
        if (n_train, seed) in done:
            print(f"  seed={seed} already done, skipping", flush=True)
            continue

        seed_everything(seed)

        # Stratified small-n sample (preserves 0.9% positive rate)
        n_pos = max(1, int(round(n_train * len(pos_pool) / len(df_pool))))
        n_neg = n_train - n_pos
        df_tr = pd.concat([
            pos_pool.sample(n_pos, random_state=seed),
            neg_pool.sample(n_neg, random_state=seed)
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        X_tr = df_tr.drop(columns=[TARGET]).values.astype(float)
        y_tr = df_tr[TARGET].values

        # Baseline
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
        rows.append({"n": n_train, "seed": seed, "method": "Baseline",
                     "auc": base_auc, "error": "", "model": "—"})
        print(f"  seed={seed}  Baseline={base_auc:.4f}", end="", flush=True)

        # Modern LLM via GReaT
        try:
            epochs = EPOCHS_SMALL if n_train <= 100 else EPOCHS_LARGE
            ckpt_dir = "/tmp/great_ckpt_modernllm"
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)

            model = GReaT(
                llm=LLM_MODEL,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                fp16=True,
                experiment_dir=ckpt_dir,
                logging_steps=1,
                logging_strategy="epoch",
            )
            print(f"  [LLM-{LLM_MODEL.split('/')[-1]}] fitting...", end="", flush=True)
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
            rows.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                         "auc": auc, "error": "", "model": LLM_MODEL})
            print(f"  ModernLLM={auc:.4f}", flush=True)

        except Exception as e:
            err = str(e)[:200]
            rows.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                         "auc": float("nan"), "error": err, "model": LLM_MODEL})
            print(f"  ModernLLM=FAIL({err})", flush=True)

        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"  [saved]", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\nSummary — {LLM_MODEL} vs GPT-2 protocol (mean ± 95% CI):", flush=True)
df_out = pd.DataFrame(rows)
print(f"{'n':>5}  {'Method':<12}  {'AUC':>10}  {'±CI95':>8}  {'vs Baseline':>12}")
print("-"*55)
for n in SMALL_NS:
    sub = df_out[df_out["n"] == n]
    if sub.empty:
        continue
    bm, _ = ci95(sub[sub["method"] == "Baseline"]["auc"].values)
    for meth in ["Baseline", "ModernLLM"]:
        vals = sub[sub["method"] == meth]["auc"].values
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        gain = m - bm if meth != "Baseline" else 0.0
        print(f"{n:>5}  {meth:<12}  {m:>10.4f}  {h:>8.4f}  {gain:>+12.4f}")

print(f"\nResults saved to: {OUT_PATH}", flush=True)
print("Done.", flush=True)
