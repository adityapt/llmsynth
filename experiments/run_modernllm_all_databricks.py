%python
# Modern LLM tabular synthesis — all three GReaT datasets
#
# Replicates the FULL GReaT evaluation protocol:
#
#   Part 1 — Small-n sweep (matches run_great_*_databricks.py):
#     Vary n ∈ SMALL_NS at α=1.0; 5 seeds per (dataset, n)
#
#   Part 2 — Alpha sweep (matches run_great_alpha_sweep_*_databricks.py):
#     Fix n ∈ SMALL_NS_SWEEP={50,100,200}; sweep α ∈ {0.1,0.2,0.3,0.5,1.0}
#     Fit ONCE per (n, seed), subsample for each α — matched design
#
# Only change vs original GReaT scripts: llm="gpt2" → LLM_MODEL (7B+)
#
# Output files:
#   {WORK_DIR}/modernllm_german_results.csv       ← Part 1
#   {WORK_DIR}/modernllm_hillstrom_results.csv    ← Part 1
#   {WORK_DIR}/modernllm_telco_results.csv        ← Part 1
#   {WORK_DIR}/modernllm_alpha_german_results.csv  ← Part 2
#   {WORK_DIR}/modernllm_alpha_hillstrom_results.csv ← Part 2
#   {WORK_DIR}/modernllm_alpha_telco_results.csv   ← Part 2
#
# SETUP (run once):
#   %pip install be_great transformers accelerate
#   dbutils.library.restartPython()
#
# GPU requirements (fp16):
#   mistralai/Mistral-7B-v0.1   → A10G (24GB) recommended
#   EleutherAI/gpt-j-6b         → A10G (16GB) or T4 (tight)
#   EleutherAI/gpt-neo-2.7B     → T4 (16GB) safe fallback

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
# Model options by GPU memory (fp16):
#   T4  16GB  → EleutherAI/gpt-neo-2.7B (~6GB)  ← default, safe on T4
#               EleutherAI/gpt-j-6b       (~12GB) ← fits T4 if little else running
#   A10G 24GB → mistralai/Mistral-7B-v0.1 (~14GB) ← set LLMSYNTH_LLM_MODEL
LLM_MODEL      = os.environ.get("LLMSYNTH_LLM_MODEL", "EleutherAI/gpt-neo-2.7B")
WORK_DIR       = os.environ.get("LLMSYNTH_WORK_DIR", "/Workspace/Users/<your-username>/Temp")
SEEDS          = [42, 123, 7, 2024, 999]
ALPHAS         = [0.1, 0.2, 0.3, 0.5, 1.0]
SMALL_NS_SWEEP = [50, 100, 200]   # n values for alpha sweep (shoulder region)
BATCH_SIZE     = 8                # gpt-neo-2.7B is smaller; increase if OOM reduce to 4

print(f"Model     : {LLM_MODEL}", flush=True)
print(f"GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
if torch.cuda.is_available():
    print(f"GPU VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# Per-dataset config — matches original GReaT scripts exactly
DATASETS = [
    {
        "name":       "German Credit",
        "data_path":  f"{WORK_DIR}/credit_default.csv",
        "out_smalln": f"{WORK_DIR}/modernllm_german_results.csv",
        "out_alpha":  f"{WORK_DIR}/modernllm_alpha_german_results.csv",
        "holdout_n":  200,
        "small_ns":   [50, 100, 200, 500],
        "pos_rate_pct": 30.0,
    },
    {
        "name":       "Hillstrom Email",
        "data_path":  f"{WORK_DIR}/hillstrom.csv",
        "out_smalln": f"{WORK_DIR}/modernllm_hillstrom_results.csv",
        "out_alpha":  f"{WORK_DIR}/modernllm_alpha_hillstrom_results.csv",
        "holdout_n":  10000,
        "small_ns":   [50, 100, 200, 500, 1000, 2000],
        "pos_rate_pct": 0.9,
    },
    {
        "name":       "Telco Churn",
        "data_path":  f"{WORK_DIR}/telco_churn.csv",
        "out_smalln": f"{WORK_DIR}/modernllm_telco_results.csv",
        "out_alpha":  f"{WORK_DIR}/modernllm_alpha_telco_results.csv",
        "holdout_n":  2000,
        "small_ns":   [50, 100, 200, 500, 1000, 2000],
        "pos_rate_pct": 26.6,
    },
]

TARGET = "target"


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


def fit_and_sample(df_tr, n_samples, llm_model, batch_size, epochs, ckpt_dir):
    """Fit GReaT with modern LLM and return n_samples synthetic rows."""
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    model = GReaT(
        llm=llm_model,
        batch_size=batch_size,
        epochs=epochs,
        fp16=True,
        experiment_dir=ckpt_dir,
        logging_steps=1,
        logging_strategy="epoch",
    )
    print(f"  [fit]", end="", flush=True)
    model.fit(df_tr)
    print(f"  [sample n={n_samples}]", end="", flush=True)
    df_syn = model.sample(n_samples, guided_sampling=True, max_length=2000)
    if len(df_syn) == 0:
        raise ValueError("Empty sample returned")
    df_syn.columns = df_tr.columns
    df_syn[TARGET] = pd.to_numeric(df_syn[TARGET], errors="coerce").fillna(0).astype(int)
    return df_syn


def evaluate(X_tr, y_tr, X_syn, y_syn, X_ho, y_ho, seed):
    X_aug = np.vstack([X_tr, X_syn])
    y_aug = np.concatenate([y_tr, y_syn])
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_aug, y_aug)
    return roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])


def run_dataset(cfg):
    name      = cfg["name"]
    data_path = cfg["data_path"]
    holdout_n = cfg["holdout_n"]
    small_ns  = cfg["small_ns"]
    slug      = name.split()[0].lower()

    print(f"\n{'='*64}", flush=True)
    print(f"Dataset: {name}  ({cfg['pos_rate_pct']}% positive rate)", flush=True)
    print(f"{'='*64}", flush=True)

    if not os.path.exists(data_path):
        print(f"  [skip] {data_path} not found — upload to WORK_DIR first.", flush=True)
        return

    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}, positive rate: {df[TARGET].mean()*100:.2f}%", flush=True)

    _, df_holdout = train_test_split(df, test_size=holdout_n, random_state=42,
                                      stratify=df[TARGET])
    X_ho = df_holdout.drop(columns=[TARGET]).values.astype(float)
    y_ho = df_holdout[TARGET].values
    df_pool = df.drop(df_holdout.index).reset_index(drop=True)
    pos_pool = df_pool[df_pool[TARGET] == 1]
    neg_pool = df_pool[df_pool[TARGET] == 0]

    # ── Part 1: Small-n sweep (α=1.0) ────────────────────────────────────────
    out_smalln = cfg["out_smalln"]
    if os.path.exists(out_smalln):
        rows1 = pd.read_csv(out_smalln).to_dict("records")
        done1 = {(r["n"], r["seed"]) for r in rows1 if r["method"] == "Baseline"}
    else:
        rows1, done1 = [], set()

    print(f"\n  [Part 1] Small-n sweep (α=1.0)", flush=True)
    for n_train in small_ns:
        print(f"\n  n={n_train}:", flush=True)
        for seed in SEEDS:
            if (n_train, seed) in done1:
                print(f"    seed={seed} skip", flush=True)
                continue

            seed_everything(seed)
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
            rows1.append({"n": n_train, "seed": seed, "method": "Baseline",
                          "auc": base_auc, "error": "", "model": "—"})
            print(f"    seed={seed}  Baseline={base_auc:.4f}", end="", flush=True)

            try:
                epochs = 100 if n_train <= 100 else 50
                df_syn = fit_and_sample(df_tr, n_train, LLM_MODEL, BATCH_SIZE,
                                        epochs, f"/tmp/great_ckpt_{slug}_smalln")
                auc = evaluate(X_tr, y_tr,
                               df_syn.drop(columns=[TARGET]).values.astype(float),
                               df_syn[TARGET].values, X_ho, y_ho, seed)
                rows1.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                              "auc": auc, "error": "", "model": LLM_MODEL})
                print(f"  ModernLLM={auc:.4f}", flush=True)
            except Exception as e:
                err = str(e)[:200]
                rows1.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                              "auc": float("nan"), "error": err, "model": LLM_MODEL})
                print(f"  FAIL({err[:60]})", flush=True)

            pd.DataFrame(rows1).to_csv(out_smalln, index=False)

    # ── Part 2: Alpha sweep (matched design, n ∈ SMALL_NS_SWEEP) ─────────────
    out_alpha = cfg["out_alpha"]
    if os.path.exists(out_alpha):
        rows2 = pd.read_csv(out_alpha).to_dict("records")
        done2 = {(r["n"], r["seed"]) for r in rows2
                 if r["method"] == "Baseline" and r["alpha"] == 0.0}
    else:
        rows2, done2 = [], set()

    print(f"\n  [Part 2] Alpha sweep (n ∈ {SMALL_NS_SWEEP}, α ∈ {ALPHAS})", flush=True)
    for n_train in SMALL_NS_SWEEP:
        if n_train not in small_ns:
            continue
        print(f"\n  n={n_train}:", flush=True)
        for seed in SEEDS:
            if (n_train, seed) in done2:
                print(f"    seed={seed} skip", flush=True)
                continue

            seed_everything(seed)
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
            rows2.append({"n": n_train, "seed": seed, "method": "Baseline",
                          "alpha": 0.0, "auc": base_auc, "error": "", "model": "—"})
            print(f"    seed={seed}  Baseline={base_auc:.4f}", flush=True)

            # Fit ONCE at max alpha, subsample for smaller alphas
            try:
                n_syn_max = int(round(max(ALPHAS) * n_train))
                epochs = 100 if n_train <= 100 else 50
                df_syn_max = fit_and_sample(df_tr, n_syn_max, LLM_MODEL, BATCH_SIZE,
                                            epochs, f"/tmp/great_ckpt_{slug}_alpha")
                for alpha in ALPHAS:
                    n_syn = int(round(alpha * n_train))
                    if n_syn == 0:
                        rows2.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                                      "alpha": alpha, "auc": base_auc,
                                      "error": "alpha*n=0", "model": LLM_MODEL})
                        continue
                    df_syn = df_syn_max.head(n_syn)
                    auc = evaluate(X_tr, y_tr,
                                   df_syn.drop(columns=[TARGET]).values.astype(float),
                                   df_syn[TARGET].values, X_ho, y_ho, seed)
                    rows2.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                                  "alpha": alpha, "auc": auc, "error": "", "model": LLM_MODEL})
                    print(f"      α={alpha}  n_syn={n_syn}  ModernLLM={auc:.4f}", flush=True)
            except Exception as e:
                err = str(e)[:200]
                for alpha in ALPHAS:
                    rows2.append({"n": n_train, "seed": seed, "method": "ModernLLM",
                                  "alpha": alpha, "auc": float("nan"),
                                  "error": err, "model": LLM_MODEL})
                print(f"      FAIL({err[:80]})", flush=True)

            pd.DataFrame(rows2).to_csv(out_alpha, index=False)

    # Summary
    print(f"\n  {name} done. Saved:", flush=True)
    print(f"    Part 1 → {out_smalln}", flush=True)
    print(f"    Part 2 → {out_alpha}", flush=True)


# ── Run all datasets ──────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    for cfg in DATASETS:
        run_dataset(cfg)
    print("\nAll datasets done.", flush=True)
