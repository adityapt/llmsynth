%python
# =============================================================================
# TabDDPM on the imbalanced marketing datasets (Hillstrom, Criteo) — GPU/Databricks
# =============================================================================
# Item #1 from the 2026-06 review: add the diffusion augmentation champion as a
# comparator so §6.8 reports TabDDPM alongside GaussianCopula / CTGAN / SMOTE.
# TabDDPM (Kotelnikov et al., ICML 2023) is the standard augmentation-utility
# baseline; its absence is the paper's principal scope limitation.
#
# DESIGN PARITY with experiments/run_confidence_intervals.py (§6.8 CI protocol):
#   - N_CAP = 10_000 rows, sampled with random_state=seed
#   - 80/20 stratified train/test split, random_state=seed
#   - SEEDS = [42, 123, 7, 2024, 999]
#   - ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]   (synthetic fraction = alpha * n_train)
#   - downstream model: GradientBoostingClassifier(n_estimators=100, max_depth=4,
#     random_state=seed)  ← identical to baseline/CTGAN/SMOTE branches
#   - metrics: auc_roc, f1_minority, avg_precision
#   - output schema: seed, method, condition, alpha, auc_roc, f1_minority, avg_precision
#
# Because the Baseline branch here is bit-identical to run_confidence_intervals.py
# (same seeds, same split, same classifier), the Baseline AUCs in the output CSV
# should match results/ci_hillstrom.csv / ci_criteo.csv exactly — a free
# cross-check that the harness is wired correctly before trusting the TabDDPM rows.
#
# GENERATOR: TabDDPM via synthcity's "ddpm" plugin. Sampling is UNCONDITIONAL
# (generate n_syn rows from the learned joint), matching how generate_ctgan /
# generate_gaussian_copula sample in the canonical harness — this keeps the
# real+synthetic augmentation comparison fair across generators.
#
# -----------------------------------------------------------------------------
# SETUP (run once on a GPU cluster, e.g. g4dn.xlarge single node):
#   %pip install synthcity==0.2.11 scikit-learn scipy pandas numpy
#   dbutils.library.restartPython()
# synthcity pulls a compatible torch; on a GPU node it will use CUDA automatically.
#
# DATA CONTRACT:
#   This script consumes a PREPARED CSV that already has a numeric `target` column
#   and numeric feature columns — i.e. the dataframe returned by
#   experiments/run_hillstrom.load_hillstrom() / run_criteo.load_criteo_uplift().
#   Export it once locally so splits match the paper exactly:
#
#       from experiments.run_hillstrom import load_hillstrom
#       df, target, task, name = load_hillstrom()
#       df.to_csv("hillstrom_prepared.csv", index=False)   # then upload to WORK_DIR
#
#   (same for load_criteo_uplift -> criteo_prepared.csv). Upload both to WORK_DIR.
# =============================================================================

import warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy import stats

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# ── Config ────────────────────────────────────────────────────────────────
TARGET   = "target"
SEEDS    = [42, 123, 7, 2024, 999]
ALPHAS   = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP    = 10_000
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# TabDDPM hyperparameters (synthcity "ddpm" plugin). These are reasonable
# defaults for ~8k-row training sets; bump N_ITER if fidelity looks weak.
N_ITER        = 2000     # training iterations
NUM_TIMESTEPS = 1000     # diffusion steps
DDPM_LR       = 1e-3
DDPM_BATCH    = 1024

# Which datasets to run. Each entry: (key, prepared_csv_filename, output_csv).
WORK_DIR = os.environ.get("LLMSYNTH_WORK_DIR", "/Workspace/Users/<your-username>/Temp")
DATASETS = [
    ("Hillstrom Email", f"{WORK_DIR}/hillstrom_prepared.csv", f"{WORK_DIR}/ci_tabddpm_hillstrom.csv"),
    ("Criteo Display",  f"{WORK_DIR}/criteo_prepared.csv",    f"{WORK_DIR}/ci_tabddpm_criteo.csv"),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr) - 1))


def evaluate(X_tr, y_tr, X_te, y_te, seed):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return {
        "auc_roc":       roc_auc_score(y_te, proba),
        "f1_minority":   f1_score(y_te, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_te, proba),
    }


def tabddpm_sample(df_train, n_syn, seed):
    """Fit TabDDPM on df_train and return n_syn synthetic rows (unconditional)."""
    loader = GenericDataLoader(df_train, target_column=TARGET)
    model = Plugins().get(
        "ddpm",
        n_iter=N_ITER,
        lr=DDPM_LR,
        batch_size=DDPM_BATCH,
        num_timesteps=NUM_TIMESTEPS,
        is_classification=True,
        device=DEVICE,
        random_state=seed,
    )
    model.fit(loader)
    syn = model.generate(count=n_syn).dataframe()
    # Align columns/types to the training frame
    syn = syn[df_train.columns]
    syn[TARGET] = pd.to_numeric(syn[TARGET], errors="coerce").fillna(0).astype(int)
    return syn


def run_dataset(name, data_path, out_path):
    print(f"\n{'='*64}\nTabDDPM CI experiment: {name}\n{'='*64}", flush=True)
    df_full = pd.read_csv(data_path)
    assert TARGET in df_full.columns, f"{data_path} must contain a '{TARGET}' column"
    print(f"Loaded {df_full.shape}, positive rate {df_full[TARGET].mean()*100:.2f}%", flush=True)

    n_use = min(N_CAP, len(df_full))

    # Resume support
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict("records")
        done = {int(r["seed"]) for r in rows if r["method"] == "TabDDPM" and r["alpha"] == ALPHAS[-1]}
        print(f"Resuming — {len(done)} seeds already complete", flush=True)
    else:
        rows, done = [], set()

    for seed in SEEDS:
        if seed in done:
            print(f"  seed {seed} already done, skipping", flush=True)
            continue
        print(f"\n  Seed {seed}:", flush=True)
        seed_everything(seed)

        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[TARGET]
        )
        X_tr = df_train.drop(columns=[TARGET]).values.astype(float)
        y_tr = df_train[TARGET].values
        X_te = df_test.drop(columns=[TARGET]).values.astype(float)
        y_te = df_test[TARGET].values

        # Baseline (should match results/ci_*.csv Baseline rows exactly)
        m = evaluate(X_tr, y_tr, X_te, y_te, seed)
        rows.append({"seed": seed, "method": "Baseline", "condition": "real_only",
                     "alpha": 0, **m})
        print(f"    Baseline: AUC={m['auc_roc']:.4f}", flush=True)

        # Fit TabDDPM once at the largest alpha, subsample for smaller alphas
        try:
            n_syn_max = int(len(df_train) * max(ALPHAS))
            print(f"    [TabDDPM] fitting (n_iter={N_ITER}, device={DEVICE})...", flush=True)
            syn_max = tabddpm_sample(df_train, n_syn_max, seed)
            for alpha in ALPHAS:
                n_syn = int(len(df_train) * alpha)
                if n_syn == 0:
                    continue
                syn = syn_max.head(n_syn)
                X_aug = np.vstack([X_tr, syn.drop(columns=[TARGET]).values.astype(float)])
                y_aug = np.concatenate([y_tr, syn[TARGET].values])
                m = evaluate(X_aug, y_aug, X_te, y_te, seed)
                rows.append({"seed": seed, "method": "TabDDPM", "condition": "augmented",
                             "alpha": alpha, **m})
                print(f"    TabDDPM α={alpha}: AUC={m['auc_roc']:.4f}", flush=True)
        except Exception as e:
            err = str(e)[:200]
            for alpha in ALPHAS:
                rows.append({"seed": seed, "method": "TabDDPM", "condition": "augmented",
                             "alpha": alpha, "auc_roc": float("nan"),
                             "f1_minority": float("nan"), "avg_precision": float("nan")})
            print(f"    TabDDPM FAILED: {err}", flush=True)

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"    [saved {out_path}]", flush=True)

    # Summary
    df_out = pd.DataFrame(rows)
    base_mean, _ = ci95(df_out[df_out["method"] == "Baseline"]["auc_roc"].values)
    print(f"\n  {name} — TabDDPM (mean ± 95% CI, gain vs baseline):", flush=True)
    print(f"  {'alpha':>6}  {'AUC':>16}  {'gain':>8}", flush=True)
    aug = df_out[df_out["method"] == "TabDDPM"]
    for alpha in ALPHAS:
        vals = aug[aug["alpha"] == alpha]["auc_roc"].values
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        print(f"  {alpha:>6}  {m:>7.4f} ± {h:<6.4f}  {m-base_mean:>+8.4f}", flush=True)
    return df_out


if __name__ == "__main__" or True:  # Databricks runs top-level
    for name, data_path, out_path in DATASETS:
        if not os.path.exists(data_path):
            print(f"\n[skip] {name}: {data_path} not found — export & upload it first "
                  f"(see DATA CONTRACT header).", flush=True)
            continue
        run_dataset(name, data_path, out_path)
    print("\nDone.", flush=True)
