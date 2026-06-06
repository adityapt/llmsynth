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
#   This script reads the CSVs saved by the canonical loaders:
#     - hillstrom.csv         (saved by experiments/run_hillstrom.load_hillstrom)
#     - criteo_uplift.csv     (saved by experiments/run_criteo.load_criteo_uplift)
#
#   These files are ALREADY preprocessed when written to disk: 'target' column
#   renamed from 'conversion', treatment/segment/visit columns dropped, object
#   columns label-encoded. Upload them once to WORK_DIR — typically the same
#   `hillstrom.csv` used by run_great_hillstrom_databricks.py is what you want.
#
# TO RE-RUN CRITEO ONLY: delete ci_tabddpm_criteo.csv from WORK_DIR first so
#   the resume logic doesn't skip any seeds, then run this notebook. Hillstrom
#   will be skipped automatically (all 5 seeds already in ci_tabddpm_hillstrom.csv).
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
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TabDDPM hyperparameters (synthcity "ddpm" plugin). These are reasonable
# defaults for ~8k-row training sets; bump N_ITER if fidelity looks weak.
N_ITER        = 2000     # training iterations
NUM_TIMESTEPS = 1000     # diffusion steps
DDPM_LR       = 1e-3
DDPM_BATCH    = 1024

# Per-dataset config:
#   (display_name, raw_csv_path, preload_nrows, pool_shuffle_seed, output_csv_path)
# preload_nrows: nrows passed to pd.read_csv; None means read all.
# pool_shuffle_seed: if not None, shuffle df_full with this seed after loading
#   to match the canonical loader's row ordering. Criteo uses 42 because
#   load_criteo_uplift() calls df.sample(..., random_state=42) on the 50K pool —
#   without this shuffle the per-seed cap sampling draws different rows and the
#   Baseline AUCs diverge from ci_criteo.csv.
WORK_DIR = os.environ.get("LLMSYNTH_WORK_DIR", "/Workspace/Users/<your-username>/Temp")
DATASETS = [
    ("Hillstrom Email", f"{WORK_DIR}/hillstrom.csv",     None,      None, f"{WORK_DIR}/ci_tabddpm_hillstrom.csv"),
    ("Criteo Display",  f"{WORK_DIR}/criteo_uplift.csv", N_CAP * 5, 42,   f"{WORK_DIR}/ci_tabddpm_criteo.csv"),
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


def load_prepared(data_path, preload_nrows, pool_shuffle_seed):
    """Read a loader-prepared CSV from WORK_DIR.

    preload_nrows caps the read (Criteo: 50K to match load_criteo_uplift).
    pool_shuffle_seed re-applies the canonical loader's post-read shuffle so
    the per-seed cap sampling draws the same rows as the reference CI runs.
    """
    df = pd.read_csv(data_path, nrows=preload_nrows)
    df = df.dropna().reset_index(drop=True)
    assert TARGET in df.columns, f"{data_path} missing '{TARGET}' column"
    non_numeric = [c for c in df.columns
                   if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
    assert not non_numeric, (
        f"{data_path} has non-numeric feature columns {non_numeric}; "
        f"re-run the loader to regenerate it."
    )
    if pool_shuffle_seed is not None:
        df = df.sample(len(df), random_state=pool_shuffle_seed).reset_index(drop=True)
    return df


def run_dataset(name, data_path, preload_nrows, pool_shuffle_seed, out_path):
    print(f"\n{'='*64}\nTabDDPM CI experiment: {name}\n{'='*64}", flush=True)
    df_full = load_prepared(data_path, preload_nrows, pool_shuffle_seed)
    print(f"Loaded {df_full.shape}, positive rate {df_full[TARGET].mean()*100:.2f}%", flush=True)

    n_use = min(N_CAP, len(df_full))

    # Resume support — a seed counts as "done" only if it has a non-NaN α=1.0 row.
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict("records")
        done = {
            int(r["seed"]) for r in rows
            if r["method"] == "TabDDPM"
            and r["alpha"] == ALPHAS[-1]
            and pd.notna(r.get("auc_roc"))
        }
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
    for name, data_path, preload_nrows, pool_shuffle_seed, out_path in DATASETS:
        if not os.path.exists(data_path):
            print(f"\n[skip] {name}: {data_path} not found — upload it to WORK_DIR "
                  f"(see DATA CONTRACT header).", flush=True)
            continue
        run_dataset(name, data_path, preload_nrows, pool_shuffle_seed, out_path)
    print("\nDone.", flush=True)
