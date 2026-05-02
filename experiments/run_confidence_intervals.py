"""
Confidence interval experiment — multi-seed re-runs of key headline results.

Runs 5 independent seeds for:
  1. Hillstrom Email      (headline: CTGAN +8.3 AUC at α=0.3)
  2. Criteo Display Ads   (headline: CTGAN +19.6 AUC at α=0.5)
  3. German Credit        (GReaT vs CTGAN vs GC at n ∈ {50, 100, 200})

Each seed produces a different train/test split AND different generator samples,
giving variance across both the evaluation partition and the synthesis process.

Output:
  results/ci_hillstrom.csv        — per-seed rows
  results/ci_criteo.csv
  results/ci_great_german.csv
  results/ci_summary.csv          — mean ± 95% CI across seeds (paper-ready)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')

from experiments.synthetic_data_eval import (
    DATA_DIR, RESULTS_DIR,
    generate_ctgan, generate_gaussian_copula, generate_smote,
    train_evaluate
)
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

SEEDS  = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP  = 10_000   # same as headline experiments


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ci95(values):
    """Return (mean, half-width of 95% CI) for a list of values."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    h  = se * stats.t.ppf(0.975, df=len(arr) - 1)
    return float(np.mean(arr)), float(h)


def augment_and_eval(df_train, df_test, target, task, gen_name, alpha, seed):
    """Generate synthetic data and evaluate augmented model for one alpha/seed."""
    X_tr = df_train.drop(columns=[target])
    y_tr = df_train[target]
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values
    n_syn = int(len(df_train) * alpha)

    if gen_name == "SMOTE":
        df_syn = generate_smote(X_tr, y_tr, n_syn)
        X_aug = np.vstack([X_tr.values.astype(float),
                           df_syn.drop(columns=[target]).values.astype(float)])
        y_aug = np.concatenate([y_tr.values, df_syn[target].values])
    else:
        gen_fn = generate_gaussian_copula if gen_name == "GaussianCopula" else generate_ctgan
        df_syn = gen_fn(df_train, target, n_syn, task)
        X_syn  = df_syn.drop(columns=[target]).values.astype(float)
        y_syn  = df_syn[target].values
        X_aug  = np.vstack([X_tr.values.astype(float), X_syn])
        y_aug  = np.concatenate([y_tr.values, y_syn])

    # Pass seed to train_evaluate via a patched model
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_aug, y_aug)
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return {
        "auc_roc":       roc_auc_score(y_te, proba),
        "f1_minority":   f1_score(y_te, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_te, proba),
    }


def baseline_eval(df_train, df_test, target, seed):
    """Evaluate real-data-only baseline for one seed."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    X_tr = df_train.drop(columns=[target]).values.astype(float)
    y_tr = df_train[target].values
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return {
        "auc_roc":       roc_auc_score(y_te, proba),
        "f1_minority":   f1_score(y_te, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_te, proba),
    }


def run_ci_experiment(df_full, target, task, name, generators, best_alphas,
                      n_cap=N_CAP, out_csv=None):
    """
    Run multi-seed CI experiment for one dataset.
    generators: list of generator names to test
    best_alphas: dict {gen_name: alpha} — the headline alpha to focus on,
                 plus a full sweep for the top generator.
    """
    print(f"\n{'='*60}")
    print(f"CI experiment: {name}  ({len(SEEDS)} seeds)")
    print(f"{'='*60}")

    n_use = min(n_cap, len(df_full))
    all_rows = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        np.random.seed(seed)

        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[target]
        )

        # Baseline
        m = baseline_eval(df_train, df_test, target, seed)
        print(f"    Baseline: AUC={m['auc_roc']:.4f}")
        all_rows.append({"seed": seed, "method": "Baseline",
                         "condition": "real_only", "alpha": 0, **m})

        # Each generator at its headline alpha + full sweep
        for gen_name in generators:
            sweep_alphas = ALPHAS  # full sweep for all generators
            for alpha in sweep_alphas:
                try:
                    m = augment_and_eval(df_train, df_test, target, task,
                                         gen_name, alpha, seed)
                    print(f"    {gen_name} α={alpha}: AUC={m['auc_roc']:.4f}")
                    all_rows.append({"seed": seed, "method": gen_name,
                                     "condition": "augmented", "alpha": alpha, **m})
                except Exception as e:
                    print(f"    {gen_name} α={alpha} failed: {e}")

    df_results = pd.DataFrame(all_rows)
    if out_csv:
        df_results.to_csv(out_csv, index=False)
        print(f"\n  Saved raw results: {out_csv}")
    return df_results


def summarise_ci(df_results, name):
    """Aggregate per-seed results into mean ± 95% CI summary."""
    rows = []
    baseline_aucs = df_results[df_results["method"] == "Baseline"]["auc_roc"].values

    baseline_mean, baseline_h = ci95(baseline_aucs)
    rows.append({
        "dataset": name, "method": "Baseline", "alpha": 0,
        "auc_mean": round(baseline_mean, 4),
        "auc_ci95": round(baseline_h, 4),
        "auc_str":  f"{baseline_mean:.3f} ± {baseline_h:.3f}",
        "gain_mean": 0, "gain_ci95": 0,
    })

    for gen_name in df_results["method"].unique():
        if gen_name == "Baseline":
            continue
        aug = df_results[(df_results["method"] == gen_name) &
                         (df_results["condition"] == "augmented")]
        for alpha in sorted(aug["alpha"].unique()):
            vals = aug[aug["alpha"] == alpha]["auc_roc"].values
            if len(vals) == 0:
                continue
            mean_auc, h_auc = ci95(vals)
            # gain per seed vs that seed's baseline
            gains = []
            for seed in df_results["seed"].unique():
                aug_val = aug[(aug["alpha"] == alpha) &
                              (df_results.loc[aug.index, "seed"] == seed)]["auc_roc"]
                base_val = df_results[(df_results["seed"] == seed) &
                                      (df_results["method"] == "Baseline")]["auc_roc"]
                if len(aug_val) > 0 and len(base_val) > 0:
                    gains.append(float(aug_val.values[0]) - float(base_val.values[0]))
            mean_gain, h_gain = ci95(gains) if gains else (float("nan"), float("nan"))
            rows.append({
                "dataset": name, "method": gen_name, "alpha": alpha,
                "auc_mean": round(mean_auc, 4),
                "auc_ci95": round(h_auc, 4),
                "auc_str":  f"{mean_auc:.3f} ± {h_auc:.3f}",
                "gain_mean": round(mean_gain, 4),
                "gain_ci95": round(h_gain, 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hillstrom
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 1: Hillstrom Email Marketing")
print("="*60)

df_hill, hill_target, hill_task, hill_name = load_hillstrom()
df_hill_ci = run_ci_experiment(
    df_hill, hill_target, hill_task, hill_name,
    generators=["GaussianCopula", "CTGAN", "SMOTE"],
    best_alphas={"CTGAN": 0.3, "SMOTE": 0.3, "GaussianCopula": 0.1},
    out_csv=RESULTS_DIR / "ci_hillstrom.csv"
)
ci_hill = summarise_ci(df_hill_ci, hill_name)
print("\nHillstrom CI Summary (AUC mean ± 95% CI):")
print(ci_hill[ci_hill["alpha"].isin([0, 0.3])][
    ["method", "alpha", "auc_str", "gain_mean", "gain_ci95"]
].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Criteo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 2: Criteo Display Advertising")
print("="*60)

df_crit, crit_target, crit_task, crit_name = load_criteo_uplift()
df_crit_ci = run_ci_experiment(
    df_crit, crit_target, crit_task, crit_name,
    generators=["GaussianCopula", "CTGAN", "SMOTE"],
    best_alphas={"CTGAN": 0.5, "SMOTE": 0.3, "GaussianCopula": 0.2},
    out_csv=RESULTS_DIR / "ci_criteo.csv"
)
ci_crit = summarise_ci(df_crit_ci, crit_name)
print("\nCriteo CI Summary (AUC mean ± 95% CI):")
print(ci_crit[ci_crit["alpha"].isin([0, 0.5])][
    ["method", "alpha", "auc_str", "gain_mean", "gain_ci95"]
].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 3. GReaT vs GC vs CTGAN on German Credit at small-n
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 3: GReaT vs CTGAN vs GC (German Credit, small-n)")
print("="*60)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
try:
    from llmsynth import GReaT
    great_available = True
except ImportError:
    try:
        from be_great import GReaT
        great_available = True
    except ImportError:
        great_available = False
        print("  GReaT not available — skipping LLM column in CI table")

credit_path = DATA_DIR / "credit_default.csv"
if not credit_path.exists():
    raise FileNotFoundError(
        f"Expected {credit_path}. Run experiments/synthetic_data_eval.py first to "
        f"populate the data/ directory, or place credit_default.csv there manually."
    )
df_credit = pd.read_csv(credit_path)
credit_target = "target"
credit_task   = "classification"

small_ns   = [50, 100, 200]
great_rows = []

# Fixed holdout: always 300 rows, fixed seed
_, df_holdout = train_test_split(df_credit, test_size=300, random_state=42,
                                  stratify=df_credit[credit_target])
X_ho = df_holdout.drop(columns=[credit_target]).values.astype(float)
y_ho = df_holdout[credit_target].values

for n_train in small_ns:
    print(f"\n  n={n_train}:")
    for seed in SEEDS:
        np.random.seed(seed)
        df_s = df_credit.drop(df_holdout.index, errors="ignore")
        if len(df_s) < n_train:
            df_s = df_credit.sample(n_train + len(df_holdout),
                                    random_state=seed).iloc[len(df_holdout):]
        df_tr = df_s.sample(n_train, random_state=seed).reset_index(drop=True)
        X_tr  = df_tr.drop(columns=[credit_target]).values.astype(float)
        y_tr  = df_tr[credit_target].values

        # Baseline
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
        great_rows.append({"n": n_train, "seed": seed, "method": "Baseline",
                           "auc": base_auc})

        # GaussianCopula augmentation at α=1.0
        try:
            df_syn = generate_gaussian_copula(df_tr, credit_target, n_train, credit_task)
            X_aug  = np.vstack([X_tr, df_syn.drop(columns=[credit_target]).values.astype(float)])
            y_aug  = np.concatenate([y_tr, df_syn[credit_target].values])
            clf.set_params(random_state=seed)
            clf.fit(X_aug, y_aug)
            gc_auc = roc_auc_score(y_ho, clf.predict_proba(X_ho)[:, 1])
            great_rows.append({"n": n_train, "seed": seed, "method": "GaussianCopula", "auc": gc_auc})
            print(f"    seed={seed}: Baseline={base_auc:.4f}  GC={gc_auc:.4f}", end="")
        except Exception as e:
            print(f"    seed={seed}: GC failed: {e}", end="")

        # CTGAN augmentation at α=1.0
        try:
            df_syn = generate_ctgan(df_tr, credit_target, n_train, credit_task)
            X_aug  = np.vstack([X_tr, df_syn.drop(columns=[credit_target]).values.astype(float)])
            y_aug  = np.concatenate([y_tr, df_syn[credit_target].values])
            clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
            clf2.fit(X_aug, y_aug)
            ct_auc = roc_auc_score(y_ho, clf2.predict_proba(X_ho)[:, 1])
            great_rows.append({"n": n_train, "seed": seed, "method": "CTGAN", "auc": ct_auc})
            print(f"  CTGAN={ct_auc:.4f}", end="")
        except Exception as e:
            print(f"  CTGAN failed: {e}", end="")

        # GReaT augmentation at α=1.0
        if great_available:
            try:
                model = GReaT(llm="distilgpt2", batch_size=32, epochs=50)
                model.fit(df_tr)
                df_syn = model.sample(n_train)
                df_syn.columns = df_tr.columns
                df_syn[credit_target] = df_syn[credit_target].astype(int)
                X_aug  = np.vstack([X_tr, df_syn.drop(columns=[credit_target]).values.astype(float)])
                y_aug  = np.concatenate([y_tr, df_syn[credit_target].values])
                clf3 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
                clf3.fit(X_aug, y_aug)
                gr_auc = roc_auc_score(y_ho, clf3.predict_proba(X_ho)[:, 1])
                great_rows.append({"n": n_train, "seed": seed, "method": "GReaT", "auc": gr_auc})
                print(f"  GReaT={gr_auc:.4f}")
            except Exception as e:
                print(f"  GReaT failed: {e}")
        else:
            print()

df_great_ci = pd.DataFrame(great_rows)
df_great_ci.to_csv(RESULTS_DIR / "ci_great_german.csv", index=False)

# Summarise GReaT CI
print("\nGReaT CI Summary (AUC mean ± 95% CI across 5 seeds):")
print(f"{'n':>6}  {'Method':<16}  {'Mean AUC':>10}  {'± 95% CI':>10}  {'vs Baseline':>12}")
print("-" * 60)
for n in small_ns:
    sub = df_great_ci[df_great_ci["n"] == n]
    base_mean, _ = ci95(sub[sub["method"] == "Baseline"]["auc"].values)
    for meth in ["Baseline", "GaussianCopula", "CTGAN", "GReaT"]:
        vals = sub[sub["method"] == meth]["auc"].values
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        gain = m - base_mean if meth != "Baseline" else 0
        print(f"{n:>6}  {meth:<16}  {m:>10.4f}  {h:>10.4f}  {gain:>+12.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Combined summary CSV
# ─────────────────────────────────────────────────────────────────────────────
ci_summary = pd.concat([ci_hill, ci_crit], ignore_index=True)
ci_summary.to_csv(RESULTS_DIR / "ci_summary.csv", index=False)
print(f"\nSaved combined CI summary: {RESULTS_DIR}/ci_summary.csv")
print("\nDone.")
