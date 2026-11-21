"""
CPU only, no GPU needed.

Extends §6.8 statistical-generator CI results in two dimensions:
  1. SEEDS: 5 -> 10  (existing [42,123,7,2024,999] + new [10,20,30,40,50])
  2. CLASSIFIERS: 1 -> 4  (existing GBC + Logistic Regression, Random Forest,
     Multi-Layer Perceptron) for downstream robustness

Generators: GaussianCopula, CTGAN, SMOTE (matches run_confidence_intervals.py).
Datasets:   Hillstrom, Criteo.

Design parity with run_confidence_intervals.py §6.8 CI protocol:
  - N_CAP = 10_000, per-seed cap sample with random_state=seed
  - 80/20 stratified train/test split with random_state=seed
  - Same ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
  - The Baseline rows under classifier='GBC' should match
    results/ci_hillstrom.csv / ci_criteo.csv Baseline rows exactly for the
    first 5 seeds — a free cross-check before trusting new-seed/new-classifier rows.

Generator fit strategy:
  - GaussianCopula, CTGAN: fit ONCE per (dataset, seed, generator) at the
    largest alpha (n_syn_max = int(n_train * max(ALPHAS))), then sub-sample
    syn_max.head(int(n_train * alpha)) for each smaller alpha.
    This is the same α-sweep pattern used by run_tabddpm_databricks.py and
    the GReaT α-sweep scripts. It is NOT what run_confidence_intervals.py
    does (which re-fits per alpha), so per-alpha synthetic data differs from
    ci_hillstrom.csv even at the original 5 seeds — but the Baseline rows
    and the protocol around the generator are identical.
  - SMOTE: re-call per alpha (SMOTE has no separate fit step; calling it is
    cheap and matches the existing protocol).

Output schema (single CSV per dataset):
  seed, method, condition, alpha, classifier,
  auc_roc, f1_minority, avg_precision, precision, recall, accuracy

Output paths:
  results/ci_multi_classifier_hillstrom.csv
  results/ci_multi_classifier_criteo.csv

Resume support: a (seed, method, classifier) triple counts as done only when
its α=1.0 row exists with a non-NaN auc_roc. Failed runs (NaN-filled) are
retried on resume.

Runtime estimate on a modern multi-core CPU:
  - CTGAN: ~5-15 min per fit; 10 seeds × 2 datasets = 20 fits = ~2-5 hrs
  - GaussianCopula: ~30 sec per fit; 20 fits = ~10 min
  - SMOTE: ~instant
  - Classifier evals: dominated by MLP (~10-60 sec per fit); ~1-3 hrs total
  - Total: ~4-10 hrs. Overnight.
"""
import warnings, os, random, sys
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, ".")
from experiments.synthetic_data_eval import (
    generate_ctgan, generate_gaussian_copula, generate_smote,
)
from experiments.run_hillstrom import load_hillstrom
from experiments.run_criteo import load_criteo_uplift

# ── Config ────────────────────────────────────────────────────────────────
SEEDS_EXISTING = [42, 123, 7, 2024, 999]
SEEDS_NEW      = [10, 20, 30, 40, 50]
SEEDS          = SEEDS_EXISTING + SEEDS_NEW
ALPHAS         = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP          = 10_000
GENERATORS     = ["GaussianCopula", "CTGAN", "SMOTE"]
CLASSIFIERS    = ["GBC", "LR", "RF", "MLP"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ci95(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) < 2:
        return float(np.mean(arr)), float("nan")
    se = stats.sem(arr)
    return float(np.mean(arr)), float(se * stats.t.ppf(0.975, df=len(arr) - 1))


def build_classifier(name: str, seed: int):
    """Return a fresh fitted-ready estimator for `name`, seeded with `seed`.

    LR and MLP are wrapped in a StandardScaler pipeline (these classifiers
    are scale-sensitive). RF and GBC use the raw features.
    """
    if name == "GBC":
        return GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                          random_state=seed)
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=seed)),
        ])
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      n_jobs=-1, random_state=seed)
    if name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                  max_iter=200,
                                  early_stopping=True,
                                  validation_fraction=0.1,
                                  random_state=seed)),
        ])
    raise ValueError(f"Unknown classifier: {name}")


def evaluate(X_tr, y_tr, X_te, y_te, classifier_name, seed):
    clf = build_classifier(classifier_name, seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return {
        "auc_roc":       roc_auc_score(y_te, proba),
        "f1_minority":   f1_score(y_te, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_te, proba),
        "precision":     precision_score(y_te, preds, pos_label=1, zero_division=0),
        "recall":        recall_score(y_te, preds, pos_label=1, zero_division=0),
        "accuracy":      accuracy_score(y_te, preds),
    }


# ── Resume support ────────────────────────────────────────────────────────

def load_resume(out_path):
    """Return (existing rows, done set of (seed, method, classifier)).

    A triple is 'done' only when its α=1.0 row exists with a non-NaN auc_roc;
    rows of unfinished triples are dropped and regenerated.
    Baseline rows are marked done by (seed, 'Baseline', classifier) regardless
    of alpha (they have alpha=0).
    """
    if not os.path.exists(out_path):
        return [], set()
    rows = pd.read_csv(out_path).to_dict("records")
    done = set()
    for r in rows:
        if r["method"] == "Baseline":
            if pd.notna(r.get("auc_roc")):
                done.add((int(r["seed"]), "Baseline", r["classifier"]))
        else:
            if (r["alpha"] == ALPHAS[-1]) and pd.notna(r.get("auc_roc")):
                done.add((int(r["seed"]), r["method"], r["classifier"]))
    # Drop rows of triples that did not finish, so that rerunning them does not leave duplicate rows.
    rows = [r for r in rows if (int(r["seed"]), r["method"], r["classifier"]) in done]
    return rows, done


# ── Main loop ─────────────────────────────────────────────────────────────

def run_dataset(loader_fn):
    df_full, target, task, name = loader_fn()
    print(f"\n{'='*68}\n{name}: {df_full.shape}, positive rate {df_full[target].mean()*100:.2f}%\n{'='*68}",
          flush=True)

    out_path = RESULTS_DIR / f"ci_multi_classifier_{name.lower().split()[0]}.csv"
    rows, done = load_resume(out_path)
    print(f"Resume: {len(done)} (seed, method, classifier) triples already complete",
          flush=True)

    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        seed_everything(seed)

        n_use = min(N_CAP, len(df_full))
        df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[target]
        )
        X_tr_real = df_train.drop(columns=[target]).values.astype(float)
        y_tr_real = df_train[target].values
        X_te      = df_test.drop(columns=[target]).values.astype(float)
        y_te      = df_test[target].values

        # ── Baseline (real-only) under every classifier ───────────────────
        for clf_name in CLASSIFIERS:
            if (seed, "Baseline", clf_name) in done:
                continue
            try:
                m = evaluate(X_tr_real, y_tr_real, X_te, y_te, clf_name, seed)
                rows.append({"seed": seed, "method": "Baseline",
                             "condition": "real_only", "alpha": 0,
                             "classifier": clf_name, **m})
                print(f"    Baseline · {clf_name}: AUC={m['auc_roc']:.4f}", flush=True)
            except Exception as e:
                rows.append({"seed": seed, "method": "Baseline",
                             "condition": "real_only", "alpha": 0,
                             "classifier": clf_name,
                             "auc_roc": float("nan"),
                             "f1_minority": float("nan"),
                             "avg_precision": float("nan"),
                             "precision": float("nan"),
                             "recall": float("nan"),
                             "accuracy": float("nan")})
                print(f"    Baseline · {clf_name} FAILED: {str(e)[:140]}", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

        # ── Each generator: fit once at max α, sub-sample for smaller α  ──
        # SMOTE is the exception — we re-call per alpha because SMOTE has no
        # separate fit/sample split and re-calling is essentially free.
        for gen_name in GENERATORS:
            # Skip this generator if every (alpha, classifier) is already done
            all_done = all(
                (seed, gen_name, c) in done for c in CLASSIFIERS
            )
            if all_done:
                print(f"    {gen_name}: all classifiers done for this seed", flush=True)
                continue

            try:
                if gen_name == "SMOTE":
                    # SMOTE: per-alpha re-call (cheap, no real fit step)
                    syn_per_alpha = {}
                    X_tr_pd = df_train.drop(columns=[target])
                    y_tr_pd = df_train[target]
                    for alpha in ALPHAS:
                        n_syn = int(len(df_train) * alpha)
                        if n_syn == 0:
                            continue
                        df_syn = generate_smote(X_tr_pd, y_tr_pd, n_syn)
                        syn_per_alpha[alpha] = df_syn
                else:
                    # Fit-once strategy: generate at max alpha, then subsample
                    n_syn_max = int(len(df_train) * max(ALPHAS))
                    gen_fn = (generate_gaussian_copula
                              if gen_name == "GaussianCopula" else generate_ctgan)
                    print(f"    [{gen_name}] fitting...", end="", flush=True)
                    df_syn_max = gen_fn(df_train, target, n_syn_max, task)
                    print(f"  ({len(df_syn_max)} synth rows)", flush=True)
                    syn_per_alpha = {}
                    for alpha in ALPHAS:
                        n_syn = int(len(df_train) * alpha)
                        if n_syn == 0:
                            continue
                        syn_per_alpha[alpha] = df_syn_max.head(n_syn).copy()
            except Exception as e:
                err = str(e)[:140]
                for alpha in ALPHAS:
                    for clf_name in CLASSIFIERS:
                        if (seed, gen_name, clf_name) in done:
                            continue
                        rows.append({"seed": seed, "method": gen_name,
                                     "condition": "augmented", "alpha": alpha,
                                     "classifier": clf_name,
                                     "auc_roc": float("nan"),
                                     "f1_minority": float("nan"),
                                     "avg_precision": float("nan"),
                                     "precision": float("nan"),
                                     "recall": float("nan"),
                                     "accuracy": float("nan")})
                print(f"    {gen_name} FAILED: {err}", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                continue

            # Evaluate every (alpha, classifier) combo
            for alpha in ALPHAS:
                if alpha not in syn_per_alpha:
                    continue
                df_syn = syn_per_alpha[alpha]
                X_syn = df_syn.drop(columns=[target]).values.astype(float)
                y_syn = df_syn[target].values
                X_aug = np.vstack([X_tr_real, X_syn])
                y_aug = np.concatenate([y_tr_real, y_syn])

                for clf_name in CLASSIFIERS:
                    if (seed, gen_name, clf_name) in done and alpha != ALPHAS[-1]:
                        # done-marker is α=1.0 only, but skip lower alphas too
                        # if a non-NaN α=1.0 row exists
                        continue
                    try:
                        m = evaluate(X_aug, y_aug, X_te, y_te, clf_name, seed)
                        rows.append({"seed": seed, "method": gen_name,
                                     "condition": "augmented", "alpha": alpha,
                                     "classifier": clf_name, **m})
                        print(f"    {gen_name} α={alpha} · {clf_name}: AUC={m['auc_roc']:.4f}",
                              flush=True)
                    except Exception as e:
                        rows.append({"seed": seed, "method": gen_name,
                                     "condition": "augmented", "alpha": alpha,
                                     "classifier": clf_name,
                                     "auc_roc": float("nan"),
                                     "f1_minority": float("nan"),
                                     "avg_precision": float("nan"),
                                     "precision": float("nan"),
                                     "recall": float("nan"),
                                     "accuracy": float("nan")})
                        print(f"    {gen_name} α={alpha} · {clf_name} FAILED: {str(e)[:120]}",
                              flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  {name} — multi-classifier CI summary (mean ± 95% CI, GBC only):",
          flush=True)
    df_out = pd.DataFrame(rows)
    for clf_name in CLASSIFIERS:
        sub = df_out[df_out["classifier"] == clf_name]
        if sub.empty:
            continue
        base_mean, _ = ci95(sub[sub["method"] == "Baseline"]["auc_roc"].values)
        print(f"\n    Classifier = {clf_name}", flush=True)
        print(f"    {'method':>16} {'alpha':>6} {'AUC':>16} {'gain':>8}", flush=True)
        aug = sub[sub["method"] != "Baseline"]
        for gen in GENERATORS:
            for alpha in ALPHAS:
                vals = aug[(aug["method"] == gen) & (aug["alpha"] == alpha)]["auc_roc"].values
                if len(vals) == 0:
                    continue
                m, h = ci95(vals)
                print(f"    {gen:>16} {alpha:>6} {m:>7.4f} ± {h:<6.4f} {m - base_mean:>+8.4f}",
                      flush=True)

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for loader_fn in [load_hillstrom, load_criteo_uplift]:
        out_path = run_dataset(loader_fn)
        print(f"\nSaved: {out_path}", flush=True)
    print("\nDone.", flush=True)
