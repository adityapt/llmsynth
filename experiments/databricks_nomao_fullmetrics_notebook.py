# Databricks notebook source
# Self-contained Nomao full-metrics rerun -- see databricks_nomao_fullmetrics.py
# for the non-notebook version and full docstring. This notebook variant exists
# because the workspace's public DBFS root is disabled, so output is returned
# via dbutils.notebook.exit() (retrieved through the Jobs API) instead of a
# shared filesystem path.
# COMMAND ----------
# MAGIC %pip install sdv==1.36.0 imbalanced-learn==0.12.4
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)
from imblearn.over_sampling import SMOTE
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SEEDS = [42, 123, 7, 2024, 999]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_CAP = 10_000


def load_nomao():
    from sklearn.datasets import fetch_openml
    print("Downloading Nomao from OpenML (id=1486)...", flush=True)
    ds = fetch_openml(data_id=1486, as_frame=True, parser="auto")
    df = ds.frame.copy()
    target_col = ds.target_names[0] if ds.target_names else df.columns[-1]
    df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(str).str.strip()
    vc = df["target"].value_counts()
    minority = vc.idxmin()
    df["target"] = (df["target"] == minority).astype(int)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    print(f"Shape: {df.shape}, positive rate: {df['target'].mean()*100:.1f}%", flush=True)
    return df


def generate_smote(X_train, y_train, n_synthetic):
    minority_count = y_train.sum()
    majority_count = len(y_train) - minority_count
    target_minority = int(minority_count + n_synthetic)
    sm = SMOTE(sampling_strategy={1: target_minority}, random_state=RANDOM_STATE,
               k_neighbors=min(5, minority_count - 1))
    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_syn = X_res[len(X_train):]
    y_syn = y_res[len(y_train):]
    df_syn = pd.DataFrame(X_syn, columns=X_train.columns)
    df_syn[y_train.name] = y_syn
    return df_syn


def generate_ctgan(df_train, target, n_synthetic):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    metadata.update_column(target, sdtype="categorical")
    synth = CTGANSynthesizer(metadata, epochs=150, batch_size=min(500, len(df_train)), verbose=False)
    synth.fit(df_train)
    df_syn = synth.sample(num_rows=n_synthetic)
    df_syn[target] = df_syn[target].astype(int)
    return df_syn


def generate_gaussian_copula(df_train, target, n_synthetic):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    metadata.update_column(target, sdtype="categorical")
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df_train)
    df_syn = synth.sample(num_rows=n_synthetic)
    df_syn[target] = df_syn[target].astype(int)
    return df_syn


def full_metrics(y_true, proba, preds):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "avg_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
    }


import time

dbutils.widgets.text("seed", "42")
dbutils.widgets.text("max_seconds", "2400")
dbutils.widgets.text("skip_combos", "")  # comma-separated "method:alpha" already done, e.g. "CTGAN:0.1,SMOTE:0.5"

RUN_SEEDS = [int(dbutils.widgets.get("seed"))]
MAX_SECONDS = int(dbutils.widgets.get("max_seconds"))
SKIP = set(s for s in dbutils.widgets.get("skip_combos").split(",") if s)
START_TIME = time.time()

df_full = load_nomao()
target = "target"
n_use = min(N_CAP, len(df_full))
rows = []
timed_out = False

for seed in RUN_SEEDS:
    df = df_full.sample(n_use, random_state=seed).reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
    X_tr_df = df_train.drop(columns=[target])
    y_tr = df_train[target]
    X_te = df_test.drop(columns=[target]).values.astype(float)
    y_te = df_test[target].values
    X_tr = X_tr_df.values.astype(float)

    if "Baseline:0" not in SKIP:
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
        clf.fit(X_tr, y_tr.values)
        m = full_metrics(y_te, clf.predict_proba(X_te)[:, 1], clf.predict(X_te))
        print(f"seed={seed} Baseline: AUC={m['auc_roc']:.4f}  [elapsed={time.time()-START_TIME:.0f}s]", flush=True)
        rows.append({"seed": seed, "method": "Baseline", "alpha": 0, **m})
        del clf
        gc.collect()

    for gen_name in ["GaussianCopula", "CTGAN", "SMOTE"]:
        for alpha in ALPHAS:
            combo_key = f"{gen_name}:{alpha}"
            if combo_key in SKIP:
                continue
            if time.time() - START_TIME > MAX_SECONDS:
                print(f"Soft time limit ({MAX_SECONDS}s) reached before {combo_key} -- stopping early with partial results.", flush=True)
                timed_out = True
                break
            try:
                n_syn = int(len(df_train) * alpha)
                if gen_name == "SMOTE":
                    df_syn = generate_smote(X_tr_df, y_tr, n_syn)
                elif gen_name == "GaussianCopula":
                    df_syn = generate_gaussian_copula(df_train, target, n_syn)
                else:
                    df_syn = generate_ctgan(df_train, target, n_syn)
                X_aug = np.vstack([X_tr, df_syn.drop(columns=[target]).values.astype(float)])
                y_aug = np.concatenate([y_tr.values, df_syn[target].values])
                clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
                clf2.fit(X_aug, y_aug)
                m = full_metrics(y_te, clf2.predict_proba(X_te)[:, 1], clf2.predict(X_te))
                print(f"seed={seed} {gen_name} alpha={alpha}: AUC={m['auc_roc']:.4f}  [elapsed={time.time()-START_TIME:.0f}s]", flush=True)
                rows.append({"seed": seed, "method": gen_name, "alpha": alpha, **m})
                del clf2, df_syn, X_aug, y_aug
            except Exception as e:
                print(f"seed={seed} {gen_name} alpha={alpha} FAILED: {e}", flush=True)
            gc.collect()
        if timed_out:
            break
    if timed_out:
        break

print(f"\nDone. {len(rows)} rows. timed_out={timed_out}", flush=True)
out_df = pd.DataFrame(rows)
csv_text = out_df.to_csv(index=False)
print(csv_text)
# COMMAND ----------
dbutils.notebook.exit(csv_text)
