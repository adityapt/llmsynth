# LLMSynth

Empirical evaluation of synthetic data generation methods for marketing and product data science, with a focus on LLM-based generators (GReaT).

All key findings are backed by **5-seed cross-validation** (seeds 42, 123, 7, 2024, 999) with 95% confidence intervals via t-distribution.

## What's Here

**Paper:** `papers/synthetic-data-marketing-eval.md`
*LLM-Enhanced Synthetic Data Generation for Marketing and Product Data Science: A Practical Evaluation Under Data Scarcity and Feature Sparsity*

**Experiments:** `experiments/`
- `synthetic_data_eval.py` — main experiment runner (Telco Churn, Bank Marketing, German Credit, Online Retail CLV)
- `run_remaining.py` — runs Credit Default + CLV datasets using cached results from first two
- `run_nomao.py` — Nomao lead dataset (full, n=10K)
- `run_nomao_sparse.py` — Nomao with 70% simulated missingness + small n=500
- `run_kdd_appetency.py` — KDD Cup 2009 Appetency (natural CRM sparsity, 70% missing)
- `run_hillstrom.py` — Hillstrom Email Marketing (GaussianCopula/CTGAN/SMOTE, real campaign data, 0.9% conversion)
- `run_criteo.py` — Criteo Uplift Display Advertising (real ad data, 0.2% conversion)
- `run_great_databricks.py` — GReaT (GPT-2) on German Credit, designed for Databricks GPU cluster
- `run_great_hillstrom_databricks.py` — GReaT (GPT-2) on Hillstrom, designed for Databricks GPU cluster
- `run_great_kaggle.py` — GReaT (GPT-2) on German Credit, alternative runner for Kaggle GPU notebooks
- `run_great_colab.py` — GReaT (distilgpt2) experiment, original Colab attempt (sampling failure documented)

**Results:** `results/`
- `metrics_*.csv` — AUC/F1/AP per dataset × generator × condition
- `ci_*.csv` — 5-seed confidence interval results per dataset
- `ci_great_german.csv` — GReaT distilgpt2 on German Credit (sampling failure; GReaT = Baseline throughout)
- `ci_great_hillstrom.csv` — GReaT GPT-2 on Hillstrom, n ∈ {50,100,200,500,1000,2000}, holdout=10K
- `ucurve_*.png` — U-shaped augmentation curves
- `lowdata_*.png` — performance vs. training set size
- `summary_table.csv` — cross-dataset summary including GReaT results

## Datasets

| Dataset | n | Task | Positive Rate | Source |
|---|---|---|---|---|
| Telco Churn | 7,032 | Classification | 26.6% | IBM Kaggle |
| Bank Marketing | 15,000 | Classification | 11.7% | UCI / OpenML |
| German Credit | 1,000 | Classification | 30.0% | OpenML id=31 |
| Online Retail CLV | 4,000 | Regression | — | UCI (fallback) |
| Nomao Lead | 10,000 | Classification | 28.3% | OpenML id=1486 |
| KDD Appetency | 5,000 | Classification | 6.7% | OpenML id=1112 |
| Hillstrom Email | 64,000 | Classification | 0.9% | MineThatData (2008) |
| Criteo Uplift | 13.9M (cap 10K) | Classification | 0.2% | Criteo AI Lab (2018) |

## Methods

### Statistical Synthesizers
- **GaussianCopula** — models multivariate dependencies via copula; fast, works well on smaller datasets. SDV v1.36
- **CTGAN** — conditional GAN for tabular data; handles mixed types and imbalance. SDV/CTGAN v0.12
- **SMOTE** — interpolation-based oversampling for minority class only. imbalanced-learn v0.12

### LLM-Based Synthesizer
- **GReaT** (Generate Realistic Tabular data) — fine-tunes a GPT-2 language model on rows serialized as natural language text (e.g. `"recency is 6, history is 230.0, target is 1"`), then samples new rows by prompting the model. be-great v0.0.13

  **GPU required.** On CPU, `model.sample()` fails regardless of model size. With GPU + `guided_sampling=True` + `experiment_dir="/tmp/..."` (not Workspace path — PyTorch binaries corrupt), sampling succeeds.

  **How to run GReaT on Databricks:**
  1. Create a GPU cluster (e.g. g4dn.xlarge, single node)
  2. Upload the data CSV to `/Workspace/Users/<you>/Temp/`
  3. Paste `run_great_databricks.py` or `run_great_hillstrom_databricks.py` into a notebook cell and run

## Key Findings

Results are 5-seed mean ± 95% CI unless noted. Gains are AUC-ROC points (absolute).

### Statistical synthesizers (GaussianCopula, CTGAN, SMOTE)

| Dataset | Positive Rate | Best Method | Gain (5-seed CI) | Verdict |
|---|---|---|---|---|
| Telco Churn | 26.6% | GaussianCopula | +0.28 pts | Skip it |
| Bank Marketing | 11.7% | CTGAN | −0.17 pts | Skip it |
| German Credit | 30.0% | CTGAN | +0.81 pts | Marginal |
| Nomao Lead | 28.3% | GaussianCopula | −0.06 pts | Skip it |
| Nomao Sparse (70% missing) | 28.3% | CTGAN | +0.5 pts (within noise) | No — sparsity swamps signal |
| Hillstrom Email | 0.9% | SMOTE | +6.98 pts | **Strong yes** |
| Criteo Display Ads | 0.2% | CTGAN | +13.23 pts | **Strong yes** |

### GReaT (LLM-based, GPT-2, GPU)

**Experiment 1 — German Credit (anonymized features `f0–f19`, 30% positive):**

| n | Baseline (mean ± CI) | GReaT (mean ± CI) | Gain |
|---|---|---|---|
| 50 | 0.6446 ± 0.1153 | 0.6375 ± 0.0765 | −0.7 pts |
| 100 | 0.7079 ± 0.0419 | 0.6377 ± 0.0333 | −7.0 pts |
| 200 | 0.7587 ± 0.0284 | 0.6995 ± 0.0519 | −5.9 pts |
| 500 | 0.7607 ± 0.0190 | 0.7307 ± 0.0213 | −3.0 pts |

**Finding:** Consistent, monotonic negative gains. Anonymized feature names (`f0–f19`) remove all LLM prior value — GPT-2 has no pretraining knowledge about features named `f0`.

**Experiment 2 — Hillstrom (semantic features, 0.9% positive, holdout=10K):**

| n | Baseline (mean ± CI) | GReaT (mean ± CI) | Gain | Win rate |
|---|---|---|---|---|
| 50 | 0.4937 ± 0.0058 | 0.5162 ± 0.0469 | **+2.3 pts** | 4/5 seeds |
| 100 | 0.4884 ± 0.0293 | 0.4999 ± 0.0525 | +1.2 pts | 3/5 seeds |
| 200 | 0.4928 ± 0.0272 | 0.4968 ± 0.0908 | +0.4 pts | 3/5 seeds |
| 500 | 0.5124 ± 0.0692 | 0.5122 ± 0.0788 | ~0 pts | 3/5 seeds |
| 1000 | 0.5160 ± 0.0698 | 0.4763 ± 0.0710 | −4.0 pts | 1/5 seeds |
| 2000 | 0.5345 ± 0.0596 | 0.4658 ± 0.0461 | **−6.9 pts** | 0/5 seeds |

**Finding:** Semantic features produce a marginal positive signal at extreme small-n (n=50, 4/5 seeds), but gains decay monotonically and become significantly harmful at n=2000. The 0.9% positive rate confounds the semantic benefit — GReaT samples mostly negative-class rows, diluting minority-class signal as n grows.

**Open experiment:** Telco Churn (semantic features + 26.6% positive rate) would cleanly separate the semantic-feature and class-imbalance effects. If GReaT helps there, the semantic hypothesis is confirmed. See `run_great_databricks.py` as a template.

### Core conclusion

**The only reliable driver of statistical synthesizer gain is extreme class imbalance (positive rate < ~5%).** For LLM-based synthesis, semantic feature names provide marginal benefit at extreme small-n, but class imbalance and large n actively undermine it. GReaT is not recommended for production augmentation without GPU access and careful validation of `model.sample()` output.

## Setup

```bash
pip install sdv imbalanced-learn scikit-learn matplotlib pandas numpy openpyxl
python experiments/synthetic_data_eval.py
```

For GReaT (LLM-based, GPU required):
```bash
pip install be_great
# Use experiments/run_great_databricks.py on a Databricks GPU cluster
# Or experiments/run_great_kaggle.py on a Kaggle GPU notebook
```
