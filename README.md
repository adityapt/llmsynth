# LLMSynth

Empirical evaluation of synthetic data generation methods for marketing and product data science, with a focus on LLM-based generators (GReaT) and diffusion models (TabDDPM).

Core CI experiments use **5-seed cross-validation** (seeds 42, 123, 7, 2024, 999) with 95% confidence intervals via t-distribution. Multi-classifier robustness experiments use **10 seeds** (+ 10, 20, 30, 40, 50).

## What's Here

**Paper:** `papers/synthetic-data-marketing-eval.md`
*Statistical and LLM-Based Synthetic Data Generation for Marketing and Product Data Science — A Controlled Empirical Evaluation Across Data Scarcity, Class Imbalance, and Feature Sparsity*

**Provenance:** `papers/synthetic-data-marketing-eval.provenance.md` — source accounting, verification status, and method-coverage notes.

**Experiments:** `experiments/`
- `synthetic_data_eval.py` — main experiment runner (Telco Churn, Bank Marketing, German Credit, Online Retail CLV)
- `run_remaining.py` — runs German Credit + CLV datasets using cached results from first two (the German Credit loader is named `load_credit_default` for historical reasons; the actual dataset is Statlog German Credit, OpenML id=31)
- `run_nomao.py` — Nomao lead dataset (full, n=10K)
- `run_nomao_sparse.py` — Nomao with 70% simulated missingness + small n=500
- `run_kdd_appetency.py` — KDD Cup 2009 Appetency (natural CRM sparsity, 70% missing) *(script only; not used in the paper — no result files generated)*
- `run_hillstrom.py` — Hillstrom Email Marketing (GaussianCopula/CTGAN/SMOTE, real campaign data, 0.9% conversion)
- `run_criteo.py` — Criteo Uplift Display Advertising (real ad data, 0.2% conversion)
- `run_confidence_intervals.py` — 5-seed CI protocol for Hillstrom + Criteo (GBC downstream, §6.8 core results)
- `run_ci_multi_classifier.py` — Extended 10-seed CI with 4 downstream classifiers (GBC, LR, RF, MLP); classifier-robustness check for §6.8 findings
- `run_great_databricks.py` — GReaT (GPT-2) on German Credit, designed for Databricks GPU cluster
- `run_great_hillstrom_databricks.py` — GReaT (GPT-2) on Hillstrom, designed for Databricks GPU cluster
- `run_great_telco_databricks.py` — GReaT (GPT-2) on Telco Churn, isolates semantic-features effect from class-imbalance effect (semantic features + 26.6% positive rate); inline data prep handles the raw IBM CSV
- `run_great_kaggle.py` — GReaT (GPT-2) on German Credit, alternative runner for Kaggle GPU notebooks
- `run_great_colab.py` — GReaT (distilgpt2) experiment, original Colab attempt (sampling failure documented)
- `run_tabddpm_databricks.py` — TabDDPM (synthcity `ddpm` plugin) on Hillstrom + Criteo, GPU/Databricks; 5-seed CI with GBC downstream

**Results:** `results/`
- `metrics_*.csv` — AUC/F1/AP per dataset × generator × condition
- `ci_*.csv` — 5-seed confidence interval results per dataset (GBC)
- `ci_multi_classifier_{hillstrom,criteo}.csv` — 10-seed CI across 4 classifiers (GBC/LR/RF/MLP)
- `ci_tabddpm_{hillstrom,criteo}.csv` — 5-seed TabDDPM CI results (GPU run, GBC)
- `ci_great_german.csv` — Baseline / GaussianCopula / CTGAN on German Credit, n ∈ {50,100,200,500,700}, 300-row holdout (statistical generators only)
- `ci_great_hillstrom.csv` — GReaT GPT-2 on Hillstrom, n ∈ {50,100,200,500,1000,2000}, holdout=10K
- `great_german_results.csv` — GReaT GPT-2 vs Baseline on German Credit, n ∈ {50,100,200,500}, holdout=200 (GPU run, 5-seed CI)
- `great_telco_results.csv` — GReaT GPT-2 vs Baseline on Telco Churn, n ∈ {50,100,200,500,1000,2000}, holdout=2000 (GPU run, 5-seed CI)
- `great_alpha_sweep_{german,telco,hillstrom}_results.csv` — Phase 5 α-sweep at small-n shoulder (n ∈ {50,100,200}, α ∈ {0.1,0.2,0.3,0.5,1.0}, 5 seeds, matched-design)
- `alpha_sweep_rigorous_analysis.csv` — per-cell paired stats (Δ, CI, d_z, raw p, BH-FDR p over 45-test family) for the α-sweep
- `summary_table.csv` — cross-dataset summary including GReaT results

## Datasets

| Dataset | n | Task | Positive Rate | Source |
|---|---|---|---|---|
| Telco Churn | 7,032 | Classification | 26.6% | IBM Kaggle |
| Bank Marketing | 15,000 | Classification | 11.7% | UCI / OpenML |
| German Credit | 1,000 | Classification | 30.0% | OpenML id=31 |
| Online Retail CLV | 4,000 | Regression | — | UCI (fallback) |
| Nomao Lead (full) | 10,000 | Classification | 28.3% | OpenML id=1486 |
| Nomao Lead (sparse, 70% missing) | 500 | Classification | 28.3% | OpenML id=1486 |
| Hillstrom Email | 64,000 (cap 10K) | Classification | 0.9% | MineThatData (2008) |
| Criteo Uplift | 13.9M (cap 10K) | Classification | 0.2% | Criteo AI Lab (2018) |

## Methods

### Statistical Synthesizers
- **GaussianCopula** — models multivariate dependencies via copula; fast, works well on smaller datasets. SDV v1.36
- **CTGAN** — conditional GAN for tabular data; handles mixed types and imbalance. SDV/CTGAN v0.12
- **SMOTE** — interpolation-based oversampling for minority class only. imbalanced-learn v0.12

### Diffusion Model
- **TabDDPM** — denoising diffusion probabilistic model for tabular data (Kotelnikov et al., ICML 2023). Evaluated via synthcity's `ddpm` plugin (v0.2.11) on Hillstrom and Criteo. GPU required (Databricks). Fit-once at α=1.0, subsampled for smaller α.

### LLM-Based Synthesizer
- **GReaT** (Generate Realistic Tabular data) — fine-tunes a GPT-2 language model on rows serialized as natural language text (e.g. `"recency is 6, history is 230.0, target is 1"`), then samples new rows by prompting the model. be-great v0.0.13

  **GPU required.** On CPU, `model.sample()` fails regardless of model size. With GPU + `guided_sampling=True` + `experiment_dir="/tmp/..."` (not Workspace path — PyTorch binaries corrupt), sampling succeeds.

  **How to run GReaT on Databricks:**
  1. Create a GPU cluster (e.g. g4dn.xlarge, single node)
  2. Upload the data CSV to `/Workspace/Users/<you>/Temp/`
  3. Paste `run_great_databricks.py` or `run_great_hillstrom_databricks.py` into a notebook cell and run

## Key Findings

Results are 5-seed mean ± 95% CI unless noted. Gains are AUC-ROC points (absolute).

### Statistical synthesizers + TabDDPM

| Dataset | Positive Rate | CTGAN | SMOTE | TabDDPM | Verdict |
|---|---|---|---|---|---|
| Telco Churn | 26.6% | +0.28 pts | — | — | Skip it |
| Bank Marketing | 11.7% | −0.17 pts | — | — | Skip it |
| German Credit | 30.0% | +0.81 pts | — | — | Marginal |
| Nomao Lead | 28.3% | −0.06 pts | — | — | Skip it |
| Nomao Sparse (70% missing) | 28.3% | +0.5 pts (noise) | — | — | No |
| **Hillstrom Email** | **0.9%** | **+5.7 pts** | **+5.8 pts** | **+1.4 pts** | **Strong yes (CTGAN/SMOTE); TabDDPM marginal** |
| **Criteo Display Ads** | **0.2%** | **+12.9 pts** | **+12.0 pts** | **+9.9 pts** | **Strong yes; CTGAN > TabDDPM > SMOTE** |

**Multi-classifier robustness (10 seeds, Hillstrom + Criteo):** CTGAN's Criteo advantage holds across GBC (+12.0 pts) and RF (+9.6 pts). LR is near ceiling on Criteo (baseline 0.963) and insensitive to augmentation. Findings are not GBC-specific.

### GReaT (LLM-based, GPT-2, GPU)

**Experiment 1 — German Credit (anonymized features `f0–f19`, 30% positive):**

| n | Baseline (mean ± CI) | GReaT (mean ± CI) | Gain |
|---|---|---|---|
| 50 | 0.6446 ± 0.1153 | 0.6375 ± 0.0765 | −0.7 pts |
| 100 | 0.7079 ± 0.0419 | 0.6377 ± 0.0333 | −7.0 pts |
| 200 | 0.7587 ± 0.0284 | 0.6995 ± 0.0519 | −5.9 pts |
| 500 | 0.7607 ± 0.0190 | 0.7307 ± 0.0213 | −3.0 pts |

**Finding:** Consistent negative gains. Anonymized feature names remove all LLM prior value.

**Experiment 2 — Hillstrom (semantic features, 0.9% positive, holdout=10K):**

| n | Baseline (mean ± CI) | GReaT (mean ± CI) | Gain | Win rate |
|---|---|---|---|---|
| 50 | 0.4937 ± 0.0058 | 0.5162 ± 0.0469 | +2.3 pts | 4/5 seeds |
| 100 | 0.4884 ± 0.0293 | 0.4999 ± 0.0525 | +1.2 pts | 3/5 seeds |
| 500 | 0.5124 ± 0.0692 | 0.5122 ± 0.0788 | ~0 pts | 3/5 seeds |
| 2000 | 0.5345 ± 0.0596 | 0.4658 ± 0.0461 | **−6.9 pts** | 0/5 seeds |

**Finding:** Marginal positive at n=50 (not robust to GReaT-fit variance — sign-flipped in re-run). Robustly harmful at n=2000 (paired p=0.001, FDR-significant).

### Core conclusion

**The only reliable driver of synthetic augmentation gain is extreme class imbalance (positive rate < ~5%).** CTGAN is the recommended method — it matches or exceeds all alternatives including TabDDPM at a fraction of the compute cost. TabDDPM delivers consistent gains on imbalanced data but does not surpass CTGAN and requires GPU. GReaT is not recommended for production augmentation: the small-n positive signal is not robust to GReaT-fit variance, and at scale under severe imbalance it actively degrades performance.

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

For TabDDPM (GPU required):
```bash
# On Databricks GPU cluster:
# %pip install synthcity==0.2.11
# Paste experiments/run_tabddpm_databricks.py into a notebook cell
# Set LLMSYNTH_WORK_DIR env var to your Workspace path
```
