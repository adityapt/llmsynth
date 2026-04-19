# LLMSynth

Empirical evaluation of synthetic data generation methods for marketing and product data science.

All key findings are backed by **5-seed cross-validation** (seeds 42, 123, 7, 2024, 999) with 95% confidence intervals via t-distribution.

## What's Here

**Paper:** `papers/synthetic-data-marketing-eval.md`
*Does Synthetic Data Improve Model Performance? A Practical Evaluation for Marketing and Product Data Science*

**Experiments:** `experiments/`
- `synthetic_data_eval.py` — main experiment runner (Telco Churn, Bank Marketing, German Credit, Online Retail CLV)
- `run_remaining.py` — runs Credit Default + CLV datasets using cached results from first two
- `run_nomao.py` — Nomao lead dataset (full, n=10K)
- `run_nomao_sparse.py` — Nomao with 70% simulated missingness + small n=500
- `run_kdd_appetency.py` — KDD Cup 2009 Appetency (natural CRM sparsity, 70% missing)
- `run_hillstrom.py` — Hillstrom Email Marketing (real campaign data, 0.9% conversion)
- `run_criteo.py` — Criteo Uplift Display Advertising (real ad data, 0.2% conversion)
- `run_great_colab.py` — GReaT (LLM-based) experiment, designed to run on Google Colab (GPU required)

**Results:** `results/`
- `metrics_*.csv` — AUC/F1/AP per dataset × generator × condition
- `ci_*.csv` — 5-seed confidence interval results per dataset
- `ucurve_*.png` — U-shaped augmentation curves
- `lowdata_*.png` — performance vs. training set size
- `summary_table.csv` — cross-dataset summary

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
- **GReaT** (Generate Realistic Tabular data) — fine-tunes a GPT-2 language model on rows serialized as natural language text (e.g. `"age is 35, balance is 1200, target is 1"`), then samples new rows by prompting the model. be-great v0.0.13

  **Key finding:** GReaT requires GPU. On CPU, `model.sample()` fails to generate valid rows after repeated attempts regardless of model size (distilgpt2 or gpt2). With GPU + `guided_sampling=True`, sampling succeeds. See `experiments/run_great_colab.py` for a ready-to-run Google Colab notebook.

  **How to run GReaT:**
  1. Open `experiments/run_great_colab.py` in Google Colab
  2. Set runtime to **T4 GPU**
  3. Upload `data/credit_default.csv`
  4. Run all cells — results saved to `great_results.csv`

## Key Findings

Results are 5-seed mean ± 95% CI. Gains are AUC-ROC points (absolute).

| Dataset | Positive Rate | Best Method | Gain (5-seed CI) | Verdict |
|---|---|---|---|---|
| Telco Churn | 26.6% | GaussianCopula | +0.28 pts | Skip it |
| Bank Marketing | 11.7% | CTGAN | −0.17 pts | Skip it |
| German Credit | 30.0% | CTGAN | +0.81 pts | Marginal |
| Nomao Lead | 28.3% | GaussianCopula | −0.06 pts | Skip it |
| Nomao Sparse (70% missing) | 28.3% | CTGAN | +1.22 pts | Marginal |
| Hillstrom Email | 0.9% | SMOTE | +6.98 pts | **Strong yes** |
| Criteo Display Ads | 0.2% | CTGAN | +13.23 pts | **Strong yes** |
| GReaT (LLM) on German Credit | 30.0% | — | pending GPU run | — |

**Core finding:** The only reliable driver of augmentation value is **extreme class imbalance** (positive rate < ~5%). Balanced datasets show negligible or negative gains regardless of dataset size.

## Setup

```bash
pip install sdv imbalanced-learn scikit-learn matplotlib pandas numpy openpyxl
python experiments/synthetic_data_eval.py
```

For GReaT (LLM-based):
```bash
pip install be_great
# Then use experiments/run_great_colab.py on Google Colab (GPU required)
```
