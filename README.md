# MarSynth

Empirical evaluation of synthetic data generation methods for marketing and product data science.

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

**Results:** `results/`
- `metrics_*.csv` — AUC/F1/AP per dataset × generator × condition
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

- **GaussianCopula** — SDV v1.36
- **CTGAN** — SDV/CTGAN v0.12
- **SMOTE** — imbalanced-learn v0.12

## Key Findings

| Setting | Augmentation Gain | Verdict |
|---|---|---|
| Large dataset, complete features | < 0.3 pts | Skip it |
| Small dataset (n≈1K), balanced | +5.3 pts AUC | Worth it |
| Small dataset + simulated sparsity | +2.5 pts (138% recovery) | Strong yes |
| Large dataset, naturally sparse + imbalanced | Marginal | Depends on imbalance |
| Real email campaign, 0.9% conversion (Hillstrom) | +8.3 pts (CTGAN) | Strong yes |
| Real display ads, 0.2% conversion (Criteo) | +19.6 pts (CTGAN) | Strong yes |

## Setup

```bash
pip install sdv imbalanced-learn scikit-learn matplotlib pandas numpy openpyxl
python experiments/synthetic_data_eval.py
```
