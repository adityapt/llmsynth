# Run manifest: seeded CTGAN reruns

Written 2026-10-28T05:48:52 by `experiments/write_run_manifest.py`.

## Setup

- Python 3.9.13, macOS-26.7.2-arm64-arm-64bit, arm64, 10 logical CPUs
- sdv 1.36.0, torch 2.8.0, scikit-learn 1.6.1, numpy 1.26.4, pandas 2.3.2
- CTGAN: SDV `CTGANSynthesizer`, epochs=150, batch size min(500, rows), `cuda=False` (CPU), `torch.set_num_threads(2)`,
  `torch.manual_seed` set from a hash of the training data and the number of rows requested.
- Other math libraries limited to one thread (OMP, OpenBLAS, MKL, Accelerate), set by `experiments/launch_ctgan_reruns.sh`
  and as defaults in `synthetic_data_eval.py`.
- Thread count matters for SMOTE on Telco: nearest-neighbour ties resolve differently with different BLAS thread counts.
  Telco, seed 7, alpha 0.5: AUC 0.83884 with 1 thread (these results), 0.84230 with 10 threads (the earlier results),
  0.84083 with 4. Telco SMOTE rows therefore differ from the earlier files by at most 0.0035 AUC. No other non-CTGAN row changed.
- The earlier Nomao control file (September) does not reproduce with the current code: its Baseline AUC differs in the fifth decimal for 4 of 5 seeds,
  and the same value is obtained with 1 and 10 threads. The Nomao rows are therefore new (10,000-row cap, `launch_nomao_fullmetrics.sh`).
  `fullmetrics_nomao_n15000_unused.csv` is an earlier run of the same experiment at 15,000 rows and is not used.
- The single-seed TSTR German Credit baseline (0.798) differs from the earlier summary table (0.775), which could not be reproduced.
- Seeds: 42, 123, 7, 2024, 999 (multi-classifier runs also 10, 20, 30, 40, 50).
- Git commit: d7264d0fc629c74ec5697a0cc7b00309a7a9de01; uncommitted files: 88
- Launch: `bash experiments/launch_ctgan_reruns.sh` from the repo root. Each script was run from the start,
  because CTGAN also draws from NumPy's global generator, which each script seeds once at its start.
- Every CTGAN output is saved in `results/synthetic_cache/` (parquet), named by torch seed, rows requested and
  training-set shape and a hash of the training data, so any metric can be recomputed without refitting. Files written
  before the hash was added have no hash suffix, and a run that passed an explicit seed to data of the same shape could overwrite one.
- Cached CTGAN outputs: 314

## Output files

| Script | File | Rows | Columns | SHA-256 |
|---|---|---|---|---|
| `run_full_metrics_hillstrom_criteo.py` | `ci_hillstrom_fullmetrics.csv` | 80 | seed, method, condition, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `0a2841312d003029` |
| `run_full_metrics_hillstrom_criteo.py` | `ci_criteo_fullmetrics.csv` | 80 | seed, method, condition, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `0a8f3a76b452f32b` |
| `run_full_metrics_benchmark_datasets.py` | `fullmetrics_telco.csv` | 80 | seed, method, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `175639331f6a7350` |
| `run_full_metrics_benchmark_datasets.py` | `fullmetrics_bank_marketing.csv` | 80 | seed, method, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `7f471b6654c701fc` |
| `run_full_metrics_benchmark_datasets.py` | `fullmetrics_german_credit.csv` | 80 | seed, method, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `381b0473da2cba7c` |
| `run_full_metrics_benchmark_datasets.py` | `fullmetrics_nomao.csv` | 80 | seed, method, alpha, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `1d6f56fe7d7976fd` |
| `run_dose_response_bank_marketing.py` | `dose_response_bank_marketing.csv` | 100 | minority_count, positive_rate_pct, seed, method, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `2cec00a2c243cf40` |
| `run_dose_response_nomao.py` | `dose_response_nomao.csv` | 100 | minority_count, positive_rate_pct, seed, method, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `a819c152b64b4b03` |
| `run_ci_multi_classifier.py` | `ci_multi_classifier_hillstrom.csv` | 640 | seed, method, condition, alpha, classifier, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `df8bb4ebda518e64` |
| `run_ci_multi_classifier.py` | `ci_multi_classifier_criteo.csv` | 640 | seed, method, condition, alpha, classifier, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `b44e108f434a72b5` |
| `measure_class_distribution_all_generators.py` | `synthetic_class_distribution_5seed.csv` | 2 | dataset, real_positive_rate, GaussianCopula_mean, GaussianCopula_std, GaussianCopula_n_seeds, CTGAN_mean, CTGAN_std, CTGAN_n_seeds, SMOTE_mean, SMOTE_std, SMOTE_n_seeds | `6195bbc624f3e600` |
| `run_nomao_sparse_ci.py` | `ci_nomao_sparse.csv` | 60 | seed, method, alpha, auc, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `878fdc1cac2258d4` |
| `run_tstr_lowdata.py` | `tstr_summary.csv` | 12 | dataset, method, alpha, condition, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `ff7ff04f82e63feb` |
| `run_tstr_lowdata.py` | `lowdata_telco_churn.csv` | 12 | n_real, condition, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `596972faada65f79` |
| `run_tstr_lowdata.py` | `lowdata_bank_marketing.csv` | 12 | n_real, condition, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `e37cb5626414be43` |
| `run_tstr_lowdata.py` | `lowdata_credit_default.csv` | 6 | n_real, condition, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `8a3d6644de04d745` |
| `run_tstr_lowdata.py` | `lowdata_nomao_lead.csv` | 12 | n_real, condition, auc_roc, f1_minority, avg_precision, precision, recall, accuracy | `e9f5e7df190f99f3` |

## Logs

- `results/rerun_logs/measure_class_distribution_all_generators.log`
- `results/rerun_logs/nomao_seed_123.log`
- `results/rerun_logs/nomao_seed_2024.log`
- `results/rerun_logs/nomao_seed_42.log`
- `results/rerun_logs/nomao_seed_7.log`
- `results/rerun_logs/nomao_seed_999.log`
- `results/rerun_logs/run_ci_multi_classifier.log`
- `results/rerun_logs/run_ctgan_seed_check.log`
- `results/rerun_logs/run_dose_response_bank_alpha02.log`
- `results/rerun_logs/run_dose_response_bank_marketing.log`
- `results/rerun_logs/run_dose_response_nomao.log`
- `results/rerun_logs/run_full_metrics_benchmark_datasets.log`
- `results/rerun_logs/run_full_metrics_hillstrom_criteo.log`
- `results/rerun_logs/run_nomao_sparse_ci.log`
- `results/rerun_logs/run_tstr_lowdata.log`
