# LLMSynth: code and results for "When Does Synthetic Data Augmentation Help? Evidence from Marketing and Product Datasets"

This repository contains the code, raw per-seed result files, and figures for the paper.
Every table, figure and number in the paper is mapped to the result files and scripts that
produced it in [`docs/provenance.md`](docs/provenance.md).

## Contents

| Path | What it holds |
|---|---|
| `experiments/` | One script per experiment, plus shared code (`synthetic_data_eval.py`), statistics (`compute_statistics.py`) and plotting scripts. `experiments/INDEX.md` lists every script with its purpose |
| `results/` | Raw per-seed result CSVs, derived statistics, `RUN_MANIFEST.md` (software versions, seeds, checksums), `rerun_logs/`, `unseeded_runs/`, and `plots/` (the figures) |
| `docs/provenance.md` | Table-by-table and figure-by-figure map to result files and scripts |

Not included: the raw datasets (downloaded on first use), the cache of CTGAN outputs
(`results/synthetic_cache/`, about 300 MB), and the manuscript drafts.

## Setup

- Python 3.9. `pip install -r requirements.txt`.
- Run every script **from the repository root**: the scripts read and write the relative paths `data/` and `results/`.
- GPU experiments (TabDDPM, GReaT with GPT-2 and Mistral-7B) were run on a single NVIDIA A100.
  Extra packages are in `requirements-gpu.txt`.
- Software versions, seeds and file checksums for the October 2026 reruns are in `results/RUN_MANIFEST.md`.
- Files named `*_databricks.py`, and `measure_tabddpm_class_distribution.py`, are Databricks notebook exports (they begin with `%python` and contain `%pip` magics). Run them as notebook cells on a GPU machine; they are not plain Python scripts. Set `LLMSYNTH_WORK_DIR` to choose where they read and write files (default `./work`).

## Data

Datasets are downloaded the first time a loader runs and cached in `data/`.
Telco Customer Churn, Hillstrom and Criteo Uplift (v2.1) are fetched from the public URLs set in
the loaders; Bank Marketing (OpenML id 1461), German Credit (OpenML id 31) and Nomao (OpenML id 1486)
come from OpenML. Row caps: Bank Marketing 15,000; Nomao, Hillstrom and Criteo 10,000; Telco and German Credit are used in full.
The dose-response experiments use a total of 10,000 training rows for both Bank Marketing and Nomao.

## Reproducing the results

| Paper item | Script(s) | Output |
|---|---|---|
| Table 2 (synthetic positive rate) | `measure_class_distribution_all_generators.py` | `synthetic_class_distribution_5seed.csv` |
| Table 3, Tables 5c-5f, Figures 1-2 (four control datasets) | `run_full_metrics_benchmark_datasets.py` (Telco, Bank Marketing, German Credit); `launch_nomao_fullmetrics.sh` (Nomao) | `fullmetrics_*.csv` |
| Table 4, Tables 5a-5b, Figure 9 (marketing datasets) | `run_full_metrics_hillstrom_criteo.py`; `run_missing_baselines_hillstrom_criteo.py`; `run_balanced_baseline.py`, `run_full_metrics_balanced.py` | `ci_*_fullmetrics.csv`, `missing_baselines_*.csv`, `ci_balanced_*.csv`, `fullmetrics_balanced_*.csv` |
| TabDDPM (Table 4, Figure 7) | `run_tabddpm_databricks.py` (`N_ITER = 2000`; set `N_ITER = 10000` for the extended run) | `ci_tabddpm_*.csv` |
| GReaT with GPT-2 (Table 4, Figure 8) | `run_great_*_databricks.py`, `run_confidence_intervals.py` | `ci_great_*.csv`, `great_*_results.csv` |
| GReaT with Mistral-7B (Figure 8) | `run_modernllm_all_databricks.py` | `modernllm_*_results_parallel.csv` |
| Dose-response (Table 6, Figures 10-12) | `run_dose_response_bank_marketing.py` (mixing ratio 1.0), `run_dose_response_bank_alpha.py` (0.2), `run_dose_response_nomao.py` | `dose_response_*.csv` |
| Multi-classifier results (Figures 5-6) | `run_ci_multi_classifier.py` | `ci_multi_classifier_*.csv` |
| Sparsity stress test (Figure 3) | `run_nomao_sparse_ci.py` | `ci_nomao_sparse.csv` |
| TSTR (Section 3.1) | `run_tstr_lowdata.py` | `tstr_summary.csv` |
| CTGAN run-to-run variance (Section 4.5) | `run_ctgan_refit_variance.py`, `run_ctgan_seed_check.py` | `ctgan_refit_variance.csv`, `ctgan_seed_check.csv` |
| Statistics (FDR family, Table 3 correction, power) | `compute_statistics.py` | `stats_*.csv`, `fdr_family_14.csv` |
| Figures | `make_plots_paper2.py`, `make_plot_*.py` | `results/plots/` |

The October 2026 seeded CTGAN reruns can be relaunched with `bash experiments/launch_ctgan_reruns.sh`.
`experiments/verify_manuscript_numbers.py` recomputes the paper's numbers from the result files; it needs
the manuscript as a markdown file (not included) and writes `results/fdr_family_14_verifier.csv`.

## Determinism

Five seeds (42, 123, 7, 2024, 999); the multi-classifier runs add 10, 20, 30, 40, 50. CTGAN runs on CPU with
PyTorch seeded before every fit and two threads. SMOTE results on Telco depend on the BLAS thread count;
see `results/RUN_MANIFEST.md`.

## Known issues and notes

1. **Bank Marketing call-duration feature.** The call-duration feature (column V12 in the OpenML file) is kept in
   every Bank Marketing run. `synthetic_data_eval.py` drops a column named `duration`, which the OpenML file does
   not have under that name, so nothing is removed. `results/bank_duration_check.csv` gives the baseline AUC with and without V12.
2. **Earlier runs.** GReaT (GPT-2 and Mistral-7B), TabDDPM, class weighting, and ADASYN, Borderline-SMOTE and random
   undersampling were run before October 2026 and were not rerun, because no CTGAN is involved.
   The GPT-2 GReaT runs (April and May 2026) predate the PyTorch seeding routine, which was added to those scripts on
   2026-05-18, and seeded NumPy only. The TabDDPM and Mistral-7B scripts call `seed_everything()`.
3. **Unseeded CTGAN runs.** The April and September runs did not seed PyTorch; their outputs are kept in `results/unseeded_runs/`
   for the comparison in Section 4.5. They were added to the history in commits `3fe90ca` (2026-04-14, April run) and
   `9c88aef` (2026-09-26, September run); the code as of those commits is what produced them. The PyTorch seeding
   (`torch.manual_seed` in `synthetic_data_eval.py`) was introduced after them. `run_ctgan_refit_variance.py` still reproduces the unseeded behaviour.
4. **Mistral-7B result files.** `run_modernllm_all_databricks.py` writes `modernllm_*_results.csv`; the files used in the paper carry
   a `_parallel` suffix that the script does not add. The script was revised repeatedly on 2026-06-12 (see the commit history).
   The final revision uses full fine-tuning, batch size 16, bf16, 8-bit AdamW and a maximum sampled row length of 200, and runs
   training sizes up to n = 500. The result files also include n = 1,000 and 2,000 (Hillstrom) and n = 1,000 (Telco), which the
   final revision does not produce, so those results come from an earlier revision. Which revision produced which result is not recorded.
5. **Run times** were not logged.

## License

MIT License. See [LICENSE](LICENSE).
