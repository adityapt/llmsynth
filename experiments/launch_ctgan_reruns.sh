#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results/rerun_logs; date > results/rerun_logs/START
for s in run_full_metrics_hillstrom_criteo run_full_metrics_benchmark_datasets run_dose_response_bank_marketing run_dose_response_nomao run_ci_multi_classifier measure_class_distribution_all_generators run_nomao_sparse_ci run_tstr_lowdata; do
  ( python3 experiments/$s.py > results/rerun_logs/$s.log 2>&1; echo "exit $?" >> results/rerun_logs/$s.log; date >> results/rerun_logs/$s.done ) &
done
wait
date > results/rerun_logs/ALLDONE
