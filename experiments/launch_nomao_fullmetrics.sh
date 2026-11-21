#!/bin/bash
# Nomao full-metrics rerun (10,000-row cap): one driver per seed in parallel, then the parts are combined
# into results/fullmetrics_nomao.csv. Run from the repo root.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results/nomao_parts results/rerun_logs
date > results/rerun_logs/NOMAO_START
for s in 42 123 7 2024 999; do
  ( NOMAO_SEEDS=$s NOMAO_FULLMETRICS_OUT=results/nomao_parts/seed_$s.csv bash experiments/run_fullmetrics_nomao_driver.sh > results/rerun_logs/nomao_seed_$s.log 2>&1 ) &
done
wait
python3 - <<'PY'
import pandas as pd
parts = [pd.read_csv(f"results/nomao_parts/seed_{s}.csv") for s in (42, 123, 7, 2024, 999)]
d = pd.concat(parts, ignore_index=True)
d.to_csv("results/fullmetrics_nomao.csv", index=False)
print("combined rows:", len(d))
PY
date > results/rerun_logs/NOMAO_DONE
