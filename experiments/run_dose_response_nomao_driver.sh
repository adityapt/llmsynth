#!/bin/bash
# Driver loop: spawns one fresh subprocess per (minority_count, seed, method),
# checking results/dose_response_nomao.csv each time to skip completed combos.
# This is the memory-robust version of run_dose_response_nomao.py — every
# unit of work gets a clean process, guaranteeing full OS-level memory
# release between CTGAN/GaussianCopula fits on Nomao's 119 features.
cd "$(dirname "$0")/.."

COUNTS=(16 64 256 512 1024)
SEEDS=(42 123 7 2024 999)
METHODS=(Baseline GaussianCopula CTGAN SMOTE)

for mc in "${COUNTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      done_check=$(python3 -c "
import pandas as pd, os
p = 'results/dose_response_nomao.csv'
if not os.path.exists(p):
    print('no')
else:
    df = pd.read_csv(p)
    hit = ((df.minority_count==$mc) & (df.seed==$seed) & (df.method=='$method')).any()
    print('yes' if hit else 'no')
" 2>/dev/null)
      if [ "$done_check" == "yes" ]; then
        continue
      fi
      echo "=== running minority_count=$mc seed=$seed method=$method ==="
      python3 experiments/_dose_response_nomao_worker.py "$mc" "$seed" "$method"
    done
  done
done
echo "Nomao dose-response driver complete."
