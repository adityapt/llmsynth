#!/bin/bash
# Driver loop for Nomao full-metrics rerun: one fresh subprocess per
# (seed, method, alpha), skipping combos already in results/fullmetrics_nomao.csv.
cd "$(dirname "$0")/.."

SEEDS=(42 123 7 2024 999)
METHODS=(Baseline GaussianCopula CTGAN SMOTE)
ALPHAS=(0.1 0.2 0.3 0.5 1.0)

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    if [ "$method" == "Baseline" ]; then
      alpha_list=(0)
    else
      alpha_list=("${ALPHAS[@]}")
    fi
    for alpha in "${alpha_list[@]}"; do
      done_check=$(python3 -c "
import pandas as pd, os
p = 'results/fullmetrics_nomao.csv'
if not os.path.exists(p):
    print('no')
else:
    df = pd.read_csv(p)
    hit = ((df.seed==$seed) & (df.method=='$method') & (df.alpha==$alpha)).any()
    print('yes' if hit else 'no')
" 2>/dev/null)
      if [ "$done_check" == "yes" ]; then
        continue
      fi
      echo "=== seed=$seed method=$method alpha=$alpha ==="
      python3 experiments/_fullmetrics_nomao_worker.py "$seed" "$method" "$alpha"
    done
  done
done
echo "Nomao full-metrics driver complete."
