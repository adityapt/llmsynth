%python
# Measure actual positive rate in TabDDPM-generated synthetic rows
# Fills in the "~approximated" rows in Table 2 of paper2-empirical.md
#
# Run on Databricks GPU cluster with:
#   %pip install synthcity==0.2.11
#   dbutils.library.restartPython()
#
# Output: prints a table ready to paste into the paper

import warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

WORK_DIR  = os.environ.get("LLMSYNTH_WORK_DIR", "/Workspace/Users/<your-username>/Temp")
TARGET    = "target"
SEEDS     = [42, 123, 7, 2024, 999]
N_CAP     = 10_000
N_SYN     = 8_000   # α=1.0 — same as main experiments
N_ITER    = 2000
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)

DATASETS = [
    ("Hillstrom Email", f"{WORK_DIR}/hillstrom.csv",     None,      42),
    ("Criteo Display",  f"{WORK_DIR}/criteo_uplift.csv", N_CAP * 5, 42),
]

results = []

for name, data_path, preload_nrows, pool_seed in DATASETS:
    if not os.path.exists(data_path):
        print(f"[skip] {name}: {data_path} not found", flush=True)
        continue

    df = pd.read_csv(data_path, nrows=preload_nrows)
    df = df.dropna().reset_index(drop=True)
    if pool_seed is not None:
        df = df.sample(min(N_CAP * 5, len(df)), random_state=pool_seed).reset_index(drop=True)

    real_pos = df[TARGET].mean()
    print(f"\n{'='*60}", flush=True)
    print(f"{name}: {df.shape}, real positive rate = {real_pos*100:.2f}%", flush=True)

    seed_pos_rates = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        df_sample = df.sample(min(N_CAP, len(df)), random_state=seed).reset_index(drop=True)
        df_train, _ = train_test_split(df_sample, test_size=0.2, random_state=seed,
                                        stratify=df_sample[TARGET])

        try:
            loader = GenericDataLoader(df_train, target_column=TARGET)
            model = Plugins().get("ddpm", n_iter=N_ITER, is_classification=True,
                                   device=DEVICE, random_state=seed)
            print(f"  seed={seed} fitting TabDDPM...", end="", flush=True)
            model.fit(loader)
            syn = model.generate(count=N_SYN).dataframe()
            syn = syn[df_train.columns]
            syn[TARGET] = pd.to_numeric(syn[TARGET], errors="coerce").fillna(0).astype(int)
            pos_rate = syn[TARGET].mean() * 100
            seed_pos_rates.append(pos_rate)
            print(f" positive rate = {pos_rate:.2f}%", flush=True)
        except Exception as e:
            print(f" FAIL({str(e)[:80]})", flush=True)

    if seed_pos_rates:
        mean_rate = np.mean(seed_pos_rates)
        std_rate  = np.std(seed_pos_rates)
        results.append({
            "dataset": name,
            "real_positive_rate": f"{real_pos*100:.2f}%",
            "tabddpm_synthetic_positive_rate": f"{mean_rate:.2f}% ± {std_rate:.2f}%",
            "n_seeds": len(seed_pos_rates),
        })
        print(f"\n  {name}: TabDDPM synthetic positive rate = {mean_rate:.2f}% ± {std_rate:.2f}%", flush=True)

print("\n" + "="*60)
print("TABLE 2 UPDATE — TabDDPM synthetic positive rate:")
print("="*60)
for r in results:
    print(f"  {r['dataset']}")
    print(f"    Real: {r['real_positive_rate']}")
    print(f"    TabDDPM: {r['tabddpm_synthetic_positive_rate']} (n={r['n_seeds']} seeds)")
print("\nDone. Paste these numbers into Table 2 in paper2-empirical.md")
