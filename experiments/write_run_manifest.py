"""Write results/RUN_MANIFEST.md: the setup and the output files of the seeded CTGAN reruns.

Records the library versions, hardware, thread settings, seeds, the git commit and uncommitted changes,
the commands that produced each file, and for every result file its row count, column names and
SHA-256. Run it after the reruns finish so the record matches the files on disk.
Usage (from the repo root): python3 experiments/write_run_manifest.py
"""
import hashlib, platform, subprocess, sys, datetime
from pathlib import Path
import pandas as pd

R = Path("results")
OUTPUTS = [
    ("run_full_metrics_hillstrom_criteo.py", ["ci_hillstrom_fullmetrics.csv", "ci_criteo_fullmetrics.csv"]),
    ("run_full_metrics_benchmark_datasets.py", ["fullmetrics_telco.csv", "fullmetrics_bank_marketing.csv",
                                               "fullmetrics_german_credit.csv", "fullmetrics_nomao.csv"]),
    ("run_dose_response_bank_marketing.py", ["dose_response_bank_marketing.csv"]),
    ("run_dose_response_nomao.py", ["dose_response_nomao.csv"]),
    ("run_ci_multi_classifier.py", ["ci_multi_classifier_hillstrom.csv", "ci_multi_classifier_criteo.csv"]),
    ("measure_class_distribution_all_generators.py", ["synthetic_class_distribution_5seed.csv"]),
    ("run_nomao_sparse_ci.py", ["ci_nomao_sparse.csv"]),
    ("run_tstr_lowdata.py", ["tstr_summary.csv", "lowdata_telco_churn.csv", "lowdata_bank_marketing.csv",
                             "lowdata_credit_default.csv", "lowdata_nomao_lead.csv"]),
]


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pkg(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "unavailable"


out = ["# Run manifest: seeded CTGAN reruns", "",
       f"Written {datetime.datetime.now().isoformat(timespec='seconds')} by `experiments/write_run_manifest.py`.", "",
       "## Setup", "",
       f"- Python {sys.version.split()[0]}, {platform.platform()}, {platform.machine()}, {sh('sysctl', '-n', 'hw.ncpu')} logical CPUs",
       f"- sdv {pkg('sdv')}, torch {pkg('torch')}, scikit-learn {pkg('sklearn')}, numpy {pkg('numpy')}, pandas {pkg('pandas')}",
       "- CTGAN: SDV `CTGANSynthesizer`, epochs=150, batch size min(500, rows), `cuda=False` (CPU), `torch.set_num_threads(2)`,",
       "  `torch.manual_seed` set from a hash of the training data and the number of rows requested.",
       "- Other math libraries limited to one thread (OMP, OpenBLAS, MKL, Accelerate), set by `experiments/launch_ctgan_reruns.sh`",
       "  and as defaults in `synthetic_data_eval.py`.",
       "- Thread count matters for SMOTE on Telco: nearest-neighbour ties resolve differently with different BLAS thread counts.",
       "  Telco, seed 7, alpha 0.5: AUC 0.83884 with 1 thread (these results), 0.84230 with 10 threads (the earlier results),",
       "  0.84083 with 4. Telco SMOTE rows therefore differ from the earlier files by at most 0.0035 AUC. No other non-CTGAN row changed.",
       "- The earlier Nomao control file (September) does not reproduce with the current code: its Baseline AUC differs in the fifth decimal for 4 of 5 seeds,",
       "  and the same value is obtained with 1 and 10 threads. The Nomao rows are therefore new (10,000-row cap, `launch_nomao_fullmetrics.sh`).",
       "  `fullmetrics_nomao_n15000_unused.csv` is an earlier run of the same experiment at 15,000 rows and is not used.",
       "- The single-seed TSTR German Credit baseline (0.798) differs from the earlier summary table (0.775), which could not be reproduced.",
       "- Seeds: 42, 123, 7, 2024, 999 (multi-classifier runs also 10, 20, 30, 40, 50).",
       f"- Git commit: {sh('git', 'rev-parse', 'HEAD')}; uncommitted files: {len(sh('git', 'status', '--short').splitlines())}",
       "- Launch: `bash experiments/launch_ctgan_reruns.sh` from the repo root. Each script was run from the start,",
       "  because CTGAN also draws from NumPy's global generator, which each script seeds once at its start.",
       "- Every CTGAN output is saved in `results/synthetic_cache/` (parquet), named by torch seed, rows requested and",
       "  training-set shape and a hash of the training data, so any metric can be recomputed without refitting. Files written",
       "  before the hash was added have no hash suffix, and a run that passed an explicit seed to data of the same shape could overwrite one.",
       f"- Cached CTGAN outputs: {len(list((R / 'synthetic_cache').glob('*.parquet')))}", "",
       "## Output files", "", "| Script | File | Rows | Columns | SHA-256 |", "|---|---|---|---|---|"]
for script, files in OUTPUTS:
    for f in files:
        p = R / f
        if p.exists():
            d = pd.read_csv(p)
            out.append(f"| `{script}` | `{f}` | {len(d)} | {', '.join(d.columns)} | `{sha(p)[:16]}` |")
        else:
            out.append(f"| `{script}` | `{f}` | MISSING | | |")
logs = sorted((R / "rerun_logs").glob("*.log"))
out += ["", "## Logs", ""] + [f"- `results/rerun_logs/{l.name}`" for l in logs]
(R / "RUN_MANIFEST.md").write_text("\n".join(out) + "\n")
print("wrote results/RUN_MANIFEST.md")
