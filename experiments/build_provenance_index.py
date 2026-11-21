"""Write papers/paper2-electronics-provenance.md: which files and scripts produced every table,
figure and key number in the Electronics manuscript.

Result-file dates come from file modification times. The scripts that write or read each result file
are found by searching experiments/*.py for the file name, so the links are read from the repo.
Usage (from the repo root): python3 experiments/build_provenance_index.py
"""
import datetime as dt
from pathlib import Path

R, E = Path("results"), Path("experiments")
SCRIPTS = {p.name: p.read_text() for p in sorted(E.glob("*.py")) + sorted(E.glob("*.sh"))}

LATEST, OLDER, SINGLE, DERIVED = "seeded rerun (October 2026)", "earlier run, not rerun", "single run", "derived or checked"


def stamp(fname):
    p = R / fname
    return dt.date.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else "MISSING"


HELPERS = {"build_provenance_index.py", "verify_manuscript_numbers.py"}


def referencing(fname, keep_helpers=False):
    """Scripts in experiments/ whose text contains the file name. This includes the script that wrote the
    file and the plot or analysis scripts that read it; a name match cannot tell them apart."""
    hits = [n for n, txt in SCRIPTS.items() if fname in txt and (keep_helpers or n not in HELPERS)]
    return ", ".join(f"`{h}`" for h in hits) or "no script in the repo mentions this file name"


# (manuscript item, [result files], status, note)
ITEMS = [
    ("Table 1 (datasets, rows used, positive rates)", [], DERIVED, "Data loaders in `synthetic_data_eval.py`, `run_hillstrom.py`, `run_nomao.py`, `run_criteo.py`; caps read from the `N_CAP` constants (Bank Marketing 15,000 rows, Nomao and the marketing datasets 10,000). Checked by `verify_manuscript_numbers.py`."),
    ("Table 2 (synthetic positive rate)", ["synthetic_class_distribution_5seed.csv"], LATEST, "Written by `measure_class_distribution_all_generators.py`. TabDDPM row removed because no data file contains it."),
    ("Table 3 and Tables 5c to 5f (four control datasets)", ["fullmetrics_telco.csv", "fullmetrics_bank_marketing.csv", "fullmetrics_german_credit.csv", "fullmetrics_nomao.csv"], LATEST, "Telco, Bank Marketing and German Credit: `run_full_metrics_benchmark_datasets.py`. Nomao (10,000 rows): `launch_nomao_fullmetrics.sh`, which runs `run_fullmetrics_nomao_driver.sh` and `_fullmetrics_nomao_worker.py` per seed and combines the parts. `fullmetrics_nomao_n15000_unused.csv` is an earlier run at 15,000 rows and is not used. Section 3.3 text, Figures 1 and 2 and Table 7 use the same files."),
    ("Table 4 and Figure 9: CTGAN, SMOTE, GaussianCopula, baseline", ["ci_hillstrom_fullmetrics.csv", "ci_criteo_fullmetrics.csv"], LATEST, "Written by `run_full_metrics_hillstrom_criteo.py`. SMOTE, GaussianCopula and baseline rows are identical to the earlier runs; CTGAN rows are new because PyTorch is now seeded."),
    ("Table 4 and Figure 9: ADASYN, Borderline-SMOTE, random undersampling", ["missing_baselines_hillstrom.csv", "missing_baselines_criteo.csv"], LATEST, "No CTGAN involved; not rerun."),
    ("Table 4: class weighting", ["ci_balanced_hillstrom.csv", "ci_balanced_criteo.csv", "fullmetrics_balanced_hillstrom.csv", "fullmetrics_balanced_criteo.csv"], LATEST, "No CTGAN involved; not rerun. AUCs are identical in the `ci_balanced_*` and `fullmetrics_balanced_*` files; the second set also holds precision, recall and accuracy."),
    ("Table 4 and Figure 7: TabDDPM at 2,000 and 10,000 iterations", ["ci_tabddpm_hillstrom.csv", "ci_tabddpm_criteo.csv", "ci_tabddpm_hillstrom_10000.csv", "ci_tabddpm_criteo_10000.csv"], OLDER, "Seeds PyTorch already; not rerun. Baselines are identical to the seeded marketing run, so paired comparisons with the CTGAN rows are valid."),
    ("Table 4, Section 3.3 and Figure 8: GReaT with GPT-2", ["ci_great_hillstrom.csv"], OLDER, "Seeds NumPy only (the seed_everything routine was added on 2026-05-18, after these runs); not rerun."),
    ("Section 3.3 and Figure 8: Mistral-7B", ["modernllm_hillstrom_results_parallel.csv", "modernllm_telco_results_parallel.csv", "modernllm_german_results_parallel.csv"], OLDER, "Seeds PyTorch already; not rerun. Produced by several revisions of `run_modernllm_all_databricks.py` on 2026-06-12; see README, Known issues. Several runs have fewer than 5 valid seeds; the text and Figure 8 say which."),
    ("Tables 5a and 5b (Hillstrom, Criteo threshold metrics)", ["ci_hillstrom_fullmetrics.csv", "ci_criteo_fullmetrics.csv", "missing_baselines_hillstrom.csv", "missing_baselines_criteo.csv", "fullmetrics_balanced_hillstrom.csv", "fullmetrics_balanced_criteo.csv"], LATEST, "TabDDPM precision, recall and accuracy were never stored, so TabDDPM is not in these tables."),
    ("Table 6 and Figures 10 to 12 (dose-response)", ["dose_response_bank_marketing.csv", "dose_response_nomao.csv"], LATEST, "Written by `run_dose_response_bank_marketing.py` and `run_dose_response_nomao.py`. Figure 12 also uses the CTGAN gains from the control and marketing files above."),
    ("Section 3.3 multi-classifier results and Figures 5 and 6", ["ci_multi_classifier_criteo.csv", "ci_multi_classifier_hillstrom.csv"], LATEST, "10 seeds, written by `run_ci_multi_classifier.py`. CTGAN fits once per seed and dataset."),
    ("Section 3.3 sparsity test and Figure 3", ["ci_nomao_sparse.csv"], LATEST, "5 seeds, written by `run_nomao_sparse_ci.py`. The dense reference is the `DenseBaseline` rows of the same file. Baseline, DenseBaseline and GaussianCopula rows reproduce the earlier file exactly."),
    ("Section 3.1 (TSTR percentages)", ["tstr_summary.csv"], SINGLE, "Single seed (42), written by `run_tstr_lowdata.py`. The German Credit baseline here (0.798) differs from the earlier summary table (0.775), which could not be reproduced with the current data file."),
    ("Section 3.5 (14-test FDR family) and Table 3 FDR correction", ["fdr_family_14.csv", "stats_table3.csv", "stats_free_baselines.csv", "stats_dose_response.csv", "stats_power.csv"], DERIVED, "Written by `compute_statistics.py` from the 5-seed files, the 10-seed multi-classifier files, and the TabDDPM, class-weighting and GReaT files. `verify_manuscript_numbers.py` recomputes the same quantities independently."),
    ("Section 4.5: unseeded refit test", ["ctgan_refit_variance.csv"], OLDER, "Measured before PyTorch was seeded. `run_ctgan_refit_variance.py` still reproduces the unseeded behaviour (it passes `seed=-1`)."),
    ("Section 4.5: seeding check", ["ctgan_seed_check.csv"], LATEST, "Written by `run_ctgan_seed_check.py`: the same seed twice, and five different seeds."),
    ("Section 4.5: comparison of the CTGAN runs", ["unseeded_runs/ci_hillstrom_april_unseeded.csv", "unseeded_runs/ci_hillstrom_fullmetrics_sept_unseeded.csv", "unseeded_runs/ci_criteo_april_unseeded.csv", "unseeded_runs/ci_criteo_fullmetrics_sept_unseeded.csv", "ci_hillstrom_fullmetrics.csv", "ci_criteo_fullmetrics.csv"], DERIVED, "The April and September runs did not seed PyTorch and are kept for this comparison. Computed inside `verify_manuscript_numbers.py`."),
    ("Section 2.2.3 (Bank Marketing duration feature)", ["bank_duration_check.csv"], LATEST, "Written by `check_bank_duration.py`."),
]

FIGS = [
    (1, "fig1_summary_comparison.png", "make_plots_paper2.py", "fullmetrics_* (controls), ci_*_fullmetrics.csv (marketing)"),
    (2, "fig2_ucurves_benchmark.png", "make_plots_paper2.py", "fullmetrics_* (controls)"),
    (3, "fig3_ucurve_sparse.png", "make_plots_paper2.py", "ci_nomao_sparse.csv"),
    (4, "fig5_marketing_ci.png", "make_plots_paper2.py", "ci_*_fullmetrics.csv"),
    (5, "fig8_mlp_rescue.png", "make_plots_paper2.py", "ci_multi_classifier_criteo.csv"),
    (6, "fig9_multiclassifier.png", "make_plots_paper2.py", "ci_multi_classifier_criteo.csv"),
    (7, "fig7_tabddpm_comparison.png", "make_plots_paper2.py", "ci_*_fullmetrics.csv, ci_tabddpm_*.csv, ci_tabddpm_*_10000.csv"),
    (8, "fig10_modernllm_comparison.png", "make_plot_fig10_modernllm.py", "ci_great_*.csv, modernllm_*_results_parallel.csv"),
    (9, "fig15_missing_baselines.png", "make_plot_missing_baselines.py", "ci_*_fullmetrics.csv, missing_baselines_*.csv"),
    (10, "fig12_dose_response.png", "make_plot_dose_response.py", "dose_response_bank_marketing.csv"),
    (11, "fig13_dose_response_combined.png", "make_plot_dose_response_combined.py", "dose_response_*.csv"),
    (12, "fig14_gain_vs_positive_rate.png", "make_plot_gain_vs_positive_rate.py", "fullmetrics_*, ci_*_fullmetrics.csv, dose_response_*.csv"),
]

NOT_STORED = [
    "Compute totals and per-seed run times (removed from the manuscript; no timing was logged).",
    "TabDDPM class rate in Table 2 (removed; no data file).",
    "TabDDPM precision, recall and accuracy (never stored; the manuscript says so).",
    "The earlier claim that four CTGAN refits ranged from 0.527 to 0.632 (replaced by `ctgan_refit_variance.csv`, `ctgan_seed_check.csv` and the comparison of the CTGAN runs).",
    "Figure 4 of the earlier draft (low-data regime, one run per size) was removed; `lowdata_*.csv` are still produced by `run_tstr_lowdata.py` but are not used in the manuscript.",
    "Row counts of the full Criteo dataset (13.9 million) are a published figure quoted in `run_criteo.py`; the local file holds 50,000 rows.",
    "The 7.88:1 ratio is from Won et al. [19], and the 11.3% positive rate is its arithmetic conversion.",
    "The class rates of GReaT were never measured (the manuscript says so).",
]

out = ["# Provenance index for the Electronics manuscript", "",
       f"Generated {dt.date.today().isoformat()} by `experiments/build_provenance_index.py`. File dates are modification times.",
       "Run `python3 experiments/verify_manuscript_numbers.py` to recompute and compare the manuscript numbers with these files.", "",
       "The script column lists every script in `experiments/` whose text mentions the file name. That includes the script that wrote the file and the plot or analysis scripts that read it, and a name match cannot tell them apart. Where no script mentions the file name, the script that wrote it was not identified.", "",
       "## Tables and numbers", "", "| Manuscript item | Result files (date) | Scripts that mention the file | Status | Note |", "|---|---|---|---|---|"]
for item, files, status, note in ITEMS:
    fl = "<br>".join(f"`{f}` ({stamp(f)})" for f in files) or "see note"
    keep = status == DERIVED
    wr = "<br>".join(f"{f}: {referencing(f, keep)}" for f in files) if files else "see note"
    out.append(f"| {item} | {fl} | {wr} | {status} | {note} |")
out += ["", "## Figures", "", "| Figure | Image file | Script | Data |", "|---|---|---|---|"]
for n, img, scr, data in FIGS:
    ok = (Path("results/plots/paper2") / img).exists() and (E / scr).exists()
    out.append(f"| {n} | `{img}` | `{scr}` | {data}{'' if ok else ' (MISSING FILE)'} |")
out += ["", "## Scripts added or changed for the manuscript", "",
        "- `experiments/verify_manuscript_numbers.py`: recomputes the manuscript numbers from the result files and compares them.",
        "- `experiments/run_ctgan_refit_variance.py`: the CTGAN refit test.",
        "- `experiments/check_bank_duration.py`: the Bank Marketing duration check.",
        "- `experiments/compute_statistics.py`: p-values, FDR correction, effect sizes and power.",
        "- `experiments/run_ctgan_seed_check.py`, `run_nomao_sparse_ci.py`, `run_tstr_lowdata.py`, `launch_nomao_fullmetrics.sh`, `launch_ctgan_reruns.sh`, `write_run_manifest.py`: the seeded CTGAN rerun and its record (`results/RUN_MANIFEST.md`).",
        "- `experiments/build_provenance_index.py`: this index.",
        "- `papers/build_docx.py`: builds the Google-Docs-friendly `.docx` from the markdown.",
        "- `experiments/make_plots_paper2.py`, `make_plot_missing_baselines.py`, `make_plot_gain_vs_positive_rate.py`: updated to read the latest runs.",
        "", "## Not stored, removed or replaced", ""] + [f"- {x}" for x in NOT_STORED]
Path("papers/paper2-electronics-provenance.md").write_text("\n".join(out) + "\n")
missing = [f for _, files, _, _ in ITEMS for f in files if not (R / f).exists()]
print("wrote papers/paper2-electronics-provenance.md;", "missing files:", missing or "none")
