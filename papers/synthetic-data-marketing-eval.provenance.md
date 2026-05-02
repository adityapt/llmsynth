# Provenance — synthetic-data-marketing-eval

**Final draft:** April 2026 (literature synthesis revised April 2026; original §6 empirical work added April 2026)
**Canonical artifact:** `papers/synthetic-data-marketing-eval.md`
**Literature-synthesis figures:** `papers/fig2-augmentation-utility.png`, `papers/fig3-method-dimension-matrix.png`, `papers/fig5-privacy-utility.png`
**Empirical-experiment plots:** `results/plots/*.png`, `results/ucurve_*.png`, `results/lowdata_*.png`, `results/summary_comparison.png`
**Experiment code:** `experiments/`
**Result tables:** `results/metrics_*.csv`, `results/ci_*.csv`, `results/summary_table.csv`

---

## Revision History

### April 2026 — Original empirical work added (§6)

Section 6 ("Empirical Validation on Public Benchmark Datasets") was added to complement the literature synthesis in §1–5. It reports original experiments on eight public datasets, all with 5-seed cross-validation and 95% CI via t-distribution. Datasets, methods, and outputs are enumerated in the "Original Empirical Work" section below.

### April 2026 — Literature-synthesis revision

Three primary sources missing from the prior draft were integrated:

1. **Davila et al. (2025)** — added as a backbone citation in §2.2 (method descriptions), §3.1 (U-shaped curve), §3.3–3.4 (when it helps/hurts), §5 (method ranking table and Figure 3), §4.1 (churn), and §4.4 (segmentation). Tables 5, 6, and 8 from Davila et al. are cited throughout.
2. **Shumailov et al. (2024)** — model collapse in *Nature* — was previously listed as a primary source but absent from the body text. A dedicated subsection (§3.2) now correctly scopes the finding to iterative recursive LLM training and explicitly distinguishes it from single-round tabular augmentation.
3. **Erickson et al. (2025)** (TabArena) and **Fonseca & Bacao (2023)** were added to the reference list and body.

---

## Source Accounting

The paper cites **20 references**, listed below with verification status.

| # | Source | Type | Access method | Verification status |
|---|--------|------|---------------|---------------------|
| 1 | Xu et al. 2019 (CTGAN, NeurIPS) | Primary paper | URL | **Supported** |
| 2 | Kotelnikov et al. 2023 (TabDDPM, ICML) | Primary paper | URL | **Supported** |
| 3 | Davila et al. 2025 (DSJ prosumer benchmark) | Primary paper | URL | **Supported** — Tables 5, 6, 8 cited throughout |
| 4 | Shumailov et al. 2024 (model collapse, Nature) | Primary paper | arXiv:2305.17493 + Nature DOI | **Supported** — correctly scoped to iterative training |
| 5 | Erickson et al. 2025 (TabArena, NeurIPS 2025) | Primary paper | URL | **Supported** — contextual reference |
| 6 | Du & Li 2024 (arXiv:2402.06806) | Primary paper | URL | **Supported** |
| 7 | Sidorenko et al. 2025 (arXiv:2504.01908, MOSTLY AI QA) | Primary paper | URL | **Supported** |
| 8 | Lautrup et al. 2024 (SynthEval, arXiv:2404.15821) | Primary paper | URL | **Supported** |
| 9 | Won et al. 2026 (MDPI Electronics 15/4/883) | Primary paper | URL | **Supported** |
| 10 | Tanha et al. 2026 (Springer IJCIS, DOI 10.1007/s44196-026-01204-3) | Primary paper | Springer full text fetched | **Supported — author list partially inferred** |
| 11 | Fonseca & Bacao 2023 (Expert Systems with Applications) | Primary paper | URL | **Supported** |
| 12 | Camino et al. 2020 (ICML Workshop) | Primary paper | URL | **Supported** |
| 13 | Shidani et al. 2025 (arXiv:2510.08095) | Primary paper | arXiv abstract fetched | **Supported — title inferred from abstract** |
| 14 | Chia Ramírez 2025 (arXiv:2510.18252) | Primary paper | arXiv abstract fetched | **Supported — title inferred from abstract** |
| 15 | Chen et al. 2025 (arXiv:2504.14061) | Primary paper | URL | **Supported** |
| 16 | Chawla et al. 2002 (SMOTE, JAIR) | Primary paper | Standard reference | **Supported** |
| 17 | Borisov et al. 2023 (GReaT, ICLR; arXiv:2210.06280) | Primary paper | URL | **Supported** — method evaluated in §6.6 |
| 18 | Patki et al. 2016 (SDV, IEEE DSAA) | Primary paper | URL | **Supported** — GaussianCopula source |
| 19 | Hillstrom 2008 (MineThatData E-Mail Challenge) | Dataset | Blog URL | **Supported** — dataset used in §6.8 |
| 20 | Diemert et al. 2018 (Criteo Uplift, AdKDD) | Dataset | Criteo AI Lab URL | **Supported** — dataset used in §6.8 |

---

## Original Empirical Work (§6)

### Datasets evaluated

| Dataset | n | Task | Positive rate | Source | Used in §6 subsection |
|---|---|---|---|---|---|
| Telco Churn | 7,032 | Classification | 26.6% | IBM Kaggle | §6.2–6.5 |
| Bank Marketing | 15,000 | Classification | 11.7% | UCI / OpenML | §6.2–6.5 |
| German Credit | 1,000 | Classification | 30.0% | OpenML id=31 | §6.2–6.5; §6.6 (GReaT) |
| Online Retail CLV | 4,000 | Regression | — | UCI fallback | §6.2–6.5 |
| Nomao Lead (full) | 10,000 | Classification | 28.3% | OpenML id=1486 | §6.7 |
| Nomao Lead (sparse, 70% missing) | 500 | Classification | 28.3% | OpenML id=1486 | §6.7 |
| Hillstrom Email | 64,000 | Classification | 0.9% | MineThatData (2008) | §6.8; §6.6 (GReaT) |
| Criteo Uplift | 13.9M (cap 10K) | Classification | 0.2% | Criteo AI Lab (2018) | §6.8 |

### Methods evaluated

| Method | Library / Version | Where evaluated |
|---|---|---|
| GaussianCopula | SDV v1.36 | All §6 datasets |
| CTGAN | SDV / CTGAN v0.12 | All §6 datasets |
| SMOTE | imbalanced-learn v0.12 | Classification datasets only |
| GReaT (GPT-2) | be-great v0.0.13 | §6.6 — German Credit, Hillstrom |
| GReaT (distilgpt2) | be-great v0.0.13 | §6.6 — Colab attempt; documented sampling failure |

### Methods *not* evaluated (cited from external benchmarks)

| Method | Where referenced | Source of claim |
|---|---|---|
| TabDDPM | §2.2, §5 ranking, §7 flowchart, §10 Rec #4 | Kotelnikov et al. 2023; Davila et al. 2025 (Tables 5, 6, 8) |
| TabSyn | §2.2, §5 ranking | Davila et al. 2025 |
| TVAE | §5 ranking | Davila et al. 2025 |
| Hybrid SMOTE+GAN | §5 ranking, §10 Rec #4 | Tanha et al. 2026 |
| CTAB-GAN+ | §5, Fig 5 | Davila et al. 2025 (Table 8) |
| MOSTLY AI / Gretel.ai | §1, §4.4, §7 | Vendor / Sidorenko et al. 2025 |
| PrivBayes / PrivSyn / PATE-GAN / DP-CTGAN | §3.4, §5 | Chen et al. 2025 |
| REaLTabFormer | §1, §7 | Cited only |

A note has been added to the §5 method-ranking table footnote and §10 Recommendation #4 making this scope distinction explicit.

### Statistical reporting

All §6.6 (GReaT) and §6.8 (Hillstrom, Criteo) results report 5-seed cross-validation (seeds 42, 123, 7, 2024, 999) with 95% CI via t-distribution. The §6.8 Criteo result includes a noted single-seed outlier (seed 2024 baseline AUC=0.567 vs. 0.965–0.979 for other seeds); the headline +12.9 AUC pts mean is reported alongside the without-outlier mean of +5.5 pts.

---

## Key Claims Verification

| Claim | Evidence | Status |
|-------|----------|--------|
| U-shaped error curve for synthetic fraction | Shidani et al. 2025; replicated in §6.3 across Telco Churn, Bank Marketing, German Credit | **Supported (literature + original)** |
| Optimal α* ≈ 0.2–0.3 across generators and datasets | §6.3 measured directly | **Supported (original)** |
| Optimal majority:minority ratio ≈ 6.6:1 | Chia Ramírez 2025 — single dataset; author cautions against overgeneralisation | **Supported with caveat** |
| TabDDPM/TabSyn top on augmentation benchmarks | Davila et al. 2025 (Table 6); Kotelnikov et al. 2023 | **Supported (external only — not replicated in §6)** |
| SMOTE top on ML utility (imbalanced), low privacy | Davila et al. 2025 (Tables 5, 8); §6.8 (Hillstrom +6.98 AUC pts) | **Supported (literature + original)** |
| Hybrid SMOTE+GAN outperforms either alone on churn | Tanha et al. 2026 | **Supported (external only)** |
| TSTR consistently lags TRTR | Du & Li 2024; Sidorenko et al. 2025; Davila et al. 2025; §6.2 measured 4–27% gaps | **Supported (literature + original)** |
| Model collapse applies to recursive/iterative training, not single-round tabular augmentation | Shumailov et al. 2024 — scope correctly described in §3.2 | **Supported** |
| LLM-based generators lower than diffusion/GAN on utility benchmarks | Davila et al. 2025; §6.6 GReaT German Credit −3 to −7 AUC pts | **Supported (literature + original)** |
| GReaT requires semantic feature names + adequate class balance | §6.6 — anonymized features (German Credit) yield consistent negative gain; semantic features (Hillstrom) yield marginal positive at small-n only | **Supported (original)** |
| Statistical DP methods higher utility, deep learning faster | Chen et al. 2025 | **Supported (external only)** |
| Causal structure corruption by off-the-shelf synthesizers | Methodological argument; no single counter-example paper | **Supported (methodological)** |
| Specific AUC % improvements in literature-synthesis figures (Figs 2, 3, 5) | Illustrative relative rankings, not exact benchmark numbers | **Explicitly labelled as illustrative** |
| §6 AUC improvements (Hillstrom, Criteo, etc.) | Directly measured, 5-seed CI | **Supported (original)** |
| 20–40% optimal synthetic fraction range | Consistent with consensus; §6.3 measured 20–30% | **Supported (literature + original)** |

---

## What Was Not Verified

- **TabDDPM and TabSyn** were not evaluated in §6. The §5 ranking row and §10 Recommendation #4 rest entirely on Davila et al. 2025 (Tables 5, 6, 8) and Kotelnikov et al. 2023. A direct replication on the §6 datasets is a natural follow-up.
- **Exact author list for Tanha et al. 2026** — "Tanha et al." from the plan's prior research pass. Readers should verify the journal page directly.
- **Exact paper titles for Shidani et al. 2025 and Chia Ramírez 2025** — titles inferred from arXiv abstract page display; should be confirmed against the official PDF title pages before formal citation.
- **SMOTE near-duplicate / privacy risk** — attributed to Davila et al. 2025, Table 8 (DCR-based privacy scores); a specific membership-inference attack paper is not cited separately.
- **KDD Cup 2009 Appetency** — script `experiments/run_kdd_appetency.py` exists in the repo but was not used in the paper; no `metrics_kdd_appetency.csv` or `ci_kdd_appetency.csv` outputs exist. Excluded from §6.
- **Telco Churn × GReaT** — flagged as an open follow-up in §11 to cleanly separate the semantic-feature and class-imbalance effects observed in §6.6.

---

## Figure Provenance

### Literature-synthesis figures (§1–5; illustrative)

| Figure | Type | Data basis | Illustrative? |
|--------|------|------------|---------------|
| Fig 1 — Taxonomy (Mermaid) | Inline diagram | Literature survey | N/A |
| Fig 2 — Augmentation utility bar chart | PNG (Vega-Lite) | Relative estimates from Davila et al. 2025, Tanha et al. 2026, Won et al. 2026 | **Yes — illustrative** |
| Fig 3 — Method × Dimension heatmap | PNG (Vega-Lite) | Relative scores from method ranking table in draft, anchored to Davila et al. 2025 | **Yes — illustrative** |
| Fig 4 — Decision flowchart (Mermaid) | Inline diagram | Synthesized decision rules | N/A |
| Fig 5 — Privacy–utility scatter | PNG (Vega-Lite) | Relative scores from method table; privacy scores from Davila et al. 2025 Table 8 | **Yes — illustrative** |

### Empirical-experiment plots (§6; directly measured)

| Plot file | Source experiment | Section |
|---|---|---|
| `results/plots/plot_tstr_gap.png` | TSTR vs TRTR across §6.2 datasets | §6.2 |
| `results/plots/plot_ucurves_grid.png` | α-mixing sweep, all §6.3 datasets | §6.3 |
| `results/plots/plot_ci_hillstrom.png` | 5-seed CI on Hillstrom by method | §6.8 |
| `results/plots/plot_ci_criteo.png` | 5-seed CI on Criteo by method | §6.8 |
| `results/plots/plot_imbalance_vs_gain.png` | Best augmentation gain vs. positive rate, all §6 datasets | §6.8 synthesis |
| `results/plots/plot_best_gain_by_dataset.png` | Best gain per dataset summary | §6.8 synthesis |
| `results/plots/plot_great_smalln.png` | GReaT performance vs n on Hillstrom / German Credit | §6.6 |
| `results/ucurve_*.png` (8 files) | Per-dataset α-mixing curves | §6.3, §6.7 |
| `results/lowdata_*.png` (5 files) | Performance vs. real training-set size | §6.4 |
| `results/summary_comparison.png` | Cross-dataset summary | §6.5 |

All §6 plots are produced by `experiments/make_plots.py` from CSVs in `results/`.

---

## Reproducibility

Experiment code: `experiments/`
Result tables: `results/metrics_*.csv`, `results/ci_*.csv`, `results/summary_table.csv`
Plot generation: `experiments/make_plots.py`
GReaT (GPU required): `experiments/run_great_databricks.py`, `run_great_hillstrom_databricks.py`, `run_great_kaggle.py`; CPU fallback (`run_great_colab.py`) documents the sampling-failure mode.
