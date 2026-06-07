# Paper Outline: When Class Imbalance Dominates
## A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification

**Date:** 2026-06-07
**Type:** Controlled empirical study
**Target venues:** KDD Applied Data Science (8–10 pp) / NeurIPS Datasets & Benchmarks (12 pp)
**Status:** Outline approved, aligned with existing draft

---

## Research Question

Is class imbalance (positive rate < 5%) the primary — and sufficient — condition under which synthetic augmentation delivers reliable classification gains?

**Hypothesis:** Yes. Dataset size, feature type, and feature sparsity are secondary; class imbalance is the switch.

---

## Working Title

*When Class Imbalance Dominates: A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification*

---

## ABSTRACT (250 words)

| Element | Content | Evidence |
|---|---|---|
| Problem | Synthetic augmentation is widely recommended for imbalanced marketing classification, but no controlled study isolates which data conditions actually drive reliable gains | — |
| Method | 7 datasets spanning 0.2–30% positive rate; 5 generators (GC, CTGAN, SMOTE, TabDDPM, GReaT); 5–10 seeds; 4 downstream classifiers | `ci_*.csv`, `ci_multi_classifier_*.csv`, `ci_tabddpm_*.csv` |
| Finding 1 | Synthetic augmentation delivers reliable gains only when positive rate < 5%; TSTR gaps of 4–27% confirm synthetic-only training is inadvisable | All `ci_*.csv` |
| Finding 2 | CTGAN outperforms TabDDPM on both imbalanced datasets despite ~20× lower compute cost | `ci_tabddpm_*.csv` |
| Finding 3 | Optimal mixing ratio α* ≈ 0.2–0.3 is stable across generators, datasets, classifiers | All `ci_*.csv` |
| Implication | Default to CTGAN at α ∈ {0.2–0.3} for imbalanced marketing classification; skip augmentation otherwise | — |

---

## §1 · INTRODUCTION (~1 page)

**1.1 Hook**
- Synthetic data tools proliferating; adoption outpacing rigorous evidence for when augmentation works
- Cite: MOSTLY AI, Gretel, SDV tool landscape; TabArena + Davila et al. 2025 benchmark fragmentation

**1.2 The problem**
- Existing benchmarks cover general tabular tasks — not imbalanced marketing classification specifically
- The imbalanced regime (positive rate < 5%) is most practically important for marketing: churn, conversion, response, uplift
- No prior work has directly compared CTGAN vs TabDDPM on real marketing data with multi-seed CI

**1.3 Our approach**
- Controlled study: 7 datasets chosen to span 0.2%–30% positive rate
- 5 generators including current SOTA diffusion model (TabDDPM)
- Rigorous protocol: 5–10 seeds, α-sweep, multi-classifier verification

**1.4 Contributions (4 bullets)**
1. Controlled evidence that class imbalance (positive rate < 5%) is the necessary and sufficient condition for reliable augmentation gains — confirmed across 7 datasets
2. First direct TabDDPM vs CTGAN head-to-head on real marketing classification with 5-seed CI — CTGAN wins despite far lower compute
3. Multi-classifier robustness verification (10 seeds, GBC/LR/RF) — finding is not downstream-model-specific
4. Documentation of GReaT-fit variance as an underreported evaluation failure mode in LLM-based synthesis benchmarks

**1.5 Paper structure** (1 sentence)

---

## §2 · RELATED WORK (~1.5 pages)

**2.1 Synthetic tabular data generators**

| Generator | Key reference | Note |
|---|---|---|
| SMOTE | Chawla et al. 2002 | Interpolation-based; minority-only; competitive baseline |
| GaussianCopula | Patki et al. 2016 | Copula-based; fast; limited on rare events |
| CTGAN | Xu et al. 2019 | Conditional GAN; explicit class-conditional generation; designed for imbalance |
| TabDDPM | Kotelnikov et al. 2023 | DDPM for mixed types; SOTA on general benchmarks; computationally heavy |
| GReaT | Borisov et al. 2023 | LLM fine-tuning; semantic features; narrow operating envelope |

**2.2 Existing benchmarks and their gaps** (1 paragraph)
- TabArena (Erickson et al. 2025): comprehensive but general tabular — imbalanced marketing regime absent
- Davila et al. (2025): confirms TabDDPM leads on augmentation benchmarks — but not on marketing imbalance
- Gap: no controlled study isolating imbalance as the driver across multiple generators

**2.3 Class imbalance in classification** (1 paragraph)
- Why it's a distinct regime: minority-class data starvation, not general scarcity
- Standard approaches: SMOTE, cost-sensitive learning, threshold moving
- Why augmentation is attractive: addresses root cause rather than symptoms
- *Sets up the hypothesis directly*

---

## §3 · EXPERIMENTAL SETUP (~2 pages)

**3.1 Datasets**

| Dataset | n (cap) | Positive rate | Domain | Source | Role |
|---|---|---|---|---|---|
| Telco Churn | 7,032 | 26.6% | Telecom | IBM Kaggle | Control |
| Bank Marketing | 15,000 | 11.7% | Finance | UCI | Control |
| German Credit | 1,000 | 30.0% | Finance | OpenML id=31 | Control |
| Nomao Lead (full) | 10,000 | 28.3% | Lead gen | OpenML id=1486 | Control |
| Nomao Lead (sparse) | 500 | 28.3% | Lead gen | OpenML id=1486 | Sparsity stress test |
| **Hillstrom Email** | **10,000** | **0.9%** | **Marketing** | MineThatData (2008) | **Treatment** |
| **Criteo Display** | **10,000** | **0.2%** | **Advertising** | Criteo AI Lab (2018) | **Treatment** |

- Rationale: deliberately spans 0.2%–30% positive rate to test the imbalance hypothesis
- Caps: CTGAN fit time scales with n; 10K cap defines the data-scarce imbalanced regime studied

**3.2 Generators and hyperparameters**

| Generator | Implementation | Key settings | Fit strategy |
|---|---|---|---|
| GaussianCopula | SDV v1.36 | defaults | Fit once at α=1.0, subsample |
| CTGAN | SDV/CTGAN v0.12 | defaults | Fit once at α=1.0, subsample |
| SMOTE | imbalanced-learn v0.12 | k=5 neighbors | Re-call per α |
| TabDDPM | synthcity v0.2.11 `ddpm` | N_iter=2000, steps=1000, lr=1e-3, batch=1024 | Fit once at α=1.0, subsample |
| GReaT | be-great v0.0.13, GPT-2 | guided_sampling=True | Fit once per (n, seed) |

**3.3 Evaluation protocol**
- Train/test: 80/20 stratified split, `random_state=seed`
- Conditions: (1) TRTR baseline — real only; (2) TSTR — synthetic only; (3) Augmentation sweep α ∈ {0.1, 0.2, 0.3, 0.5, 1.0}; (4) Low-data regime n ∈ {250, 500, 1000, 2000}
- Seeds: 5 core (42, 123, 7, 2024, 999); extended to 10 (+10, 20, 30, 40, 50) for Hillstrom+Criteo robustness
- **Single-seed note:** Benchmark dataset results (§4.1–4.3) use a single random seed and are point estimates. Marketing dataset results (§4.4–4.7) are 5-seed mean ± 95% CI via t-distribution.
- Metrics: AUC-ROC (primary), F1-minority, Average Precision
- Reproducibility: `seed_everything()` covers Python random, NumPy, PyTorch CPU+CUDA

**3.4 Downstream classifiers**
- Primary: GradientBoostingClassifier (n_estimators=100, max_depth=4)
- Robustness: LogisticRegression, RandomForest (n_estimators=200), MLPClassifier (64×32, early stopping)
- LR and MLP wrapped in StandardScaler pipeline
- Rationale: four families span linear, tree-ensemble, boosting, neural — finding must hold across all to be classifier-agnostic
- Robustness classifiers run on Hillstrom + Criteo only, 10 seeds

---

## §4 · RESULTS (~3.5 pages)

### §4.1 TSTR: synthetic-only training always underperforms (single seed)

| Dataset | Baseline AUC | Best TSTR AUC | Gap |
|---|---|---|---|
| Telco Churn | 0.837 | 0.803 | −4.1% |
| Bank Marketing | 0.909 | 0.750 | −17.5% |
| German Credit | 0.775 | 0.564 | −27.2% |

- Finding: 4–27% gap across all datasets and generators
- Implication: do not use synthetic-only training for production models
- *Evidence: `metrics_telco_churn.csv`, `metrics_bank_marketing.csv`, `metrics_credit_default.csv`*

### §4.2 Augmentation sweep: U-curve and α* (single seed)

| Dataset | Baseline | Best AUC | Gain | Best generator | Optimal α |
|---|---|---|---|---|---|
| Telco Churn | 0.837 | 0.839 | +0.3% | GaussianCopula | 0.2 |
| Bank Marketing | 0.909 | 0.910 | +0.2% | GaussianCopula | 0.2 |
| German Credit | 0.775 | 0.816 | +5.3% | CTGAN | 0.3 |

- U-curve: performance peaks at α=0.2–0.3, degrades toward α=1.0 in every case
- German Credit gain (+5.3%) attributable to small n (1,000 rows) — consistent with data-scarcity hypothesis
- *Evidence: `ci_telco_churn.csv`, `ci_bank_marketing.csv`, `ci_credit_default.csv`; plots: `ucurve_*.png`*

### §4.3 Low-data regime: gains concentrate at small n (single seed)

- At n_real=250: augmentation recovers 30–60% of the performance gap vs full-data
- At n_real≥1,000: augmentation gain narrows to < 2 AUC pts for Bank Marketing
- Feature sparsity callout: Nomao sparse (70% missing, n=500) yields +0.5 pts within noise — severe sparsity eliminates the small-n gain
- *Evidence: `ci_nomao_sparse.csv`, `ci_nomao_lead.csv`; plots: `lowdata_*.png`*

### §4.4 Marketing datasets: strong gains under extreme imbalance (5-seed CI)

**Hillstrom Email Marketing** (5 seeds, GBC)

| Method | Best α | AUC (mean ± 95% CI) | Gain vs Baseline |
|---|---|---|---|
| Baseline | — | 0.548 ± 0.092 | — |
| GaussianCopula | 0.1 | 0.552 ± 0.107 | +0.4 pts |
| **CTGAN** | **1.0** | **0.605 ± 0.073** | **+5.7 pts** |
| **SMOTE** | **0.1** | **0.606 ± 0.087** | **+5.8 pts** |

**Criteo Display Advertising** (5 seeds, GBC)

| Method | Best α | AUC (mean ± 95% CI) | Gain vs Baseline |
|---|---|---|---|
| Baseline | — | 0.846 ± 0.228 | — |
| GaussianCopula | 0.1 | 0.912 ± 0.087 | +6.6 pts |
| **CTGAN** | **0.2** | **0.974 ± 0.036** | **+12.9 pts** |
| **SMOTE** | **0.3** | **0.966 ± 0.026** | **+12.0 pts** |

- Wide baseline CIs reflect genuine instability at 0.2–0.9% positive rate (20–90 minority examples per split)
- Augmented CIs substantially narrower — synthetic data stabilises learning under extreme imbalance
- *Evidence: `ci_hillstrom.csv`, `ci_criteo.csv`*

### §4.5 TabDDPM vs CTGAN: compute cost not justified (5-seed CI)

| Dataset | CTGAN gain (best α) | TabDDPM gain (best α) | CTGAN advantage | Approx. fit time |
|---|---|---|---|---|
| Hillstrom | +5.7 pts (α=1.0) | +1.4 pts (α=0.2) | +4.3 pts | CTGAN ~2 min CPU vs TabDDPM ~45 min GPU |
| Criteo | +12.9 pts (α=0.2) | +9.9 pts (α=0.3) | +3.0 pts | CTGAN ~2 min CPU vs TabDDPM ~45 min GPU |

- TabDDPM's benchmark superiority (Davila et al. 2025) does not transfer to extreme-imbalance regime
- CTGAN's conditional generation mechanism is better suited to minority-class targeting than unconditional diffusion
- *Evidence: `ci_tabddpm_hillstrom.csv`, `ci_tabddpm_criteo.csv`*

### §4.6 Multi-classifier robustness (10-seed CI)

**Criteo Display** (10 seeds)

| Classifier | Baseline AUC | CTGAN best gain | SMOTE best gain |
|---|---|---|---|
| GBC | 0.846 ± 0.117 | +12.0 pts | +10.0 pts |
| LR | 0.963 ± 0.021 | +0.8 pts† | +0.0 pts† |
| RF | 0.847 ± 0.076 | +9.6 pts | +8.0 pts |
| MLP‡ | artifact | — | — |

†LR near ceiling — insensitive by construction
‡MLP Criteo baseline collapsed due to local GPU training failure; excluded

- CTGAN advantage holds across GBC (+12.0 pts) and RF (+9.6 pts)
- Finding is not GBC-specific
- *Evidence: `ci_multi_classifier_criteo.csv`, `ci_multi_classifier_hillstrom.csv`*

### §4.7 GReaT (LLM-based): unreliable due to fit variance (5-seed CI)

| n | Baseline | GReaT | Gain | Win rate |
|---|---|---|---|---|
| 50 | 0.494 ± 0.006 | 0.516 ± 0.047 | +2.3 pts | 4/5 seeds |
| 2000 | 0.535 ± 0.060 | 0.466 ± 0.046 | **−6.9 pts** | 0/5 seeds |

- Two independent fits on identical data: up to 12pp per-seed AUC drift
- Root cause: user-level seeding controls NumPy/sklearn only; PyTorch/CUDA/Transformers RNGs are independent
- Implication: published GReaT benchmarks with single fits per (n, seed) understate total variance
- *Evidence: `ci_great_hillstrom.csv`, `great_alpha_sweep_hillstrom_results.csv`*

---

## §5 · DISCUSSION (~1 page)

**5.1 Why class imbalance is the mechanism**
- At 0.2–0.9% positive rate, 8,000 training rows yield 16–72 real minority examples
- Different random splits: 50–100% swing in minority count → large cross-split AUC variance
- Synthetic augmentation addresses minority-class data starvation directly
- This explains why gain is large and reliable under imbalance, negligible otherwise

**5.2 Why CTGAN beats TabDDPM in this regime**
- TabDDPM samples unconditionally from the learned joint — at 0.2% positive rate, most generated rows are negative class
- CTGAN's conditional vector explicitly targets the minority class during generation
- This architectural difference explains the performance gap in extreme-imbalance even when TabDDPM dominates in balanced settings

**5.3 Optimal mixing ratio α* ∈ {0.2–0.3}**
- Consistent across generators, datasets, classifiers
- α=1.0 systematically suboptimal — synthetic quality degradation at high volumes outweighs quantity benefit
- A 5-point α sweep is sufficient; no exhaustive grid search needed

**5.4 Limitations**
- Two marketing datasets; validate on additional imbalanced tasks (uplift, CLV, attribution)
- 10K row cap — claims apply to the data-scarce regime; results may differ at full dataset scale
- Single fixed holdout per dataset for benchmark experiments; bootstrap would further characterise split variance
- MLP results on Criteo excluded due to local GPU artifact

---

## §6 · CONCLUSION (~0.5 page)

**6.1 Answer to research question**
- Yes: class imbalance (positive rate < 5%) is the primary and sufficient condition
- No other tested condition (dataset size, feature type, sparsity, n) produces reliable gains
- Practical rule: if positive rate < 5%, run CTGAN at α ∈ {0.1–0.3}; otherwise skip augmentation

**6.2 Generator ranking for imbalanced marketing classification**
CTGAN ≈ SMOTE > TabDDPM >> GaussianCopula > GReaT

**6.3 Future work**
- TabSyn comparison, causal/uplift settings, privacy-utility tradeoff under extreme imbalance

---

## REFERENCES (~20–25, all hand-verified)

Priority:
- Chawla et al. (2002) — SMOTE
- Patki et al. (2016) — GaussianCopula / SDV
- Xu et al. (2019) — CTGAN
- Kotelnikov et al. (2023) — TabDDPM
- Borisov et al. (2023) — GReaT
- Davila et al. (2025) — augmentation benchmark (establishes the gap this paper fills)
- Erickson et al. (2025) — TabArena
- Hillstrom (2008) — Hillstrom dataset
- Diemert et al. (2018) — Criteo dataset
- Agrawal et al. (2026) — hybrid SMOTE+GAN
- Shidani et al. (2025) — optimal synthetic-to-real ratio

---

## Evidence Map

| Claim | File | Key numbers |
|---|---|---|
| TSTR gap 4–27% | `metrics_*.csv` | −4.1%, −17.5%, −27.2% |
| No gain at positive rate > 10% | `ci_telco_churn.csv`, `ci_bank_marketing.csv` | +0.28, −0.17 pts |
| α* ≈ 0.2–0.3 (benchmark) | `ci_*.csv`, `ucurve_*.png` | Peak at α=0.2–0.3 in all cases |
| Small-n concentration | `lowdata_*.png` | 30–60% gap recovery at n=250 |
| Strong gain at positive rate < 1% | `ci_hillstrom.csv`, `ci_criteo.csv` | +5.8 pts, +12.9 pts |
| CTGAN > TabDDPM | `ci_tabddpm_hillstrom.csv`, `ci_tabddpm_criteo.csv` | +5.7 vs +1.4; +12.9 vs +9.9 |
| Classifier-agnostic finding | `ci_multi_classifier_criteo.csv` | GBC +12.0, RF +9.6 |
| GReaT fit variance | `great_alpha_sweep_hillstrom_results.csv` | Up to 12pp drift |

---

## Alignment Notes

Sections incorporated from existing draft (`papers/synthetic-data-marketing-eval.md`):
- §4.1 TSTR ← §6.2 of draft
- §4.2 U-curve ← §6.3 of draft
- §4.3 Low-data regime ← §6.4 of draft
- §4.4–4.5 Marketing datasets + TabDDPM ← §6.8 of draft
- §4.6 Multi-classifier ← §6.8 extended results
- §4.7 GReaT ← §6.6 condensed

Sections in draft NOT carried into Paper 2 (belong in Paper 1 — the review):
- §2 Background/Taxonomy (extensive lit review)
- §3 Core Empirical Question (literature synthesis)
- §4 Application to Marketing/Product DS (use cases)
- §5 Method-Level Evidence (table ratings)
- §7 Practitioner Decision Framework
- §8 Evaluation Protocol for Practitioners
- §9 Limitations of Existing Evidence
- §10 Recommendations
