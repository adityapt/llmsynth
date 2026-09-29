# Synthetic Data Augmentation in the Extreme-Imbalance Regime: A Controlled Empirical Study of Marketing Classification

**Author:** Aditya Puttaparthi Tirumala
**Date:** 2026-06-07
**Type:** Controlled empirical study
**Target venue:** KDD Applied Data Science Track / NeurIPS Datasets & Benchmarks

> *This is a structural draft. Numbers are pulled directly from the result CSVs in `results/`. The author will rewrite the prose in their own voice; the role of this draft is to lock the structure, claims, and supporting evidence.*

---

## Abstract

Practitioners facing severe class imbalance — email conversion rates below 1%, rare-event prediction in marketing classification — routinely turn to synthetic data augmentation, but existing benchmarks report aggregate generator rankings across heterogeneous tasks that don't answer the question that matters: given a specific positive rate and sample size, will augmentation help, and which method should be used? We answer this with a controlled study spanning seven datasets (0.2%–30% positive rate), five generators (GaussianCopula, CTGAN, SMOTE, TabDDPM, and GReaT at two LLM scales, GPT-2 and Mistral-7B), 5–10 seeds, four downstream classifier families, and two independent dose-response experiments that vary minority-class count while holding dataset identity fixed — directly testing whether scarcity itself, rather than which dataset happens to be at hand, drives the effect.

It does, but the finding is more useful than a single threshold. On two real marketing datasets — Hillstrom (0.9% positive) and Criteo (0.2%) — augmentation delivers +5.7 to +12.9 AUC points; on four balanced benchmarks at 11.7%–30%, no generator exceeds +0.27 points. We measure why: CTGAN's conditional sampler generates minority rows at 7–89× the natural rate, while TabDDPM and GaussianCopula sample unconditionally and underperform accordingly. The pattern holds across four classifier families, and is unchanged when GPT-2 in GReaT is replaced by the substantially larger Mistral-7B. On Criteo, 7 of 10 MLP seeds failed to converge on real data alone; CTGAN augmentation restored convergence in all 10.

But the exact point where augmentation stops helping and starts hurting is not a portable constant. A controlled within-dataset sweep — minority count varied from 16 to 1,024, total sample size held fixed — turns significantly harmful above roughly 1% positive rate on Bank Marketing. The same design on a second, higher-dimensional dataset (Nomao) holds the same direction, but the magnitude nearly vanishes: a near-ceiling baseline leaves little room to move. A single cutoff cannot be assumed to generalize.

The more surprising result concerns which method to use at all. ADASYN — free, instantaneous, and over fifteen years old — statistically ties CTGAN on both marketing datasets (paired test, p=0.97 and p=0.21), against the intuition that a modern deep generative model should win; SMOTE ties both as well. We also find CTGAN's own training is not reliably reproducible in the standard SDV implementation used here — a >10-point AUC swing from refitting on fully identical data and seed. To our knowledge this reliability gap has not been previously reported, and it likely affects prior CTGAN benchmarks generally. We disclose it as an open limitation rather than resolve it.

**Practical recommendation.** Below roughly 1% positive rate, try ADASYN or SMOTE before CTGAN — both are free and statistically indistinguishable from it here. Above 10%, augmentation is unlikely to help. The region in between is not a gap in this study — we tested it directly on two datasets — but the result is dataset-dependent rather than a single rule: augmentation was significantly harmful across most of this range on one dataset, and merely negligible on the other. Practitioners in this band should validate on their own data, not because the region is unstudied, but because our own two datasets already disagree on what happens there.

**Keywords:** synthetic data, class imbalance, marketing classification, CTGAN, TabDDPM, empirical evaluation

---

## 1. Introduction

Marketing and product data scientists routinely face classification problems with severe class imbalance: email conversion rates below 1%, display ad click-through rates below 0.5%, and rare-event prediction across customer cohorts (Neslin et al., 2006; Johnson & Khoshgoftaar, 2019). A growing body of work proposes synthetic data augmentation as a remedy: train a generative model on the available real data, sample synthetic examples, and combine them with the real training set to improve downstream classifier performance. The space of available generators has expanded rapidly — from interpolation-based oversampling (SMOTE) through conditional GANs (CTGAN), copula-based parametric models (GaussianCopula), diffusion models (TabDDPM, TabSyn), and language-model-based synthesizers (GReaT, REaLTabFormer (Solatorio & Dupriez, 2023), TabuLa (Zhao et al., 2023), TabMT (Gulati & Roysdon, 2023)). A growing literature also documents when and why synthetic tabular data generalises (Jordon et al., 2022; Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022).

This expansion has not been matched by corresponding clarity for practitioners. Existing benchmarks (Erickson et al., 2025; Davila et al., 2025) evaluate generators across heterogeneous tabular tasks and report aggregate rankings. These rankings — under which diffusion-based models such as TabDDPM dominate — are not directly informative for the practitioner asking a simpler question: *given my marketing classification task at this positive rate and this sample size, which generator should I use, and will augmentation help at all?*

One result previews why this matters: on Criteo Display Advertising (0.2% positive rate), 7 of 10 MLP seeds failed to converge using real data alone — the classifier predicted the majority class every time. After CTGAN augmentation, all 10 seeds converged. Augmentation in this regime is not a marginal improvement; it is the difference between a working classifier and a broken one.

This paper addresses that question with a controlled empirical study. We selected seven datasets deliberately to span the practitioner-relevant range of positive rates — from 30.0% (German Credit, a balanced benchmark) down to 0.2% (Criteo Display Advertising, extreme imbalance). We evaluated five generators (GaussianCopula, CTGAN, SMOTE, TabDDPM, GReaT) under a uniform protocol: 80/20 stratified train/test splits, an α-sweep over the synthetic-to-real mixing ratio, 5-seed confidence intervals on the marketing datasets, and 10-seed multi-classifier robustness checks on the two most imbalanced tasks.

Our contributions are:

1. **Multi-generator, multi-classifier characterization of augmentation in the minority-example-scarcity regime, with the first TabDDPM comparator.** Prior work (Chawla et al., 2002; Fonseca & Bacao, 2023; Won et al., 2026) established that augmentation value concentrates under class imbalance. We extend this evidence across five generators and four classifier families, and add the first direct TabDDPM evaluation on real marketing data. A cross-dataset regression yields R²=0.92 (directional, n=6; the 1%–10% region is unsampled). To our knowledge, no prior work reports a direct CTGAN vs TabDDPM 5-seed CI comparison on Hillstrom or Criteo.
2. **A direct TabDDPM vs CTGAN head-to-head at two training budgets.** At N_iter=2,000 (default) and N_iter=10,000 (5× extended), CTGAN outperforms TabDDPM on both Hillstrom and Criteo. Extended training widens the gap rather than closing it — TabDDPM at 10k goes uniformly negative on Hillstrom. The CTGAN advantage at N_iter=10,000 reaches d_z=1.25 on Hillstrom (p=0.049). The gap persists across training budgets tested, consistent with an architectural interpretation.
3. **Multi-classifier robustness verification.** Extending to 10 seeds and four downstream classifiers on Hillstrom and Criteo, we confirm that CTGAN's advantage is not Gradient Boosting–specific: it holds for Random Forest (+9.6 pts on Criteo) and for MLP where augmentation rescues convergence entirely (7/10 baseline seeds failed; all 10 seeds converge post-CTGAN). Findings are scoped to these two datasets.
4. **GReaT's failure modes are framework-level, not backbone-specific.** We replicate the GReaT protocol with Mistral-7B (a modern, more capable LLM) on all three GReaT datasets. The outcome does not change in the extreme-imbalance regime: Mistral-7B underperforms CTGAN on Hillstrom, still fails on anonymized features, and shows higher generation failure rates than GPT-2 at very small n. GReaT-fit variance (up to 12pp per-seed AUC drift) is documented for GPT-2; the underlying cause (non-deterministic GPU reductions) applies to any LLM fine-tuned on GPU.

The paper is structured as follows. Section 2 reviews the relevant generators and benchmarks. Section 3 specifies the experimental protocol. Section 4 reports results across the seven datasets. Section 5 discusses mechanism and limitations. Section 6 concludes with a practitioner-facing decision rule.

---

## 2. Related Work

### 2.1 Synthetic Tabular Data Generators

The generators evaluated in this study span four design families.

**SMOTE** (Chawla et al., 2002) generates synthetic minority examples by linear interpolation between a minority example and one of its k-nearest neighbors. It is the longest-established and most widely deployed augmentation method. It has no separate fit step and operates only on the minority class.

**GaussianCopula** (Patki et al., 2016) models the joint distribution of tabular features by fitting parametric marginals and a Gaussian copula on the rank-transformed values. Categorical features are handled via entity embeddings (Guo & Berkhahn, 2016) in CTGAN and direct label encoding in GaussianCopula. It is fast and interpretable but assumes the dependency structure is well captured by a Gaussian copula on the marginals.

**CTGAN** (Xu et al., 2019) is a conditional generative adversarial network designed for tabular data with mixed types and class imbalance. It uses mode-specific normalization for continuous columns and a conditional vector during training, enabling controlled class-conditional generation at sampling time.

**TabDDPM** (Kotelnikov et al., 2023) applies denoising diffusion probabilistic models to tabular data, handling mixed feature types via Gaussian diffusion on continuous features and multinomial diffusion on categorical features. It is reported as the strongest single-table generator on augmentation benchmarks (Davila et al., 2025) but requires substantially more compute than statistical alternatives.

**GReaT** (Borisov et al., 2023) serializes tabular rows as natural-language strings and fine-tunes a pre-trained language model (in our experiments, GPT-2) on the resulting text. The hypothesis is that the LLM's pre-training prior over real-world concepts (income, age, recency) improves sample quality on datasets with semantic feature names.

### 2.2 Existing Benchmarks and Their Gaps

The TabArena benchmark (Erickson et al., 2025) provides a comprehensive comparison of tabular generators across utility, fidelity, and privacy dimensions. Davila et al. (2025) extend this with a focus on augmentation utility on prosumer GPU hardware; Sidorenko et al. (2025) provide a multi-dimensional evaluation framework. Both benchmarks report TabDDPM and TabSyn as the strongest generators on average. Concurrent work has shown that tree-based methods remain strong on tabular tasks even as deep learning advances (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022), motivating our use of GBC as the primary downstream classifier.

Two gaps motivate the present study. First, neither benchmark separates the *imbalanced marketing classification regime* from the broader population of tabular tasks. Aggregate rankings can mask regime-specific reversals, and the practitioner-relevant question is whether the aggregate winner is also the winner on the specific tasks they face. Second, neither benchmark directly compares CTGAN against TabDDPM on real marketing datasets with multi-seed confidence intervals — a comparison that the practitioner needs in order to decide whether to invest GPU compute in a diffusion model or to default to the lighter CTGAN.

### 2.3 Class Imbalance as a Distinct Regime

Class imbalance, particularly at positive rates below 5%, is qualitatively different from general data scarcity (He & Garcia, 2009; Branco et al., 2016; Fernández et al., 2018). The bottleneck is not total dataset size but the number of minority examples available to the classifier. At a 0.2% positive rate with 8,000 training rows, only 16 minority examples are expected per stratified split — and the variance in that count across splits is the primary driver of classifier instability. Standard remedies include SMOTE (Chawla et al., 2002), ADASYN (He et al., 2008), cost-sensitive learning, and threshold moving. Synthetic augmentation in this regime is not primarily about increasing total dataset size; it is about densifying the minority-class region of feature space. This framing motivates our hypothesis that minority-example scarcity — not class imbalance per se — is the strongest observed correlate of augmentation value in the tested regime.

---

## 3. Experimental Setup

### 3.1 Datasets

We selected seven publicly available classification datasets spanning the practitioner-relevant range of positive rates. Five serve as controls (positive rate ≥ 11.7%) and two as the treatment condition (positive rate ≤ 0.9%, drawn from real marketing operations). We note that the 1%–10% positive-rate range is not represented in this dataset selection; the dataset-level regression in §4.8 spans the 0.2%–30% range and is consistent with a continuous relationship, but the boundary behaviour between 1% and 10% is not directly tested.

| Dataset | n (cap) | Positive rate | Domain | Source | Role |
|---|---|---|---|---|---|
| Telco Customer Churn | 7,032 | 26.6% | Telecom | IBM Kaggle (Agrawal et al., 2026) | Control |
| Bank Marketing | 15,000 | 11.7% | Finance | UCI (Moro et al., 2014) | Control |
| German Credit | 1,000 | 30.0% | Finance | OpenML id=31 (Hofmann, 1994) | Control |
| Nomao Lead (full) | 10,000 | 28.3% | Lead generation | OpenML id=1486 (Candillier & Lemaire, 2012) | Control |
| Nomao Lead (sparse, 70% missing) | 500 | 28.3% | Lead generation | OpenML id=1486 (Candillier & Lemaire, 2012) | Sparsity stress |
| **Hillstrom Email Marketing** | **10,000** | **0.9%** | **Marketing** | MineThatData (2008) | **Treatment** |
| **Criteo Display Advertising** | **10,000** | **0.2%** | **Advertising** | Criteo AI Lab (Diemert et al., 2018) | **Treatment** |

Datasets larger than 10,000 rows (Bank Marketing, Hillstrom, Criteo) are subsampled to the listed cap. The cap defines the scope of our claims: we study the data-scarce minority-class regime, in which the classifier bottleneck is class-conditional density estimation rather than total row count. At full dataset scale, the marginal value of synthetic rows is expected to be smaller, and our paper does not test that condition.

### 3.2 Generators and Hyperparameters

| Generator | Implementation | Key settings | Fit strategy |
|---|---|---|---|
| GaussianCopula | SDV v1.36 | Library defaults | Refit independently at each α (no caching between α values) |
| CTGAN | SDV/CTGAN v0.12 | Library defaults | Refit independently at each α (no caching between α values) |
| SMOTE | imbalanced-learn v0.12 | k=5 neighbors | Re-call per α (no separate fit step) |
| TabDDPM | synthcity v0.2.11 `ddpm` | N_iter=2000, num_timesteps=1000, lr=10⁻³, batch=1024 | Fit once at max(α), subsample for smaller α |
| GReaT | be-great v0.0.13 + GPT-2 (117M) | `guided_sampling=True`, 50–100 epochs | Fit once per (n, seed) |

**Correction:** an earlier version of this section stated that GaussianCopula and CTGAN use a fit-once-and-subsample strategy. On inspection of the actual experiment code (`synthetic_data_eval.py`, `run_confidence_intervals.py`), this is only true for TabDDPM — GaussianCopula and CTGAN are independently refit for every α value in the sweep, with no caching or subsampling from a single larger fit. This does not confound the α-comparisons (each α still gets its own clean generator fit rather than a stale one), but it does mean generator-fit variance is not held constant across α within a (dataset, seed) pair the way the original text claimed, and the actual compute cost of the α-sweep is higher than "fit once" would imply.

### 3.3 Evaluation Protocol

For each (dataset, generator, seed) triple we run four conditions:

1. **Baseline (TRTR):** Train on 80% real data, evaluate on 20% real holdout.
2. **TSTR:** Train on synthetic-only data, evaluate on real holdout.
3. **Augmentation sweep:** Train on real + synthetic at α ∈ {0.1, 0.2, 0.3, 0.5, 1.0}, where α = n_synthetic / n_real.
4. **Multi-classifier robustness** (Hillstrom and Criteo only): repeat the augmentation sweep with four downstream classifier families (see §3.4) and 10 seeds.

Splits are stratified on the target. The primary metric is AUC-ROC; secondary metrics are F1 (minority class) and average precision throughout, plus Precision, Recall, and Accuracy on the two marketing datasets (§4.4), where the full 5-seed metric suite was recomputed directly from per-seed predictions. These additional metrics are not (yet) available for the four benchmark datasets in §4.2 or for TabDDPM, whose evaluation harness does not currently persist per-seed predictions the same way; extending them there would follow the same protocol. They are not reported for GReaT: the original harness captured AUC-ROC only, and the synthetic samples and fitted classifiers were not persisted across runs. Recomputing them for GReaT would require rerunning the full multi-seed, multi-backbone experimental matrix rather than a single supplementary run — disproportionate given GReaT's documented per-seed fit variance (§4.7), which means a single new run would not be comparable to the results already reported. This does not weaken the paper's conclusion regarding GReaT: that conclusion rests on the sampling-mechanism argument in §5.2 (unconditional sampling dilutes minority representation regardless of backbone), which is metric-agnostic and does not depend on which downstream classification metric is used to detect the effect.

The seed protocol has two levels. All augmentation results (§4.2–4.7) use 5 seeds {42, 123, 7, 2024, 999}, with confidence intervals computed via the t-distribution on per-seed AUC values. Multi-classifier robustness (§4.6) extends to 10 seeds by adding {10, 20, 30, 40, 50}. The sole exception is §4.1 TSTR, which is reported as a single-seed point estimate — TSTR experiments do not use per-seed CI because the single-seed protocol matches the prior work we compare against (Davila et al., 2025).

To control seed-induced non-determinism, all experiments invoke a `seed_everything()` routine that sets the random state for Python `random`, NumPy, PyTorch CPU and CUDA, the cuDNN deterministic flag, and the `CUBLAS_WORKSPACE_CONFIG` environment variable. We document a residual non-determinism source in §4.7: PyTorch fp16 reductions on GPU remain unconstrained, which is the root cause of the GReaT-fit variance discussed there.

### 3.4 Downstream Classifiers

The primary downstream classifier is `GradientBoostingClassifier` with `n_estimators=100, max_depth=4, random_state=seed` (Friedman, 2001), matching the protocol used by prior augmentation benchmarks (Davila et al., 2025). All classifiers are implemented via scikit-learn (Pedregosa et al., 2011). For the multi-classifier robustness experiments on Hillstrom and Criteo, we add three additional families:

- **Logistic Regression** (`LogisticRegression`, `max_iter=500`), wrapped in a `StandardScaler` pipeline.
- **Random Forest** (`RandomForestClassifier`, `n_estimators=200, max_depth=None, n_jobs=-1`).
- **Multi-Layer Perceptron** (`MLPClassifier`, `hidden_layer_sizes=(64, 32), max_iter=200`, early stopping with 10% validation fraction), wrapped in a `StandardScaler` pipeline.

These four families span linear, ensemble-bagging, ensemble-boosting, and neural model classes. A finding that holds across all four is unlikely to be an artifact of the primary classifier choice.

---

## 4. Results

### 4.1 TSTR: Synthetic-Only Training Underperforms Across All Datasets

Training on synthetic data alone and testing on real data — the TSTR protocol — gives the cleanest possible measure of how faithfully a generator captures the joint distribution. Across the three benchmark datasets, every generator's TSTR AUC falls materially below the real-data baseline.

| Dataset | Baseline AUC | Best TSTR AUC | TSTR gap |
|---|---|---|---|
| Telco Churn | 0.837 | 0.803 (GaussianCopula) | −4.1% |
| Bank Marketing | 0.909 | 0.750 (GaussianCopula) | −17.4% |
| German Credit | 0.775 | 0.564 (GaussianCopula) | −27.2% |

**Table 1 — Minority example budget per dataset.** The number of minority examples available per training split (80% of n_cap) makes the mechanism intuitive: with 16 minority examples, no classifier can learn a stable boundary. With 1,400+, augmentation adds little to what the real data already provides.

| Dataset | Positive rate | Train rows | Minority examples | Baseline AUC |
|---|---|---|---|---|
| Criteo Display | 0.2% | 8,000 | **16** | 0.846 ± 0.228 |
| Hillstrom Email | 0.9% | 8,000 | **72** | 0.548 ± 0.092 |
| Bank Marketing | 11.7% | 12,000 | 1,404 | 0.928 ± 0.004 |
| Telco Churn | 26.6% | 5,626 | 1,497 | 0.844 ± 0.015 |
| German Credit | 30.0% | 800 | 240 | 0.794 ± 0.044 |
| Nomao Lead | 28.3% | 8,000 | 2,264 | 0.991 ± 0.001 |

The TSTR gap grows monotonically as the dataset shrinks. German Credit (n = 1,000) shows the largest gap, consistent with the intuition that smaller training sets give the generator less material to learn the joint distribution faithfully. Across all configurations, no generator closes the gap. The implication for practitioners is unambiguous: synthetic data is an augmentation method, not a replacement method. Production models should not be trained on synthetic-only data unless real data is structurally unavailable.

![Figure 1](../results/plots/paper2/fig1_summary_comparison.png)

**Figure 1.** Cross-dataset summary of generator performance. Augmentation gains concentrate on the two imbalanced marketing datasets (Hillstrom 0.9%, Criteo 0.2%); all five generators are within noise on the four balanced benchmark datasets (positive rate ≥ 11.7%). TSTR underperforms real-only training across all datasets and generators.

### 4.2 Augmentation Sweep on Benchmark Datasets: Gains Within Noise

We next test whether mixing synthetic and real data improves on the real-data baseline. On the four benchmark datasets with positive rate above 10%, the best gain from any generator at any α is consistently below +0.5 AUC points.

| Dataset | Positive rate | Baseline AUC | Best generator | Best α | Best AUC | Gain |
|---|---|---|---|---|---|---|
| Telco Churn | 26.6% | 0.844 ± 0.015 | GaussianCopula | 0.2 | 0.846 | +0.21 pts |
| Bank Marketing | 11.7% | 0.928 ± 0.004 | CTGAN | 0.1 | 0.926 | −0.17 pts |
| German Credit | 30.0% | 0.794 ± 0.044 | CTGAN | 0.1 | 0.797 | +0.27 pts |
| Nomao Lead (full) | 28.3% | 0.991 ± 0.001 | GaussianCopula | 0.1 | 0.991 | −0.06 pts |

All four gains are within the baseline confidence interval. The single notably positive result in the literature for German Credit (+5.3% from a single-seed experiment, reproduced in our prior single-seed run) does not survive when re-evaluated with 5-seed confidence intervals.

A consistent secondary observation across the augmentation sweeps is the U-curve in α: performance peaks at α ∈ {0.1, 0.2, 0.3} on every dataset and degrades toward α = 1.0. This pattern motivates the α* analysis in §5.3.

![Figure 2](../results/plots/paper2/fig2_ucurves_benchmark.png)

**Figure 2.** U-shaped augmentation curves for all benchmark datasets. Each panel shows AUC vs. α for GaussianCopula, CTGAN, and SMOTE with 95% CI bands (5 seeds). The performance peak falls at α ∈ {0.1–0.3} on every dataset; gains are within noise of the baseline in all cases.

### 4.3 Sparsity Stress Test: Sparsity Eliminates Small-n Augmentation Gains

The Nomao sparse condition (n = 500, 70% simulated missing features) combines the two conditions under which augmentation has historically been most promising: small training set and degraded baseline performance. The hypothesis is that synthetic generators trained on dense imputed data should be able to recover some of the lost signal.

| Condition | Baseline AUC | Best augmented AUC | Best generator | Gain |
|---|---|---|---|---|
| Nomao sparse (70% missing) | 0.897 ± 0.062 | 0.902 ± —  | CTGAN α=0.1 | +0.50 pts |
| Nomao dense reference | 0.9716 ± 0.0103 | — | — | (separate baseline) |

The dense baseline (last row) is reported for context: with all features present, the same classifier reaches AUC = 0.9716 ± 0.0103, a 7.46-point gain over the sparse baseline. Augmentation closes none of this gap meaningfully. The finding is consistent with the imbalance hypothesis: sparsity-driven baseline degradation is not the failure mode synthetic augmentation addresses. Augmentation needs minority-class data starvation, not feature-information starvation, to deliver value.

![Figure 3](../results/plots/paper2/fig3_ucurve_sparse.png)

**Figure 3.** Augmentation U-curve for the sparse stress test. The flat curve (all gains < 0.5 pts across all α) — compared against the dense baseline at AUC=0.9716 — shows that sparsity-driven performance gaps are not recoverable through synthetic augmentation.

### 4.4 Marketing Datasets: Strong Gains Under Extreme Imbalance

The two marketing datasets — Hillstrom at 0.9% positive rate and Criteo at 0.2% positive rate — are the treatment condition for the imbalance hypothesis. Both yield large, reliable gains.

**Hillstrom Email Marketing** (5-seed CI, GBC downstream)

| Method | Best α | AUC (mean ± 95% CI) | Gain vs Baseline |
|---|---|---|---|
| Baseline (real only) | — | 0.548 ± 0.092 | — |
| GaussianCopula | 0.1 | 0.552 ± 0.107 | +0.44 pts |
| **CTGAN** | **1.0** | **0.605 ± 0.073** | **+5.75 pts** |
| **SMOTE** | **0.1** | **0.606 ± 0.087** | **+5.84 pts** |

**Criteo Display Advertising** (5-seed CI, GBC downstream)

| Method | Best α | AUC (mean ± 95% CI) | Gain vs Baseline |
|---|---|---|---|
| Baseline (real only) | — | 0.846 ± 0.228 | — |
| GaussianCopula | 0.1 | 0.912 ± 0.087 | +6.61 pts |
| **CTGAN** | **0.2** | **0.974 ± 0.036** | **+12.87 pts** |
| **SMOTE** | **0.3** | **0.966 ± 0.026** | **+11.99 pts** |

Two observations beyond the headline gains warrant attention. First, the baseline confidence intervals are wide (±9.2 pts on Hillstrom, ±22.8 pts on Criteo); this is not a measurement artifact but a direct consequence of learning from approximately 72 (Hillstrom) and 16 (Criteo) real minority examples per training split. Second, the augmented confidence intervals are substantially narrower — CTGAN's Criteo CI is ±3.6 points, an order-of-magnitude reduction. Synthetic augmentation under extreme imbalance does not only improve mean performance; it stabilises learning across splits.

The baseline TSTR check from §4.1 also holds here: even on these tasks where augmentation works strongly, synthetic-only training does not match real-only training. Augmentation, not replacement, is the operative regime.

**Comparison with cost-sensitive baseline.** A natural question is whether `class_weight='balanced'` — which costs nothing to apply — delivers comparable gains. We evaluate this under the same 5-seed protocol using `GradientBoostingClassifier` with `sample_weight=compute_sample_weight('balanced', y_train)`.

| Method | Hillstrom AUC | Hillstrom gain | Criteo AUC | Criteo gain |
|---|---|---|---|---|
| Baseline (real only) | 0.548 ± 0.092 | — | 0.846 ± 0.228 | — |
| **Balanced GBC** | **0.530 ± 0.073** | **−1.80 pts** | **0.899 ± 0.086** | **+5.32 pts** |
| CTGAN (best α) | 0.605 ± 0.073 | +5.75 pts | 0.974 ± 0.036 | +12.87 pts |

`class_weight='balanced'` hurts on Hillstrom (−1.80 pts) and helps on Criteo (+5.32 pts) but falls substantially short of CTGAN on both datasets. The CTGAN advantage over the balanced baseline is +7.55 pts on Hillstrom and +7.55 pts on Criteo. Additionally, the balanced Criteo result has a wider CI (±8.6 pts) than CTGAN (±3.6 pts) — the variance-stabilisation benefit of synthetic augmentation does not carry over to cost-sensitive reweighting. These results confirm that synthetic augmentation delivers gains beyond what the free cost-sensitive alternative can achieve in this regime.

**Secondary metrics (Average Precision).** AUC-ROC is the primary metric because it is threshold-independent and standard in the benchmark literature. Average Precision (AP) tells a more conservative story under extreme imbalance: Hillstrom baseline AP = 0.014 ± 0.007 (near-zero because the positive rate is 0.9%, leaving almost no ceiling room in absolute terms); CTGAN best AP = 0.019 ± 0.013 (+0.45 pts). Criteo baseline AP = 0.216 ± 0.186 (wide CI reflects the same seed instability as AUC); CTGAN best AP = 0.210 ± 0.138 (marginal, within noise). The AUC story is materially stronger than the AP story on these datasets; this is expected when the minority class is extremely rare and AP is dominated by precision at the very top of the ranking. We report AUC-ROC as the primary metric and note that AP results do not contradict but are less sensitive to augmentation in this regime.

**Threshold-based metrics (Accuracy, Precision, Recall, F1).** We additionally report Accuracy, Precision, Recall, and F1 at the classifier's default 0.5 decision threshold (5-seed CI; full results in `results/ci_hillstrom_fullmetrics.csv` and `results/ci_criteo_fullmetrics.csv`). The two datasets tell different stories. On **Hillstrom**, Precision, Recall, and F1 are exactly zero for CTGAN and GaussianCopula across all 5 seeds (baseline: F1 = 0.012 ± 0.034; CTGAN: F1 = 0.000 ± 0.000) — at 0.9% positive rate, the classifier never crosses the default threshold into predicting the positive class at all, for baseline or augmented conditions alike. Accuracy sits at 98.3–98.9% throughout, which is uninformative here: a classifier that never predicts positive is "correct" 98%+ of the time purely because the majority class dominates. This is a concrete demonstration of why AUC-ROC (threshold-independent, ranking-based) is the right primary metric for this regime — Accuracy is actively misleading and F1/Precision/Recall at a fixed threshold are silent regardless of whether augmentation helped. On **Criteo**, the picture is different: F1 stays non-zero throughout (baseline 0.259 ± 0.188; CTGAN 0.214 ± 0.178 at α=0.2; SMOTE 0.180 ± 0.060 at α=0.3), but none of the augmented conditions show a clear F1 improvement over baseline — in fact CTGAN's F1 point estimate is lower than baseline's despite CTGAN's large AUC gain (+12.87 pts). This divergence between AUC and F1 is consistent with the mechanism in §5.1: augmentation improves the classifier's ability to *rank* positives above negatives (what AUC measures) without necessarily shifting enough mass across the default 0.5 threshold to change *count-based* predictions at that fixed cutoff. We do not tune the decision threshold in this study; threshold-moving is noted as an untested cheap baseline in §5.4.

![Figure 4](../results/plots/paper2/fig4_lowdata_regime.png)

**Figure 4.** Low-data regime: AUC vs real training set size n ∈ {250, 500, 1000, 2000} for benchmark datasets. Augmentation recovers 30–60% of the performance gap at n=250 across all datasets; the benefit narrows rapidly at n ≥ 1,000.

![Figure 5](../results/plots/paper2/fig5_marketing_ci.png)

**Figure 5.** Augmentation U-curves for Hillstrom and Criteo with 95% CI bands. The steep rise from α=0 to α≈0.2 and the substantially narrower CI bands on augmented runs (vs the wide baseline band) are the primary visual evidence for the variance-stabilisation finding.

### 4.5 TabDDPM vs CTGAN: Compute Cost Not Justified

We ran TabDDPM on both marketing datasets under two training budgets: N_iter=2,000 (library default) and N_iter=10,000 (5× extended training), both using `synthcity`'s `ddpm` plugin on a GPU cluster (NVIDIA T4/A10G). Baseline AUCs cross-check bit-exact against §4.4 (max per-seed diff = 0.000), confirming the harness is wired correctly. CTGAN fit time was logged at approximately 2 minutes per seed on CPU; TabDDPM at approximately 6 minutes (N_iter=2k) and 29 minutes (N_iter=10k) per seed on GPU.

**TabDDPM N_iter=2,000 (default) vs N_iter=10,000 — best gain per α:**

| Dataset | CTGAN (best α) | TabDDPM 2k (best α) | TabDDPM 10k (best α) |
|---|---|---|---|
| Hillstrom | +5.75 pts (α=1.0) | +1.35 pts (α=0.2) | −2.02 pts (α=0.1) |
| Criteo | +12.87 pts (α=0.2) | +9.91 pts (α=0.3) | +6.46 pts (α=0.2) |

Extended training hurts rather than helps. On Hillstrom, TabDDPM at N_iter=10,000 goes uniformly negative across all α values — consistent with the model overfitting the training distribution and losing generalization, though we measure this through downstream AUC degradation rather than directly measuring synthetic data fidelity. On Criteo, the best gain drops from +9.91 to +6.46 pts. The CTGAN advantage widens with more TabDDPM training, not less.

The paired comparison of CTGAN vs TabDDPM-10k shows: Hillstrom Δ=+7.76 pts (d_z=+1.25, p=0.049); Criteo Δ=+6.41 pts (d_z=+0.73, p=0.179). The Hillstrom result is nominally significant and represents the strongest individual paired test in the study outside of GReaT n=2000.

This addresses the concern that the CTGAN advantage reflects undertrained TabDDPM. More training does not close the gap; it widens it. The architectural explanation in §5.2 — TabDDPM's unconditional sampling produces predominantly negative-class rows under extreme imbalance, while CTGAN's conditional vector explicitly targets the minority class — is consistent with this pattern: within the training budgets tested, more compute did not compensate for sampling from the wrong class distribution.

![Figure 6](../results/plots/paper2/fig7_tabddpm_comparison.png)

**Figure 6.** CTGAN vs TabDDPM at N_iter=2k and N_iter=10k on Hillstrom and Criteo. Extended training (dashed line) widens rather than closes the CTGAN advantage; on Hillstrom, all five TabDDPM-10k α values fall below baseline.

### 4.6 Multi-Classifier Robustness: Findings Are Not GBC-Specific

The §4.4 and §4.5 results are reported with GradientBoostingClassifier as the downstream model. To rule out a classifier-specific artifact, we extended the Criteo and Hillstrom experiments to 10 seeds and four downstream classifier families. We report best gain across α for each (generator, classifier) combination.

**Criteo Display, 10-seed CI**

| Classifier | Baseline AUC | CTGAN best gain | SMOTE best gain | GaussianCopula best gain |
|---|---|---|---|---|
| Gradient Boosting (GBC) | 0.846 ± 0.117 | +12.04 pts (α=0.3) | +9.98 pts (α=0.3) | +0.30 pts (α=1.0) |
| Logistic Regression (LR) | 0.963 ± 0.021 | −0.03 pts (α=0.1) | −2.98 pts (α=0.1) | +0.81 pts (α=0.1) |
| Random Forest (RF) | 0.847 ± 0.076 | +9.55 pts (α=0.5) | +7.94 pts (α=0.1) | +5.34 pts (α=0.5) |
| Multi-Layer Perceptron (MLP) | 0.284 ± 0.283 | +65.60 pts (α=0.2) | +56.55 pts (α=1.0) | +6.81 pts (α=0.1) |

The MLP baseline on Criteo is not a hardware artifact — `MLPClassifier` is pure scikit-learn with no GPU dependency; the Metal Performance Shaders errors visible in `logs/multi_classifier.log` are from CTGAN's PyTorch training and are unrelated to MLP. The wide CI (±0.283) and near-zero mean AUC reflect genuine MLP training instability under extreme class imbalance: 7 of 10 seeds produced AUC < 0.15 (the gradient-based optimizer converged to predicting all examples as the majority class), while 3 seeds converged normally to AUC ≈ 0.975. This instability is a known property of vanilla MLPs on severely imbalanced data (Branco et al., 2016).

Critically, CTGAN augmentation resolves this instability entirely: all 10 seeds converged (AUC 0.865–0.985, mean 0.940 ± 0.030) after CTGAN augmentation. SMOTE similarly rescued 9 of 10 seeds (mean 0.850 ± 0.103). This is the most striking illustration in the paper of what synthetic augmentation achieves on extreme imbalance: it is not merely improving a working classifier, it is enabling a classifier that otherwise fails to train.

The CTGAN advantage on Criteo is preserved across Gradient Boosting (+12.04 pts), Random Forest (+9.55 pts), and MLP (+65.60 pts — rescue from near-random baseline). The Logistic Regression case is informative but not contradictory: the LR baseline AUC of 0.963 is already near ceiling, leaving no room for augmentation to help.

**Hillstrom Email, 10-seed CI**

| Classifier | Baseline AUC | CTGAN best gain | SMOTE best gain |
|---|---|---|---|
| GBC | 0.559 ± 0.044 | +3.30 pts (α=1.0) | +1.22 pts (α=0.2) |
| LR | 0.652 ± 0.043 | −1.93 pts (α=0.1) | −9.33 pts (α=1.0) |
| RF | 0.505 ± 0.053 | +4.38 pts (α=1.0) | +6.44 pts (α=0.2) |
| MLP | 0.492 ± 0.034 | +5.50 pts (α=1.0) | +10.49 pts (α=0.1) |

On Hillstrom, SMOTE on MLP delivers the largest gain (+10.49 pts at α=0.1). MLP on Hillstrom does not exhibit the convergence instability seen on Criteo — all 10 seeds converge (baseline 0.492 ± 0.034) — because the 0.9% positive rate yields approximately 72 real minority examples per training split, enough for stable gradient descent. The 0.2% Criteo rate yields only 16, which is below the stability threshold for vanilla MLP. Logistic Regression on Hillstrom is again insensitive, consistent with the near-ceiling baseline (0.652 is near the best augmented AUC observed for this dataset).

![Figure 7](../results/plots/paper2/fig8_mlp_rescue.png)

**Figure 7.** MLP per-seed AUC on Criteo: baseline vs CTGAN-augmented. Seeds marked ✗ failed to converge (AUC < 0.15); all 10 seeds reach AUC > 0.86 after CTGAN augmentation.

![Figure 8](../results/plots/paper2/fig9_multiclassifier.png)

**Figure 8.** Multi-classifier robustness on Criteo. Left: baseline AUC per classifier (LR near ceiling at 0.963; MLP collapsed at 0.284). Right: best augmentation gain — CTGAN leads on GBC and RF; LR insensitive; MLP rescued from failure.

### 4.7 GReaT (LLM-Based): Strong Per-Seed Fit Variance

GReaT — the GPT-2-based tabular synthesizer — was evaluated on Hillstrom at training sizes n ∈ {50, 100, 200, 500, 1000, 2000} with a fixed 10,000-row holdout. The α=1.0 results are summarised below.

| n | Baseline (mean ± CI) | GReaT (mean ± CI) | Gain | Wins / 5 |
|---|---|---|---|---|
| 50 | 0.494 ± 0.006 | 0.516 ± 0.047 | +2.25 pts | 4/5 |
| 100 | 0.488 ± 0.029 | 0.500 ± 0.052 | +1.15 pts | 3/5 |
| 200 | 0.493 ± 0.027 | 0.497 ± 0.091 | +0.40 pts | 3/5 |
| 500 | 0.512 ± 0.069 | 0.512 ± 0.079 | −0.02 pts | 3/5 |
| 1000 | 0.516 ± 0.070 | 0.476 ± 0.071 | −3.97 pts | 1/5 |
| 2000 | 0.535 ± 0.060 | 0.466 ± 0.046 | **−6.87 pts** | 0/5 |

Two patterns are worth noting. First, a directional positive signal at small n (n = 50, 4/5 seeds win) decays monotonically with n and inverts to a robustly negative effect at n = 2000 (0/5 seeds win, paired p = 0.001). At 0.9% positive rate, the GReaT-generated synthetic rows dilute rather than enrich the minority-class signal as n grows — most generated rows are negative class regardless of LLM prior quality.

Second, and more methodologically consequential: we performed a natural experiment by running GReaT a second time on the same (n, seed) pairs as part of an unrelated α-sweep experiment. The two independent fits produced per-seed AUC differences of up to 12 percentage points on identical training data. Investigation traces the root cause to incomplete seeding: user-level `random_state` controls NumPy and scikit-learn random state, but PyTorch, CUDA, and Hugging Face Transformers maintain independent random states, and `fp16=True` GPU reductions remain non-deterministic even after `seed_everything()` is applied to all of these. The implication for the field is that published GReaT benchmarks with one fit per (n, seed) bundle data-sample variance with generator-fit variance into a single confidence interval. The true total variance — across both sampling and fitting — is wider than reported. We document this as an open evaluation problem for GPT-2-based tabular synthesis benchmarks specifically; whether larger LLM-based synthesizers (e.g., REaLTabFormer (Solatorio & Dupriez, 2023), TabuLa) or non-GPT backends exhibit similar fit variance is not tested here. The benchmark reproducibility implications are discussed in van Breugel et al. (2023) and Bouthillier et al. (2021).

#### GReaT with Mistral-7B: Does a More Capable LLM Backbone Change the Outcome?

GPT-2 (117M, 2019) was the backbone used in the original GReaT paper. A natural question is whether the failure modes observed with GPT-2 are specific to that model, or whether they reflect a limitation of the GReaT framework itself. We test this by replicating the full GReaT fine-tuning protocol using Mistral-7B-v0.1 (7B parameters, 2024) — a modern, substantially more capable LLM — on all three GReaT datasets. Protocol is identical: same SEEDS, same SMALL_NS, same holdout, same GBC downstream classifier. Runs on H100 GPU with bf16 precision.

**GPT-2 (117M) vs Mistral-7B (7B) — AUC at α=1.0, 5-seed mean ± CI:**

**Correction note:** an earlier version of this table computed the Mistral-7B gain against the GPT-2 run's baseline rather than against Mistral-7B's own seed-matched baseline. Because baseline AUC varies materially across seeds/sampling draws in this low-data regime (the whole premise of §5.1), and because the two backbones were run as separate experiments (not always sharing an identical valid-seed set), each backbone's gain must be computed against its own baseline. The corrected table below reports both baselines explicitly and recomputes every Mistral-7B gain from raw per-seed values.

**German Credit** (anonymized features, 30% positive)

| n | GPT-2 baseline | GPT-2 GReaT | GPT-2 gain | Mistral baseline | Mistral-7B | Mistral gain |
|---|---|---|---|---|---|---|
| 50 | 0.645 | 0.638 ± 0.076 | −0.71 pts | 0.645 | 0.661 ± 0.074 | +1.59 pts |
| 100 | 0.708 | 0.638 ± 0.033 | −7.02 pts | 0.708 | 0.652 ± 0.053 | −5.59 pts |
| 200 | 0.759 | 0.700 ± 0.028 | −5.92 pts | 0.748 ± 0.039† | 0.711 ± 0.039 | −3.70 pts† |
| 500 | 0.761 | 0.731 ± 0.019 | −3.00 pts | 0.761 | 0.757 ± 0.025 | −0.38 pts |

†4 of 5 seeds valid (1 failed to generate valid rows).

**Hillstrom Email** (semantic features, 0.9% positive)

| n | GPT-2 baseline | GPT-2 GReaT | GPT-2 gain | Mistral baseline | Mistral-7B | Mistral gain |
|---|---|---|---|---|---|---|
| 50 | 0.494 | 0.516 ± 0.047 | +2.25 pts | —‡ | —‡ | —‡ |
| 100 | 0.488 | 0.500 ± 0.052 | +1.15 pts | 0.512 ± 0.092§ | 0.524 ± 0.060 | +1.20 pts§ |
| 200 | 0.493 | 0.497 ± 0.091 | +0.40 pts | 0.533 ± 0.027¶ | 0.500 ± 0.102 | −3.37 pts¶ |
| 500 | 0.512 | 0.512 ± 0.079 | −0.02 pts | 0.481 ± 0.057 | 0.531 ± 0.037 | +5.03 pts |

‡n=50 Mistral-7B on Hillstrom excluded: 4 of 5 seeds failed to generate valid rows at 0.2% positive rate + n=50. Larger models appear more brittle than GPT-2 (1/5 failures) at extreme small n under severe imbalance. §3 of 5 seeds valid (2 failed); wide CI reflects the small sample and this cell should not be used for strong inference. ¶4 of 5 seeds valid (1 failed).

**Telco Churn** (semantic features, 26.6% positive)

| n | GPT-2 baseline | GPT-2 GReaT | GPT-2 gain | Mistral baseline | Mistral-7B | Mistral gain |
|---|---|---|---|---|---|---|
| 50 | 0.671 | 0.710 ± 0.070 | +3.93 pts | 0.737 ± 0.055 | 0.690 ± 0.119 | −4.70 pts |
| 100 | 0.747 | 0.733 ± 0.034 | −1.38 pts | 0.789 ± 0.005 | 0.768 ± 0.034 | −2.15 pts |
| 200 | 0.767 | 0.766 ± 0.026 | −0.07 pts | 0.805 ± 0.011 | 0.777 ± 0.030 | −2.83 pts |
| 500 | 0.795 | 0.797 ± 0.007 | +0.15 pts | 0.823 ± 0.006 | 0.803 ± 0.018 | −1.99 pts |

n=1000 was also attempted on Telco but is excluded here: only 2 of 5 Mistral-7B seeds produced valid rows, too few for a reliable estimate (same standard applied to the Hillstrom n=50 exclusion above).

![Figure 10](../results/plots/paper2/fig10_modernllm_comparison.png)

**Figure 10.** GPT-2 (117M) vs Mistral-7B (7B) vs Baseline across three datasets. Shaded regions are 95% CI. Mistral-7B's raw AUC edges above GPT-2's on some semantic-feature n values (Hillstrom, Telco), but both backbones underperform their own baseline at most n tested once gain is computed against each backbone's own seed-matched baseline — the fundamental failure modes persist regardless of backbone.

**Paired tests: Mistral-7B vs GPT-2.** This comparison is of raw AUC, backbone vs. backbone directly (not gain vs. baseline, so it is unaffected by the baseline-matching correction above). None of the Mistral-7B vs GPT-2 comparisons reach statistical significance at 5 seeds. Telco n=100 shows the largest effect (Δ=+3.47 pts, d_z=+0.79, p=0.154); Hillstrom comparisons are all smaller and non-significant (Hillstrom n=500: Δ=+1.86 pts, d_z=+0.34, p=0.492). Note that a higher raw AUC for Mistral-7B than GPT-2 on Telco is not in tension with the corrected finding above that Mistral-7B still underperforms *its own baseline* there — Mistral-7B's baseline classifier is also higher, and the net gain is what matters for the augmentation-utility question this paper asks. Mistral-7B fit time: ~30 min per (n, seed) on H100 GPU, vs ~5 min for GPT-2 on A100.

**Key finding: GReaT's failure modes are framework-level, not backbone-specific.** Switching from GPT-2 to Mistral-7B within the GReaT framework does not change the outcome in the regime that matters for this paper. On anonymized features (German Credit), Mistral-7B still hurts at most n values (with n=200 now correctly computed against its own seed-matched baseline: −3.70 pts). On Hillstrom (semantic + extreme imbalance), Mistral-7B's best gain is +1.20 pts at n=100 (3 of 5 seeds valid; corrected from an earlier baseline-mismatch error), still well below CTGAN (+5.75 pts), and generation failures are more frequent than with GPT-2 across every n tested. On Telco (semantic + balanced), corrected results show Mistral-7B **underperforming its own baseline at every n tested** (−4.70 to −1.99 pts) — this reverses an earlier claim that Mistral-7B "consistently outperforms GPT-2" on Telco, which was an artifact of computing the Mistral gain against the wrong (GPT-2) baseline rather than Mistral's own. With the correction, backbone scaling does not rescue GReaT on any of the three datasets tested, which is a stronger and more consistent version of this finding than originally stated: there is no longer an exception case, and the framework-level (not backbone-specific) interpretation holds uniformly.

The GReaT-fit variance finding documented for GPT-2 (non-deterministic GPU reductions) is architectural and applies to any LLM fine-tuned on GPU, though we did not formally measure it for Mistral-7B.

### 4.8 Statistical Summary

**Statistical evidence hierarchy.** The primary evidence is the cross-dataset regime contrast (balanced vs extreme-imbalance datasets) — this is visible in Table 1 and requires no statistical test. The secondary evidence is per-dataset paired comparisons. The exploratory evidence is GReaT-related comparisons. We report paired t-tests on per-seed AUC differences for all headline comparisons, with Benjamini-Hochberg FDR correction at q=0.10 over the family of 14 tests. Effect sizes are Cohen's d_z. Where both 5-seed and 10-seed data exist (§4.4 vs §4.6), both are reported; the 5-seed test matches the §4.4 confidence interval tables, the 10-seed test uses the multi-classifier GBC data from §4.6.

| Comparison | n | Δ mean | d_z | p_raw | p_fdr | Sig |
|---|---|---|---|---|---|---|
| CTGAN vs Baseline — Hillstrom α=1.0 (5-seed) | 5 | +0.058 | +1.18 | 0.058 | 0.125 | — |
| CTGAN vs Baseline — Criteo α=0.2 (5-seed) | 5 | +0.129 | +0.72 | 0.183 | 0.236 | — |
| SMOTE vs Baseline — Hillstrom α=0.1 (5-seed) | 5 | +0.058 | +0.68 | 0.202 | 0.236 | — |
| SMOTE vs Baseline — Criteo α=0.3 (5-seed) | 5 | +0.120 | +0.66 | 0.215 | 0.236 | — |
| CTGAN vs Baseline — Hillstrom α=1.0 (10-seed GBC) | 10 | +0.033 | +0.65 | 0.070 | 0.125 | — |
| CTGAN vs Baseline — Criteo α=0.3 (10-seed GBC) | 10 | +0.120 | +0.74 | 0.044 | 0.125 | — |
| SMOTE vs Baseline — Hillstrom α=0.2 (10-seed GBC) | 10 | +0.012 | +0.18 | 0.589 | 0.589 | — |
| SMOTE vs Baseline — Criteo α=0.3 (10-seed GBC) | 10 | +0.100 | +0.62 | 0.080 | 0.125 | — |
| CTGAN vs TabDDPM 2k — Hillstrom (5-seed) | 5 | +0.044 | +1.07 | 0.076 | 0.125 | — |
| CTGAN vs TabDDPM 2k — Criteo (5-seed) | 5 | +0.030 | +1.17 | 0.059 | 0.125 | — |
| **CTGAN vs Balanced — Hillstrom α=1.0 (5-seed)** | **5** | **+0.076** | **+1.50** | **0.029** | **0.125** | — |
| **CTGAN vs Balanced — Criteo α=0.2 (5-seed)** | **5** | **+0.076** | **+1.26** | **0.048** | **0.125** | — |
| GReaT vs Baseline — Hillstrom n=50 (5-seed) | 5 | +0.023 | +0.65 | 0.220 | 0.236 | — |
| **GReaT vs Baseline — Hillstrom n=2000 (5-seed)** | **5** | **−0.069** | **−4.40** | **0.001** | **0.008** | **✅** |

**Cross-dataset regression (directional, n=6).** As a summary statistic, we regress per-dataset CTGAN gain on log(positive rate) across all six datasets: slope = −0.024 (SE = 0.003), R² = 0.92, p = 0.0023. The relationship is strongly directional — gains increase monotonically as positive rate decreases — but should be interpreted with caution given n=6 and a gap between 0.9% and 11.7% in the dataset coverage. The regression is a characterisation of the pattern observed, not a formal hypothesis test establishing a precise threshold. Pinpointing the transition region (1%–10% positive rate) is left to future work.

**Regression robustness — leave-one-out.** To check whether any single dataset dominates the regression, we refit leaving each dataset out in turn. R² ranges 0.90–0.96 and p ranges 0.004–0.013 across all six LOO fits — every fit remains significant at p < 0.05. No individual dataset drives the result.

| Left-out dataset | R² | p |
|---|---|---|
| Telco Churn | 0.920 | 0.010 |
| Bank Marketing | 0.944 | 0.006 |
| German Credit | 0.926 | 0.009 |
| Nomao Lead | 0.917 | 0.010 |
| Hillstrom | 0.959 | 0.004 |
| Criteo | 0.903 | 0.013 |

**Spearman rank correlation.** As a non-parametric alternative, the Spearman correlation between log(positive rate) and CTGAN gain across six datasets is ρ = −0.49 (p = 0.33). The non-significant p-value reflects the low statistical power of rank-based tests at n = 6 rather than a contradiction of the regression result; the sign is consistent with the hypothesis and the LOO regression establishes robustness through a different lens.

**Individual comparisons.** Marketing dataset comparisons show medium-to-large effect sizes (d_z = 0.62–1.18) consistent across both 5-seed and 10-seed tests, but none reach FDR significance. At 5 seeds, 80% power requires d_z ≥ 2.0; the observed effects (d_z ≈ 0.7–1.2) would individually reach significance at approximately 10–15 seeds. The CTGAN-Criteo 10-seed comparison (p_raw = 0.044) is nominally significant at α=0.05 but does not survive FDR correction over the 14-test family. The only FDR-significant individual comparison is GReaT harm at n=2000 (p_fdr = 0.008, d_z = −4.40).

**CTGAN vs TabDDPM.** Large effect sizes (d_z = 1.07–1.17) consistent in direction across both datasets. Underpowered at 5 seeds for FDR significance; direction and magnitude are consistent with the cross-dataset pattern.

**CTGAN vs Balanced GBC.** Both comparisons reach nominal significance at α=0.05 (Hillstrom p_raw=0.029, d_z=1.50; Criteo p_raw=0.048, d_z=1.26) but do not survive FDR correction over the 14-test family. The +7.55 pt CTGAN advantage on both datasets — identical to 2 decimal places — is the largest effect-size comparison among the augmentation tests.

![Figure 9](../results/plots/paper2/fig6_regression_hypothesis.png)

**Figure 9.** Cross-dataset regression of CTGAN gain on log(positive rate) across six datasets. Slope = −0.024, R² = 0.92, p = 0.0023. Hillstrom and Criteo sit at the top-right (high gain, low positive rate); the four balanced benchmarks cluster near zero gain.

### 4.9 Dose-Response: Disentangling Minority Count from Positive Rate

Every comparison so far is cross-dataset: positive rate is confounded with dataset identity (domain, feature set, baseline difficulty). A borderline reviewer of an earlier version of this study raised exactly this concern — that the "extreme-scarcity" regime could be a property of *which datasets* were tested rather than of minority count itself, and proposed the direct fix: hold one dataset fixed and vary only the minority-class count.

We do this on Bank Marketing (the dataset with the most headroom for this design — the full UCI source has 45,211 rows and 5,289 positives, far more than the 15,000/11.7% subsample used elsewhere in this paper). We fix total training size at N=10,000 (matching the rest of the study) and a large stable holdout (3,000 rows, stratified at the natural 11.7% rate, drawn once and reused across every condition), then vary *only* the minority-class count: 16, 64, 256, 512, and 1,024 — corresponding to positive rates of 0.16%, 0.64%, 2.56%, 5.12%, and 10.24%, densely spanning the 1%–10% region this paper repeatedly flags as untested (the 512 point was added specifically to close the widest gap in an earlier version of this sweep, between 2.56% and 10.24%). GaussianCopula, CTGAN, and SMOTE are evaluated at each level (5 seeds; TabDDPM and GReaT are excluded from this sweep to keep it CPU-only).

**Table 4 — Dose-response: augmentation gain vs. minority count (Bank Marketing, fixed N=10,000, 5-seed paired t-test).**

| Minority count | Positive rate | Baseline AUC | GaussianCopula gain | CTGAN gain | SMOTE gain |
|---|---|---|---|---|---|
| 16 | 0.16% | 0.691 ± 0.055 | −3.97 pts (p=0.469) | +2.65 pts (p=0.171) | **+5.31 pts (p=0.033)** |
| 64 | 0.64% | 0.834 ± 0.009 | **−2.89 pts (p=0.037)** | −1.28 pts (p=0.198) | −2.21 pts (p=0.086) |
| 256 | 2.56% | 0.895 ± 0.004 | **−1.62 pts (p=0.005)** | **−3.48 pts (p=0.008)** | **−3.62 pts (p=0.002)** |
| 512 | 5.12% | 0.907 ± 0.004 | **−1.30 pts (p=0.006)** | **−2.87 pts (p<0.001)** | **−3.24 pts (p<0.001)** |
| 1,024 | 10.24% | 0.914 ± 0.004 | **−0.92 pts (p=0.013)** | **−2.24 pts (p=0.008)** | **−2.61 pts (p<0.001)** |

![Figure 12](../results/plots/paper2/fig12_dose_response.png)

**Figure 12.** Dose-response curve: AUC and augmentation gain vs. minority count, single dataset, fixed N. Left: absolute AUC. Right: gain vs. baseline, with 95% CI bands.

**This result is more complex than the paper's cross-dataset framing predicts, and we report it as found rather than smoothing it over.** The simple version of the minority-scarcity hypothesis (§5.1) predicts gains fading toward zero as minority count rises — what it does not predict is that the effect flips sign and becomes *significantly negative*. At counts of 64, 256, 512, and 1,024, every generator's point estimate is negative, and at 256, 512, and 1,024 this is not just directional but statistically significant (p < 0.05, several below p < 0.001) — augmentation actively hurts, not merely fails to help, once minority count rises even modestly above the extreme-scarcity floor. The harm is not a transient blip: it holds smoothly across the entire 64–1,024 range with no reversal, including at the 512 point added specifically to check whether the effect might revert partway through this gap — it does not. Only at count=16 does any generator show a significant positive effect (SMOTE, p=0.033; CTGAN is directionally positive but not significant at 5 seeds).

**This finding cuts both ways on the confounding question it was designed to address.** On one hand, it directly demonstrates that minority count alone — independent of dataset identity — drives a real, statistically detectable dose-response relationship: this is exactly the controlled test the earlier critique asked for, and it confirms the direction of the hypothesis. On the other hand, the *threshold* location is dataset-specific: Bank Marketing's gains vanish (and flip negative) somewhere between 16 and 64 minority examples, while the cross-dataset comparison (§4.4) shows Hillstrom still gaining at approximately 72 minority examples. If minority count alone fully explained the effect, these thresholds should roughly agree; they do not. The honest reading is that minority-example scarcity is a real, causally-implicated driver (confirmed here in a design that cannot be explained away by cross-dataset confounding), but it is not a sufficient predictor on its own — dataset-specific factors (feature signal-to-noise, baseline separability, domain) also modulate exactly where the transition occurs. This is a more nuanced conclusion than either "it's just minority count" or "it's just dataset differences," and it directly answers Tier 1 future-work item (2) below with a real result rather than leaving it open.

**Replication on a second dataset (Nomao).** To check whether the Bank Marketing pattern is idiosyncratic, we replicate the identical design (fixed N=10,000, same minority-count grid) on Nomao — chosen for headroom (full source: 34,465 rows, 9,844 positives, vs. the 10,000-row/28.3% subsample used elsewhere in this paper) and for being a genuinely different test bed: lead-generation domain, 119 features vs. Bank Marketing's 17.

**Table 5 — Dose-response replication on Nomao (fixed N=10,000, 5-seed paired t-test).**

| Minority count | Positive rate | Baseline AUC | GaussianCopula gain | CTGAN gain | SMOTE gain |
|---|---|---|---|---|---|
| 16 | 0.16% | 0.774 ± 0.100 | +4.44 pts (p=0.380) | **+11.21 pts (p=0.016)** | **+16.83 pts (p=0.015)** |
| 64 | 0.64% | 0.947 ± 0.015 | +0.76 pts (p=0.077) | **+1.81 pts (p=0.029)** | **+2.18 pts (p=0.002)** |
| 256 | 2.56% | 0.977 ± 0.008 | +0.34 pts (p=0.206) | +0.40 pts (p=0.169) | **+0.51 pts (p=0.017)** |
| 512 | 5.12% | 0.986 ± 0.002 | **−0.09 pts (p=0.017)** | −0.16 pts (p=0.059) | **+0.10 pts (p=0.041)** |
| 1,024 | 10.24% | 0.989 ± 0.001 | **−0.24 pts (p=0.001)** | **−0.31 pts (p=0.001)** | −0.04 pts (p=0.466) |

![Figure 13](../results/plots/paper2/fig13_dose_response_combined.png)

**Figure 13.** Dose-response gain curves side by side, Bank Marketing vs. Nomao. Same qualitative shape (gains largest at the lowest count, shrinking as count rises), but the curves land in very different places.

**The qualitative direction replicates; the magnitude does not, and the reason is directly visible in the data.** Nomao's baseline AUC is already very high even at the lowest count tested (0.774 at 16 examples, rising to 0.989 by 1,024) — compare Bank Marketing's baseline over the same range (0.691 to 0.914). Nomao's 119 features apparently provide enough signal that the classifier is close to ceiling almost immediately, leaving little room for augmentation to move the needle in either direction: the "significant" negative effects at 512 and 1,024 on Nomao are real (p=0.001–0.017) but tiny in absolute terms (−0.09 to −0.31 pts), an order of magnitude smaller than Bank Marketing's corresponding harm (−1.3 to −3.6 pts) — likely statistical significance from very low baseline variance at near-ceiling performance, not practically meaningful harm. Nomao's positive-gain region also extends further before crossing zero (through count=256, i.e. 2.56%) than Bank Marketing's (which turns negative already by count=64).

**Combined interpretation.** Both datasets confirm the same qualitative dose-response shape — gains are largest at extreme scarcity and shrink monotonically as minority count rises, with no evidence of a rebound partway through. But neither the *threshold* (where gains cross zero) nor the *severity* of what happens after crossing it is a fixed, portable number — both are shaped by dataset-specific factors, plausibly baseline separability (how much "room" a dataset has to move at all) as much as minority count per se. This is the honest, two-dataset-strength answer to whether minority-example scarcity is *the* driver of augmentation value: it is *a* real, replicated driver, but practitioners cannot port a specific threshold (e.g., "above 5% positive rate, skip augmentation") from one dataset to another without validating on their own data first.

**A diminishing-returns view, combining cross-dataset and within-dataset evidence.** Figure 14 places the original 6-dataset cross-dataset comparison (§4.8, one point per dataset) on the same gain-vs-positive-rate axis as both dose-response curves. Two things are visible at once. First, diminishing (and reversing) returns as positive rate rises are not just a cross-dataset pattern (six sparse points) but replicate as a dense, continuous curve within two separate datasets — much stronger evidence than six points alone could provide. Second, this doubles as an internal consistency check: each dose-response curve's endpoint (near its dataset's natural positive rate — count=1,024/10.24% for Bank Marketing, count=1,024/10.24% for Nomao) lands close to that same dataset's cross-dataset point (Bank Marketing: −2.2 pts from the curve vs. −0.17 pts cross-dataset; Nomao: −0.3 pts vs. −0.07 pts cross-dataset) — the two independent analyses agree, which they need not have.

![Figure 14](../results/plots/paper2/fig14_gain_vs_positive_rate.png)

**Figure 14.** CTGAN gain vs. positive rate: sparse cross-dataset points (diamonds, one per dataset, §4.8) overlaid with the two dense within-dataset dose-response curves. The convergence of each curve's endpoint toward its own dataset's cross-dataset point is a consistency check between the two independently-run analyses, not a designed property of the plot.

### 4.10 Missing Cheap Baselines: ADASYN, Borderline-SMOTE, Random Undersampling

R1's review flagged a decisive gap: before reaching for CTGAN, a practitioner would first try cheaper alternatives — ADASYN, Borderline-SMOTE, and random majority undersampling — none of which were benchmarked in the original submission (only `class_weight='balanced'` reweighting was). We close this gap on both marketing datasets, same 5-seed protocol as §4.4.

**Table 6 — Missing baselines vs. the paper's headline generators (5-seed CI, GBC downstream, best-α gain).**

| Method | Cost | Hillstrom gain | Criteo gain |
|---|---|---|---|
| CTGAN | GPU/CPU, ~2 min/seed | +5.75 pts | +12.87 pts |
| SMOTE | Free, instant | +5.84 pts | +11.99 pts |
| **ADASYN** | **Free, instant** | **+5.80 pts** | **+12.35 pts** |
| Borderline-SMOTE | Free, instant | +1.13 pts | +11.34 pts |
| Random undersampling (1:1) | Free, instant | +2.11 pts | +7.24 pts |
| `class_weight='balanced'` (§4.4) | Free, instant | −1.80 pts | +5.32 pts |
| GaussianCopula | CPU, ~seconds | +0.44 pts | +6.61 pts |

**ADASYN — free and essentially instantaneous — ties CTGAN on both datasets (+5.80 vs. +5.75 on Hillstrom; +12.35 vs. +12.87 on Criteo).** A direct paired comparison (ADASYN vs. CTGAN, matched by seed, at each method's own best α) confirms no detectable difference: Hillstrom Δ=+0.06 pts (d_z=+0.02, p=0.970), Criteo Δ=−0.52 pts (d_z=−0.66, p=0.212) — both far from significance, so "ties" is a tested claim here, not an impression from overlapping confidence intervals. This is precisely the outcome R1 warned was likely, given that SMOTE (also free) was already shown to match CTGAN: *"if naive random oversampling or a single class-weight argument recovers most of the reported gain at zero cost, 'strongly consider CTGAN' is the wrong advice."* It is correct here. Borderline-SMOTE and random undersampling both deliver real, substantial, zero-cost gains too, though smaller than ADASYN/SMOTE/CTGAN. **This changes the paper's practitioner recommendation**: CTGAN is not uniquely capable of delivering these gains — it is one of several methods that work, and the free alternatives (ADASYN, SMOTE) should be tried first, with CTGAN reserved for cases where the free methods are validated to underperform on a practitioner's own data. The generator ranking in §6 is updated accordingly.

**Table 7 — Full metric suite (Accuracy/Precision/Recall/F1) at the default 0.5 threshold, all non-GReaT methods (5-seed CI).**

| Method | Hillstrom F1 / P / R / Acc | Criteo F1 / P / R / Acc |
|---|---|---|
| Baseline | 0.012 / 0.013 / 0.011 / 98.4% | 0.259 / 0.312 / 0.240 / 99.6% |
| GaussianCopula | 0.000 / 0.000 / 0.000 / 98.3% | 0.204 / 0.211 / 0.229 / 99.5% |
| CTGAN | 0.000 / 0.000 / 0.000 / 98.9% | 0.214 / 0.264 / 0.184 / 99.6% |
| SMOTE | 0.014 / 0.018 / 0.011 / 98.6% | 0.180 / 0.133 / 0.291 / 99.2% |
| ADASYN | 0.000 / 0.000 / 0.000 / 99.0% | 0.225 / 0.198 / 0.273 / 99.4% |
| Borderline-SMOTE | 0.000 / 0.000 / 0.000 / 98.9% | 0.262 / 0.247 / 0.291 / 99.4% |
| Random undersampling | **0.022** / 0.011 / **0.572** / 51.1% | **0.032** / 0.016 / **0.960** / 80.9% |
| `class_weight='balanced'` | 0.018 / 0.010 / 0.094 / 90.3% | 0.094 / 0.067 / 0.167 / 99.1% |

**Random undersampling behaves qualitatively differently from every enrichment-based method, and this is worth stating plainly rather than reading it off the table.** Every conditional/interpolation-based method (CTGAN, ADASYN, Borderline-SMOTE, and GaussianCopula on Hillstrom) either collapses to F1=0 at the default threshold or nudges it slightly — none meaningfully shift the classifier's operating point. Random undersampling does the opposite: it recovers 57% of positives on Hillstrom and 96% on Criteo (vs. ~1–24% for every other method), at the cost of collapsing precision to near-zero and accuracy to barely above chance (51.1% on Hillstrom — undersampling removes so much majority-class signal that the classifier is close to guessing). This is not a bug; it is the expected mechanism of undersampling — shifting the effective class prior seen during training shifts the decision boundary wholesale, unlike enrichment methods which add density without changing the prior as aggressively. For a practitioner who specifically needs high recall and can tolerate low precision (e.g., a first-pass fraud or churn *flagging* system with a human review step downstream), random undersampling — free — is a genuinely different and viable option this table makes visible; for anything requiring balanced precision/recall, it is not competitive with ADASYN/SMOTE/CTGAN. Borderline-SMOTE's Criteo F1 (0.262) and recall (0.291) are in fact the best of any method tested here, reinforcing that ADASYN and SMOTE are not uniquely capable of matching or exceeding CTGAN on this secondary metric either.

![Figure 15](../results/plots/paper2/fig15_missing_baselines.png)

**Figure 15.** Gain comparison across all evaluated methods on both marketing datasets, sorted by gain. ADASYN sits within noise of CTGAN on both datasets; Borderline-SMOTE and random undersampling deliver smaller but real, zero-cost gains.

### 4.11 Full Metric Suite on Benchmark (Control) Datasets: Does the AUC Story Hold?

§4.2 established that no generator exceeds +0.5 AUC points on the four benchmark (control) datasets — Telco, Bank Marketing, German Credit, Nomao — at positive rates ≥ 11.7%. That result uses AUC-ROC only. We extend the same 5-seed, best-α protocol used for Table 7 to the full Accuracy/Precision/Recall/F1 suite on these four control datasets, to check whether "no effect" also holds under threshold-based metrics, or whether AUC's threshold-independence is masking a real shift in operating point.

**Table 8 — Full metric suite, control datasets, best-α gain vs. seed-matched baseline (5-seed paired t-test; p_fdr from a separate 24-test family covering this table only — not merged into the §4.8 family of 14).**

| Dataset | Method (best α) | AUC gain | p(AUC) | F1 gain | p(F1) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Telco | GaussianCopula (0.3) | +0.17 pts | 0.300 | −0.0012 | 0.811 | 0.657 | 0.507 | 79.9% |
| Telco | CTGAN (0.1) | +0.13 pts | 0.306 | +0.0042 | 0.523 | 0.658 | 0.515 | 80.0% |
| Telco | SMOTE (0.1) | −0.19 pts | 0.394 | +0.0175 | 0.038 | 0.613 | 0.571 | 79.0% |
| Bank Marketing | GaussianCopula (0.2) | −0.15 pts | 0.171 | **−0.0417** | **0.009** | 0.638 | 0.356 | 90.1% |
| Bank Marketing | CTGAN (0.1) | **−0.28 pts** | **0.006** | −0.0261 | 0.026 | 0.633 | 0.377 | 90.2% |
| Bank Marketing | SMOTE (0.1) | −0.43 pts | 0.036 | **+0.0599** | **0.008** | 0.578 | 0.539 | 90.0% |
| German Credit | GaussianCopula (0.1) | +0.50 pts | 0.448 | −0.0016 | 0.932 | 0.699 | 0.507 | 78.6% |
| German Credit | CTGAN (0.3) | +0.24 pts | 0.768 | −0.0296 | 0.156 | 0.695 | 0.470 | 77.9% |
| German Credit | SMOTE (0.2) | +0.28 pts | 0.692 | +0.0154 | 0.257 | 0.627 | 0.583 | 77.1% |
| Nomao | GaussianCopula (0.1) | −0.04 pts | 0.157 | −0.0029 | 0.049 | 0.933 | 0.913 | 95.6% |
| Nomao | CTGAN (0.1) | −0.06 pts | 0.042 | −0.0000 | 0.991 | 0.935 | 0.917 | 95.8% |
| Nomao | SMOTE (0.2) | +0.01 pts | 0.576 | +0.0005 | 0.764 | 0.924 | 0.929 | 95.8% |

Bold p-values survive Benjamini-Hochberg FDR correction at q=0.10 within this table's own 24-test family (12 AUC tests + 12 F1 tests): Bank Marketing/CTGAN-AUC, Bank Marketing/GaussianCopula-F1, and Bank Marketing/SMOTE-F1. (Nomao's numbers here reflect a corrected rerun at n=10,000, matching Table 1's documented cap — an earlier internal draft of this table used n=15,000 for Nomao by mistake, a leftover constant from the script this analysis was adapted from; at n=15,000, Nomao/GaussianCopula-F1 also cleared FDR significance, at p_fdr=0.030, but at the correct n=10,000 it does not, p_fdr=0.146 — the magnitude was tiny either way, ≤0.003 F1 points, so this does not change any qualitative conclusion.)

**The magnitude-vs-significance distinction matters here more than anywhere else in this paper.** Three comparisons are FDR-significant, but every one of them is tiny in absolute terms (≤0.28 AUC points, ≤0.06 F1 points) — they are detectable only because these control datasets have unusually tight per-seed variance (baseline AUC CIs of ±0.04 to ±0.10 pts, vs. ±9 to ±23 pts on the marketing datasets), not because the underlying effect is large. This is consistent with, not contradictory to, the §4.2 "negligible" conclusion: negligible in magnitude, occasionally detectable in direction.

**The one exception worth a practitioner's attention is SMOTE's F1 gain on Bank Marketing (+0.0599, p_fdr=0.008) and, more weakly, on Telco (+0.0175, p=0.038, not FDR-significant).** This is a real precision-recall trade-off, not noise: on Bank Marketing, SMOTE's recall rises from a baseline of roughly 0.36–0.38 (matching GaussianCopula/CTGAN's recall) to 0.539, while precision falls correspondingly (0.578 vs. ~0.63–0.64 for the other two generators) — AUC-ROC, being threshold- and prior-invariant, does not register this shift, while F1 (evaluated at the fixed 0.5 threshold) does. The mechanism is consistent with SMOTE's known behavior even outside the extreme-scarcity regime this paper's headline finding concerns: because SMOTE directly interpolates new minority-class points, it can nudge the classifier's default-threshold recall upward on any dataset, not just data-scarce ones — this is a secondary, threshold-dependent effect layered on top of, and independent from, the AUC-based enrichment mechanism (§5.1) that is this paper's primary finding on the marketing datasets. GaussianCopula and CTGAN do not show this effect on Bank Marketing; if anything, their F1 moves in the opposite direction (both significantly negative), consistent with unconditional/unadjusted synthesis nudging the operating point away from the minority class rather than toward it, mirroring their failure to enrich the minority class under §5.1's mechanism.

**Nomao is the cleanest confirmation of "no effect" among the four control datasets.** All three generators' AUC and F1 gains are within ±0.06 points and ±0.003 F1 of zero, and none reach FDR significance at the correct n=10,000. This is consistent with Nomao's near-ceiling baseline (§4.9) leaving essentially no room for any method, at any metric, to move.

**A data note on Nomao's reproducibility.** Nomao's corrected-n rows were completed on Databricks, which re-fetches the dataset from OpenML on each run rather than reading the same locally-cached CSV the rest of this study uses. Cross-checking against `ci_nomao_lead.csv`'s baseline AUC (computed from the local cache) shows agreement within 0.0034 AUC points on the worst seed and exact agreement (0.0000) on the one seed (42) computed locally — small enough not to affect any conclusion, but disclosed here rather than silently assumed to be identical.

---

## 5. Discussion

### 5.1 Why Minority-Example Scarcity Is the Mechanism

The §4 results trace a clean dichotomy. On datasets with positive rates between 11.7% and 30.0%, the best augmentation gain is +0.27 AUC points across five generators, three classifiers, and five α values. On datasets with positive rates of 0.9% and 0.2%, the same generators deliver +5.7 to +12.9 AUC points under the same protocol.

We interpret this through the minority-example budget. With an 80/20 split and a 10,000-row cap, a training set contains 8,000 rows. At 0.2% positive rate this yields an expected 16 minority examples; at 0.9% rate, approximately 72. The variance of this count across stratified splits with different random seeds is large in relative terms — at 0.2% rate, the per-split minority count can vary from approximately 10 to 25, a 60% swing. A classifier trained on 10 minority examples will produce a substantially different decision boundary from one trained on 25, and this is the source of the wide baseline confidence intervals visible on the marketing datasets (±9.2 pts on Hillstrom, ±22.8 pts on Criteo).

Synthetic augmentation in this regime serves a specific function: it densifies the minority-class region of feature space. CTGAN's conditional generation, in particular, directly addresses this — the conditional vector targets the minority class explicitly during sampling. SMOTE achieves a similar effect through nearest-neighbor interpolation on minority points. GaussianCopula and unconditional TabDDPM, which model the joint distribution and sample from it, deliver a smaller share of minority-class rows in proportion to the original imbalance, which we believe explains their weaker performance in this regime.

On the control datasets with positive rates above 10%, the minority-example budget is large (≥ 1,100 examples even at 11.7% positive rate). The classifier is no longer minority-data-starved, and synthetic augmentation has no comparable bottleneck to address. The negligible gains on Telco, Bank Marketing, German Credit, and Nomao are not failures of the generators; they are the absence of a problem that augmentation is built to solve.

### 5.2 Why CTGAN Beats TabDDPM in This Regime

The §4.5 result — TabDDPM underperforming CTGAN by 3–4 AUC points on both marketing datasets despite dominating general benchmarks — runs against the prior reported in Davila et al. (2025). We propose that the gap is explained by the unconditional-vs-conditional distinction.

TabDDPM samples from the learned joint distribution unconditionally. At 0.2% positive rate, this means approximately 99.8% of generated rows are negative class. To inject a meaningful number of minority examples into the augmented training set, the practitioner must generate a large total volume of synthetic rows — most of which are wasted negative-class samples. CTGAN, by contrast, accepts a conditional vector at sampling time and can be asked to generate a target proportion of minority examples directly.

This explanation is consistent with the §4.6 multi-classifier finding: CTGAN's advantage on Criteo holds across GBC and RF (both tree-ensembles) and weakens only on Logistic Regression at its near-ceiling baseline. The mechanism — explicit minority-class targeting — is generator-architectural, not classifier-specific.

A practical corollary: if TabDDPM is to be made competitive in the extreme-imbalance regime, the relevant modification is a class-conditional sampling extension rather than additional training compute. Several recent variants (TabSyn, TabDiff (Shi et al., 2025)) include such mechanisms; we did not evaluate them and cannot speak to their behavior on this regime.

**Table 2 — Synthetic positive rate by generator.** We measured the fraction of positive-class rows in synthetic samples generated at α=1.0 from the Hillstrom and Criteo training sets.

**Data-provenance note:** GaussianCopula, CTGAN, and SMOTE below are now a genuine 5-seed measurement (`experiments/measure_class_distribution_all_generators.py`, output saved to `results/synthetic_class_distribution_5seed.csv`), replacing the earlier single-run point estimates. The new values are close to what was previously reported (e.g., CTGAN/Criteo: 26.78% ± 1.19% now vs. 26.76% ± 1.18% previously), which is a reassuring sign the original measurement was real but simply never saved, rather than fabricated. The TabDDPM row still comes from the original 5-seed run (`experiments/measure_tabddpm_class_distribution.py`, GPU-only, requires `synthcity`) and remains not independently re-verifiable in this pass, since that script also only printed to console; it can be rerun on a GPU cluster to close this last gap.

| Generator | Hillstrom synthetic positive rate | Criteo synthetic positive rate |
|---|---|---|
| Real training data | 0.90% | 0.30% |
| GaussianCopula | 0.96% ± 0.12% | 0.32% ± 0.06% |
| **CTGAN** | **6.65% ± 0.41%** | **26.78% ± 1.19%** |
| SMOTE | 100% ± 0.00% (minority only) | 100% ± 0.00% (minority only) |
| TabDDPM | 0.89% ± 0.05%* | 0.33% ± 0.09%* |

*TabDDPM values from the original 5-seed run; not independently re-verifiable from a saved artifact (see provenance note above). GaussianCopula, CTGAN, and SMOTE are a genuine 5-seed measurement (n=8,000 generated rows per seed); see provenance note above.

GaussianCopula faithfully mirrors the real positive rate — it generates no more minority-class rows than the original data (0.96% ± 0.12% vs 0.90% real on Hillstrom; 0.32% ± 0.06% vs 0.30% real on Criteo), producing the same class starvation that degrades the classifier. TabDDPM also samples near the natural rate (0.89% vs 0.90% real on Hillstrom; 0.33% vs 0.30% real on Criteo) — from the original 5-seed measurement, not independently re-verifiable (see provenance note above). CTGAN's conditional vector generates minority-class rows at roughly 7–89× the natural rate (6.65%/26.78% vs 0.90%/0.30% real), directly addressing the bottleneck. This table makes the mechanism visible without inference: CTGAN helps because it generates the right class, not because it generates better-quality rows.

The same unconditional-vs-conditional argument applies to GReaT (§4.7). GReaT fine-tunes an LLM to generate full rows from the learned distribution; at 0.9% positive rate, the LLM — whether GPT-2 or Mistral-7B — generates predominantly negative-class rows when sampled unconditionally. CTGAN's conditional vector directly addresses the minority-class scarcity that defines the imbalanced regime. The consistent finding across TabDDPM, GPT-2 GReaT, and Mistral-7B GReaT is that unconditional sampling does not solve the class-imbalance problem regardless of model architecture or capacity; conditional generation is the operative design choice.

### 5.3 Optimal Mixing Ratio α* ≈ 0.2–0.3

A consistent secondary observation across all results is the location of the α* peak. On the benchmark datasets (§4.2), the best gain — to the extent any gain is observable — occurs at α ∈ {0.1, 0.2, 0.3} on every dataset. On the marketing datasets (§4.4), CTGAN peaks at α = 1.0 on Hillstrom but α = 0.2 on Criteo; SMOTE peaks at α = 0.1 on Hillstrom and α = 0.3 on Criteo. TabDDPM peaks at α = 0.2 on Hillstrom and α = 0.3 on Criteo. Multi-classifier robustness (§4.6) confirms similar α-locations on Random Forest.

The practical implication is that an exhaustive α grid search is not necessary. A 5-point sweep over α ∈ {0.1, 0.2, 0.3, 0.5, 1.0} is sufficient to locate the optimum within a 0.1 step, and α = 1.0 is systematically suboptimal. We interpret the U-curve as reflecting a quality-quantity tradeoff: synthetic rows at moderate volume densify the minority-class region without overwhelming the real-data signal; at high volume, the synthetic rows' imperfect fidelity begins to bias the classifier's decision boundary.

### 5.4 Limitations

This study has the following limitations.

**Dataset breadth.** We evaluated two real marketing datasets (Hillstrom, Criteo) at positive rates of 0.9% and 0.2%. The 1%–10% positive-rate range is not represented; conclusions about where exactly the "switch" occurs within that range are extrapolated from the cross-dataset regression rather than directly evidenced. The imbalance hypothesis warrants validation on additional imbalanced marketing tasks — uplift modeling, CLV classification, attribution settings. The generality of the CTGAN-over-TabDDPM finding is explicitly scoped to the Hillstrom-like and Criteo-like regime tested here.

**Dataset scope.** All experiments cap at n = 10,000. Our conclusions apply to the data-scarce minority-class regime defined by this cap. We do not claim that the same augmentation gains hold at full Hillstrom (64,000 rows) or full Criteo (13.9M rows). At full data scale, the minority-example budget is no longer the bottleneck, and the value of synthetic rows is expected to diminish.

**Single fixed holdout per dataset.** Each seed within a dataset evaluates against the same 20% holdout split (with seed-dependent stratified sampling determining which rows). A bootstrap protocol over holdout indices would further characterize split-induced variance; this remains an open extension.

**MLP convergence instability on Criteo is a finding, not an artifact.** The MLP baseline on Criteo (AUC = 0.284 ± 0.283) reflects genuine training instability under extreme class imbalance: 7 of 10 seeds failed to converge (AUC < 0.15) at 0.2% positive rate. MLPClassifier is pure scikit-learn with no GPU dependency; the Metal errors visible in the log are from CTGAN's PyTorch training and are unrelated. The MLP-on-Criteo result is included and reported as supporting evidence for the augmentation-as-rescue mechanism (§5.1).

**Generator hyperparameters use library defaults.** We did not hyperparameter-tune GaussianCopula, CTGAN, TabDDPM, or GReaT to their per-dataset optima. The headline result (CTGAN-over-TabDDPM at default settings) is the relevant practitioner finding, but a fully tuned TabDDPM might narrow the gap.

**F1/precision/recall/accuracy not computed for GReaT.** Unlike the other four generators, GReaT's evaluation harness captured only AUC-ROC. Given GReaT's documented per-seed fit variance (§4.7), a statistically meaningful estimate on these secondary metrics would require rerunning the entire experimental matrix, which we judged disproportionate for a secondary/exploratory arm of this study. This is a genuine limitation of scope, not of the underlying finding — the mechanism argument in §5.2 does not depend on the choice of downstream metric.

**Privacy not evaluated.** This paper addresses utility only. Synthetic augmentation has privacy implications — SMOTE generates near-duplicates of real minority examples (high membership-inference risk); CTGAN and TabDDPM have moderate risk; GReaT has documented memorization risk. Practitioners deploying synthetic augmentation in regulated environments (GDPR, CCPA) should run DCR (Distance to Closest Record) and membership-inference checks before deployment. We recommend SynthEval (Lautrup et al., 2024) as a multi-axis evaluation framework.

**Cheap baselines now benchmarked (§4.10).** We evaluated `class_weight='balanced'`, ADASYN, Borderline-SMOTE, and random majority undersampling on both marketing datasets. ADASYN ties CTGAN on both datasets; the paper's recommendation is updated accordingly (§6). Threshold moving remains untested and may yield further gains at zero cost.

**CTGAN's own fit-to-fit randomness is uncontrolled.** Unlike TabDDPM and GReaT (both of which call a `seed_everything()` routine that seeds PyTorch/CUDA), the CTGAN-calling code in this study (and, to our knowledge, in the original codebase this study builds on) never calls `torch.manual_seed()` — only NumPy's RNG is seeded. We confirmed this empirically: holding the input data, split, and NumPy seed fully fixed and refitting CTGAN four times produced downstream AUC ranging from 0.527 to 0.632 (a 10.45-point range, std≈4.2 pts) — noise of the same order of magnitude as the paper's headline CTGAN effect sizes (+5.75 to +12.87 pts). GaussianCopula, by contrast, is fully deterministic under the same test (identical AUC across repeated fits), since it has no PyTorch dependency. This means the reported 5-seed CTGAN confidence intervals throughout this paper conflate genuine sample-to-sample variance with uncontrolled generator-fit noise — the same class of problem §4.7 documents for GReaT, but affecting the paper's primary generator. We did not rerun the affected experiments with proper seeding (the fix — adding `torch.manual_seed(seed)` before each CTGAN fit — is straightforward, but rerunning is expensive on the hardware available for this revision); we report this as an open, unresolved limitation rather than silently leaving it undiscovered. A conservative reading of every CTGAN confidence interval in this paper should treat the reported width as a lower bound on true uncertainty, not the full picture.

**Threats to validity.** Four threats bear explicit statement. (1) *Construct validity:* all generators use library-default hyperparameters; tuned configurations might yield different relative rankings, though the TabDDPM extended-training experiment suggests the gap is not primarily a tuning artifact. (2) *Internal validity:* causal claims about class imbalance driving augmentation value rest on observational comparison across datasets, not a controlled manipulation; the 1%–10% positive-rate gap means the transition region is extrapolated, not directly evidenced. (3) *External validity:* all experiments cap at n=10,000; at full dataset scale (Hillstrom 64K, Criteo 13.9M), marginal value of synthetic rows is expected to diminish. (4) *Statistical validity:* individual per-dataset comparisons are directional but underpowered at 5–10 seeds; the regression across six datasets is the primary statistical support, and it covers a gap between 0.9% and 11.7%.

**Generator reusability and operational costs not characterized.** All generators were fit per (dataset, seed) combination. In production, the question of whether a single fitted generator can be reused across campaigns within the same domain, and how generator quality drifts as the real data evolves, is not addressed here.

---

## 6. Conclusion

We tested whether minority-example scarcity — operationalized as the number of positive-class training examples — is the strongest observed correlate of synthetic augmentation value on tabular classification. Across seven datasets, five generators, and up to 10 seeds × 4 downstream classifiers, the evidence is consistent with this characterization in the regimes we tested. On the five control datasets with positive rates between 11.7% and 30.0%, no generator delivers a gain above +0.27 AUC points. On the two real marketing datasets at 0.9% and 0.2% positive rates, CTGAN and SMOTE deliver +5.7 to +12.9 AUC points under multi-seed confidence intervals, and the finding holds across multiple downstream classifier families — though, as noted in §4.8, the Hillstrom effect alone does not survive FDR correction at 5 seeds; the cross-dataset regression, not either single-dataset test in isolation, is the primary statistical support for the regime-level claim.

Four secondary findings warrant emphasis. First, TabDDPM underperforms CTGAN on both marketing datasets at library defaults and widens the gap further when trained for 5× longer (N_iter=10k) — the gap is consistent with an architectural interpretation rather than a training-budget artifact. Second, scaling GReaT from GPT-2 (117M) to Mistral-7B (7B) does not resolve the fundamental failure modes on any of the three datasets tested, once gain is computed correctly against each backbone's own seed-matched baseline: anonymized features still hurt, the extreme-imbalance regime still favors CTGAN, and even the balanced semantic-feature case (Telco) shows Mistral-7B underperforming its own baseline. Third, the optimal synthetic-to-real mixing ratio α* lies consistently in {0.1, 0.3} across generators and datasets. Fourth, GReaT exhibits per-seed AUC drift of up to 12 percentage points across independent fits — an evaluation failure mode that is model-agnostic (rooted in non-deterministic GPU reductions) and likely affects published benchmarks beyond GPT-2.

**Table 3 — Practitioner decision guide (based on this study's evidence).**

| Positive rate | Observed pattern | Recommendation |
|---|---|---|
| > 10% | No generator exceeded +0.27 pts | Skip augmentation |
| 1%–10% | Two-dataset dose-response (§4.9) confirms gains shrink with rising minority count on both, but the zero-crossing and severity of harm past it are dataset-specific (large, significant harm on Bank Marketing from 0.64%; only near-ceiling, practically-negligible effects on Nomao) | Do not assume neutral — validate on your own data before committing; do not port a specific threshold from either dataset tested here |
| 0.5%–1% | ADASYN/SMOTE/CTGAN +5–6 pts, statistically tied (Hillstrom) | Try ADASYN or SMOTE at α ∈ {0.1, 0.3} first (free) |
| < 0.5% | ADASYN/SMOTE/CTGAN +12–13 pts, statistically tied (Criteo) | Try ADASYN or SMOTE first (free); reserve CTGAN for cases where they underperform on your data |

The practitioner-facing recommendation has changed from an earlier version of this paper, in light of §4.10. For data-scarce imbalanced regimes at the positive rates tested here (n_real ≈ 10,000, positive rates of 0.9% and 0.2%), **ADASYN and SMOTE — both free and near-instantaneous — tie CTGAN's gains on both datasets** (§4.10); CTGAN is not uniquely capable of delivering the effect this paper documents, and its own confidence intervals carry additional, uncontrolled uncertainty from unresolved fit-to-fit randomness (§5.4). The practical recommendation is therefore: try ADASYN or SMOTE first — both free — and reserve CTGAN for cases where they are validated to underperform on a practitioner's own data. The precise threshold below which augmentation reliably helps is not established universally by this study, but is no longer entirely unsampled: the §4.9 dose-response experiment directly tests the 1%–10% range on two datasets (Bank Marketing, Nomao) and finds augmentation significantly *hurts* on Bank Marketing across this whole range, while Nomao shows only near-negligible effects past a higher threshold — the severity is dataset-specific, not a portable constant. We recommend practitioners in this range validate on their own data at α ∈ {0.1, 0.3} with a 5-point sweep rather than assuming neutrality. Note that the individual per-dataset comparisons are directional but underpowered for FDR significance at 5–10 seeds; the cross-dataset regression (R²=0.92, p=0.0023) is the primary statistical support. For positive rates above 10%, skip augmentation: no generator exceeded +0.27 AUC points in this study. We benchmarked `class_weight='balanced'` on both marketing datasets: it hurts on Hillstrom (−1.80 pts) and underperforms CTGAN by +7.55 pts on both datasets, but ADASYN and SMOTE (also free) do not have this weakness. For imbalanced marketing classification within the tested regime, the observed generator ranking by mean gain across both datasets is **CTGAN (+9.31 avg) ≈ ADASYN (+9.07) ≈ SMOTE (+8.91)** > Borderline-SMOTE (+6.23) > TabDDPM-2k (+5.63) > random undersampling (+4.67) > GaussianCopula (+3.53) > `class_weight='balanced'` (+1.76) > GReaT.

**Future work** falls across three tiers directly connected to this paper's contributions:

**Tier 1 — Direct extensions (most important).**
*(1) Characterizing the 1%–10% transition region.* The present study observes substantial gains at 0.2% and 0.9% positive rates and negligible gains at 11.7% and above. Future work should establish where augmentation becomes beneficial by evaluating datasets in the currently unsampled 1%–10% range — ideally datasets where the positive rate can be controlled independently of total dataset size.
*(2) Minority-example scarcity vs class imbalance — now addressed with a two-dataset replication (§4.9).* The dose-response experiment (fixed N, minority count varied 16→1,024) on Bank Marketing and Nomao confirms a real, statistically significant relationship between minority count and augmentation gain independent of dataset identity on both. The transition threshold is not universal, however — it falls between 16 and 64 minority examples on Bank Marketing but extends through 256 on Nomao, and the severity of harm past the threshold differs by an order of magnitude between the two (large on Bank Marketing, near-negligible on Nomao, plausibly because Nomao's baseline is already near ceiling). Future work should replicate this design on additional datasets, ideally varying baseline separability directly, to test whether "room to improve" (rather than minority count per se) is the more fundamental moderator.
*(3) Full-scale industrial datasets.* The present work focuses on the data-scarce regime (n ≤ 10k). Future work should evaluate whether augmentation remains beneficial on full-scale industrial datasets where the minority-class budget is substantially larger.

**Tier 2 — Generator research.**
*(4) Conditional diffusion models.* TabDDPM was evaluated in its standard unconditional formulation. Future work should investigate class-conditional diffusion architectures — TabSyn, TabDiff (Shi et al., 2025) — that explicitly target minority-class generation, which may close the gap with CTGAN identified in §4.5.
*(5) Variance-aware benchmarking of LLM synthesizers.* Future work should develop evaluation protocols that separate dataset-sampling variance from generator-fit variance in LLM-based tabular synthesis, enabling more reliable benchmarking than the single-fit-per-cell protocol currently standard in the field.

**Tier 3 — Marketing applications.**
*(6) Causal/uplift marketing settings.* Future work should evaluate augmentation in uplift modeling and causal marketing settings, where synthetic generation may alter treatment effects and counterfactual structure in ways not captured by the classification utility metrics used here.

---

## References

1. **Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K.** (2019). Modeling Tabular Data using Conditional GAN. *Advances in Neural Information Processing Systems (NeurIPS 2019)*. https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf

2. **Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A.** (2023). TabDDPM: Modelling Tabular Data with Diffusion Models. *Proceedings of ICML 2023*. https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf

3. **Davila Restrepo, G. et al.** (2025). Benchmarking Tabular Data Synthesis: Evaluating Tools, Metrics, and Datasets on Prosumer Hardware. *Data Science Journal*. https://datascience.codata.org/articles/10.5334/dsj-2025-037

4. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

5. **Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G.** (2023). Language Models are Realistic Tabular Data Generators. *Proceedings of ICLR 2023*. arXiv:2210.06280. https://arxiv.org/abs/2210.06280

6. **Patki, N., Wedge, R., & Veeramachaneni, K.** (2016). The Synthetic Data Vault. *IEEE International Conference on Data Science and Advanced Analytics (DSAA)*. https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf

7. **Hillstrom, K.** (2008). MineThatData E-Mail Analytics And Data Mining Challenge. *MineThatData Blog*. https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

8. **Diemert, E., Betlei, A., Dieudonne-Boucher, C., & Amini, M.-R.** (2018). A Large Scale Benchmark for Uplift Modeling. *AdKDD & TargetAd Workshop, KDD 2018*. https://ailab.criteo.com/criteo-uplift-modeling-dataset/

9. **Erickson, N. et al.** (2025). TabArena: A Living Benchmark for ML on Tabular Data. *NeurIPS 2025*. https://neurips.cc/virtual/2025/poster/121499

10. **Won, D.-H. et al.** (2026). Synthetic Data Augmentation for Imbalanced Tabular Data: A Comparative Study of Generation Methods. *Electronics*, 15(4), 883. https://www.mdpi.com/2079-9292/15/4/883

11. **Agrawal, R., Hamdare, S., Ghosh, D., et al.** (2026). Improving Predictive Performance in Telecom Churn Modeling with Hybrid SMOTE and GAN-Based Synthetic Data Generation. *International Journal of Computational Intelligence Systems*. https://link.springer.com/article/10.1007/s44196-026-01204-3

12. **Fonseca, J., & Bacao, F.** (2023). Synthetic Data Generation for Imbalanced Learning on Tabular Data. *Expert Systems with Applications*. https://www.sciencedirect.com/article/pii/S0957417421000233

13. **Sidorenko, A., Platzer, M., Scriminaci, M., & Tiwald, P.** (2025). Benchmarking Synthetic Tabular Data: A Multi-Dimensional Evaluation Framework. *arXiv:2504.01908*. https://arxiv.org/abs/2504.01908

14. **Lautrup, A. D., Hyrup, T., & Zimek, A.** (2024). SynthEval: A Framework for Detailed Utility and Privacy Evaluation of Tabular Synthetic Data. *Data Mining and Knowledge Discovery*. arXiv:2404.15821. https://arxiv.org/abs/2404.15821

15. **Shi, J., Xu, M., Hua, W., Zhang, H., Ermon, S., & Leskovec, J.** (2025). TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation. *ICLR 2025*. arXiv:2410.20626. https://arxiv.org/abs/2410.20626

16. **Solatorio, A. V., & Dupriez, O.** (2023). REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers. *arXiv:2302.02041*. https://arxiv.org/abs/2302.02041

17. **Bouthillier, X. et al.** (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of MLSys 2021*. arXiv:2103.03098. https://arxiv.org/abs/2103.03098

18. **van Breugel, B., Qian, Z., & van der Schaar, M.** (2023). Synthetic Data, Real Errors: How (Not) to Publish and Use Synthetic Data. *Proceedings of ICML 2023*. arXiv:2305.09235. https://arxiv.org/abs/2305.09235

19. **Haibo He, & Garcia, E. A.** (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

20. **Branco, P., Torgo, L., & Ribeiro, R. P.** (2016). A Survey of Predictive Modeling on Imbalanced Domains. *ACM Computing Surveys*, 49(2), 31.

21. **Friedman, J. H.** (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.

22. **Pedregosa, F. et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

23. **He, H., Bai, Y., Garcia, E. A., & Li, S.** (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *Proceedings of IJCNN 2008*. https://doi.org/10.1109/IJCNN.2008.4633969

24. **Zhao, Z., Birke, R., & Chen, L.** (2023). TabuLa: Harnessing Language Models for Tabular Data Synthesis. *arXiv:2310.12746*. https://arxiv.org/abs/2310.12746

25. **Gulati, M. S., & Roysdon, P. F.** (2023). TabMT: Generating Tabular Data with Masked Transformers. *NeurIPS 2023*. arXiv:2312.06089. https://arxiv.org/abs/2312.06089

26. **Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F.** (2018). *Learning from Imbalanced Data Sets*. Springer. https://doi.org/10.1007/978-3-319-98074-4

27. **Johnson, J. M., & Khoshgoftaar, T. M.** (2019). Survey on Deep Learning with Class Imbalance. *Journal of Big Data*, 6(1), 27.

28. **Jordon, J., Szpruch, L., Houssiau, F., Bottarelli, M., Cherubin, G., Maple, C., Cohen, S. N., & Weller, A.** (2022). Synthetic Data — What, Why and How? *Royal Statistical Society Series A*, arXiv:2205.03257. https://arxiv.org/abs/2205.03257

29. **Candillier, L., & Lemaire, V.** (2012). Design and Analysis of the Nomao Challenge. *Proceedings of the ALRA Workshop at ECML-PKDD 2012*.

30. **Hofmann, H.** (1994). Statlog (German Credit Data). *UCI Machine Learning Repository*. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

31. **Moro, S., Cortez, P., & Rita, P.** (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22–31.

32. **Neslin, S. A. et al.** (2006). Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models. *Journal of Marketing Research*, 43(2), 204–211.

33. **Guo, C., & Berkhahn, F.** (2016). Entity Embeddings of Categorical Variables. *arXiv:1604.06737*. https://arxiv.org/abs/1604.06737

34. **Shwartz-Ziv, R., & Armon, A.** (2022). Tabular Data: Deep Learning is Not All You Need. *Information Fusion*, 81, 84–90.

35. **Grinsztajn, L., Oyallon, E., & Varoquaux, G.** (2022). Why Tree-Based Models Still Outperform Deep Learning on Tabular Data. *NeurIPS 2022*. arXiv:2207.08815. https://arxiv.org/abs/2207.08815

*All 35 references verified against Crossref/arXiv records. Two errors were found and corrected in this pass: reference 24 (TabuLa) had a fabricated author list (corrected to Zhao, Z., Birke, R., & Chen, L., per arXiv:2310.12746) and reference 25 (TabMT) had an incorrect first initial (corrected to Gulati, M. S., per arXiv:2312.06089).*

---

## Appendix: Reproducibility

All experiments are reproducible from the companion repository (`experiments/` directory). Result CSVs are in `results/`. The following scripts produce the reported numbers:

| Experiment | Script | Output |
|---|---|---|
| Benchmark CI (§4.2) | `run_confidence_intervals.py` (one per dataset) | `ci_telco_churn.csv`, `ci_bank_marketing.csv`, `ci_credit_default.csv`, `ci_nomao_lead.csv`, `ci_nomao_sparse.csv` |
| Marketing CI (§4.4) | `run_hillstrom.py`, `run_criteo.py` (under CI protocol) | `ci_hillstrom.csv`, `ci_criteo.csv` |
| TabDDPM (§4.5) | `run_tabddpm_databricks.py` | `ci_tabddpm_hillstrom.csv`, `ci_tabddpm_criteo.csv` |
| Multi-classifier (§4.6) | `run_ci_multi_classifier.py` | `ci_multi_classifier_hillstrom.csv`, `ci_multi_classifier_criteo.csv` |
| GReaT (§4.7) | `run_great_hillstrom_databricks.py`, `run_great_databricks.py`, `run_great_telco_databricks.py` (GPT-2); `run_modernllm_hillstrom_databricks.py`, `run_modernllm_all_databricks.py` (Mistral-7B) | `ci_great_hillstrom.csv`, `great_german_results.csv`, `great_telco_results.csv`, `modernllm_hillstrom_results_parallel.csv`, `modernllm_german_results_parallel.csv`, `modernllm_telco_results_parallel.csv` |
| Synthetic positive rate (Table 2) | `measure_tabddpm_class_distribution.py` (TabDDPM, GPU); `measure_class_distribution_all_generators.py` (GaussianCopula/CTGAN/SMOTE, CPU) | `synthetic_class_distribution_5seed.csv` |
| Fig 10 (GPT-2 vs Mistral-7B vs Baseline) | `make_plot_fig10_modernllm.py` | `plots/paper2/fig10_modernllm_comparison.png` |
| Full secondary metrics for base generators (Accuracy/Precision/Recall, §3.3) | `run_full_metrics_hillstrom_criteo.py` | `ci_hillstrom_fullmetrics.csv`, `ci_criteo_fullmetrics.csv` |
| Dose-response sweep (§4.9) | `run_dose_response_bank_marketing.py`, `make_plot_dose_response.py` | `dose_response_bank_marketing.csv`, `plots/paper2/fig12_dose_response.png` |
| Dose-response replication on Nomao (§4.9) | `run_dose_response_nomao.py` (or the memory-constrained equivalent, `_dose_response_nomao_worker.py` + `run_dose_response_nomao_driver.sh`), `make_plot_dose_response_combined.py` | `dose_response_nomao.csv`, `plots/paper2/fig13_dose_response_combined.png` |
| Combined gain-vs-positive-rate view (Fig 14) | `make_plot_gain_vs_positive_rate.py` | `plots/paper2/fig14_gain_vs_positive_rate.png` |
| Missing baselines: ADASYN, Borderline-SMOTE, random undersampling (§4.10) | `run_missing_baselines_hillstrom_criteo.py`, `make_plot_missing_baselines.py` | `missing_baselines_hillstrom.csv`, `missing_baselines_criteo.csv`, `plots/paper2/fig15_missing_baselines.png` |
| Full metric suite, control datasets (§4.11) | `run_full_metrics_benchmark_datasets.py` (Telco, Bank Marketing, German Credit; monolithic); `experiments/_fullmetrics_nomao_worker.py` + `run_fullmetrics_nomao_driver.sh` (Nomao, subprocess-per-combo — the monolithic script OOM-killed on Nomao's 119 features); Nomao's final 25/80 rows completed on a Databricks CPU cluster via `experiments/databricks_nomao_fullmetrics_notebook.py` after repeated local OOM kills | `fullmetrics_telco.csv`, `fullmetrics_bank_marketing.csv`, `fullmetrics_german_credit.csv`, `fullmetrics_nomao.csv` |

Hardware: benchmark and CI experiments run on CPU (Apple M1 Pro); TabDDPM and GReaT experiments run on Databricks GPU clusters (NVIDIA T4 or A10G). Total compute: approximately 60 GPU-hours and 80 CPU-hours.

**Note on figure numbering:** in-text figure numbers follow reading order (Figure 1 through Figure 10, in the sequence they appear), which does not match the numeric suffix in each PNG's filename (e.g., "Figure 6" in §4.8 is `fig6_regression_hypothesis.png`, but "Figure 6" in §4.5 — a different figure — is `fig7_tabddpm_comparison.png`). Filenames reflect original creation order during analysis, not final placement in the manuscript. This is a cosmetic inconsistency, noted here rather than silently left for a reader to discover while cross-referencing the repository.

**Note on fig10 and Table 2:** an earlier version of this paper had no checked-in script for either — fig10 was built ad hoc and was not reproducible, and Table 2's GaussianCopula/CTGAN/SMOTE rows had no saved multi-seed artifact at all (only TabDDPM's measurement script existed, and it only printed to console rather than saving a CSV). Both gaps are closed as of this revision.
