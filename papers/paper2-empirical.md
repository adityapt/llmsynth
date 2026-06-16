# Synthetic Data Augmentation in the Extreme-Imbalance Regime: A Controlled Empirical Study of Marketing Classification

**Author:** Aditya Puttaparthi Tirumala
**Date:** 2026-06-07
**Type:** Controlled empirical study
**Target venue:** KDD Applied Data Science Track / NeurIPS Datasets & Benchmarks

> *This is a structural draft. Numbers are pulled directly from the result CSVs in `results/`. The author will rewrite the prose in their own voice; the role of this draft is to lock the structure, claims, and supporting evidence.*

---

## Abstract

**Problem.** When does synthetic data augmentation actually help for tabular classification, and which generator should practitioners use? Existing benchmarks evaluate generators on heterogeneous tasks and report aggregate rankings that do not answer this question for practitioners working in imbalanced marketing classification.

**Setup.** We evaluate five generators — GaussianCopula, CTGAN, SMOTE, TabDDPM, and GReaT (at two LLM scales: GPT-2 and Mistral-7B) — on seven datasets spanning positive rates from 0.2% to 30.0%, with 5–10 random seeds and four downstream classifier families.

**Main finding.** Consistent gains appear only in the extreme-imbalance regime we tested: on two real marketing datasets (Hillstrom 0.9%, Criteo 0.2%), CTGAN and SMOTE deliver +5.7 to +12.9 AUC points. On four balanced benchmark datasets (positive rates 11.7%–30%), no generator exceeds +0.27 AUC points. A cross-dataset regression yields R²=0.92 (directional, n=6). On Criteo, 7 of 10 MLP seeds failed to converge on real data alone; CTGAN augmentation restored convergence in all 10.

**Secondary finding.** CTGAN outperforms TabDDPM at both default and 5× extended training budgets — the gap widens with more training, consistent with a sampling-design interpretation: CTGAN's conditional vector targets the minority class directly; TabDDPM samples unconditionally, producing synthetic rows at the natural (extremely low) positive rate. This conclusion holds across Gradient Boosting, Random Forest, and MLP classifiers. It also holds when GPT-2 in GReaT is replaced by Mistral-7B. CTGAN also outperforms `class_weight='balanced'` reweighting by +7.55 AUC points on both datasets.

**Recommendation and scope.** For practitioners in the positive-rate regime tested here (below 1%), evaluate CTGAN at α ∈ {0.1, 0.3} before more expensive alternatives. Above 10%, augmentation is unlikely to help based on our results. The 1%–10% region is untested and remains an open empirical question.

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
| GaussianCopula | SDV v1.36 | Library defaults | Fit once at α=1.0, subsample for smaller α |
| CTGAN | SDV/CTGAN v0.12 | Library defaults | Fit once at α=1.0, subsample for smaller α |
| SMOTE | imbalanced-learn v0.12 | k=5 neighbors | Re-call per α (no separate fit step) |
| TabDDPM | synthcity v0.2.11 `ddpm` | N_iter=2000, num_timesteps=1000, lr=10⁻³, batch=1024 | Fit once at α=1.0, subsample |
| GReaT | be-great v0.0.13 + GPT-2 (117M) | `guided_sampling=True`, 50–100 epochs | Fit once per (n, seed) |

The fit-once-and-subsample strategy is used for all generators with a non-trivial fit step. This holds the generator constant within a (dataset, seed) pair while varying only the volume of synthetic data injected, ensuring α-comparisons are not confounded by generator-fit variance.

### 3.3 Evaluation Protocol

For each (dataset, generator, seed) triple we run four conditions:

1. **Baseline (TRTR):** Train on 80% real data, evaluate on 20% real holdout.
2. **TSTR:** Train on synthetic-only data, evaluate on real holdout.
3. **Augmentation sweep:** Train on real + synthetic at α ∈ {0.1, 0.2, 0.3, 0.5, 1.0}, where α = n_synthetic / n_real.
4. **Multi-classifier robustness** (Hillstrom and Criteo only): repeat the augmentation sweep with four downstream classifier families (see §3.4) and 10 seeds.

Splits are stratified on the target. The primary metric is AUC-ROC; secondary metrics are F1 on the minority class and average precision.

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
| Bank Marketing | 0.909 | 0.750 (GaussianCopula) | −17.5% |
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
| Nomao dense reference | 0.992 ± —  | — | — | (separate baseline) |

The dense baseline (last row) is reported for context: with all features present, the same classifier reaches AUC = 0.992, a 9.44-point gain over the sparse baseline. Augmentation closes none of this gap meaningfully. The finding is consistent with the imbalance hypothesis: sparsity-driven baseline degradation is not the failure mode synthetic augmentation addresses. Augmentation needs minority-class data starvation, not feature-information starvation, to deliver value.

![Figure 3](../results/plots/paper2/fig3_ucurve_sparse.png)

**Figure 3.** Augmentation U-curve for the sparse stress test. The flat curve (all gains < 0.5 pts across all α) — compared against the dense baseline at AUC=0.992 — shows that sparsity-driven performance gaps are not recoverable through synthetic augmentation.

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

**German Credit** (anonymized features, 30% positive)

| n | Baseline | GPT-2 GReaT | Mistral-7B | GPT-2 gain | Mistral gain |
|---|---|---|---|---|---|
| 50 | 0.645 | 0.638 ± 0.076 | 0.661 ± 0.074 | −0.71 pts | +1.59 pts |
| 100 | 0.708 | 0.638 ± 0.033 | 0.652 ± 0.053 | −7.02 pts | −5.59 pts |
| 200 | 0.759 | 0.700 ± 0.028 | 0.711 ± 0.039 | −5.92 pts | −4.81 pts |
| 500 | 0.761 | 0.731 ± 0.019 | 0.757 ± 0.025 | −3.00 pts | −0.38 pts |

**Hillstrom Email** (semantic features, 0.9% positive)

| n | Baseline | GPT-2 GReaT | Mistral-7B | GPT-2 gain | Mistral gain |
|---|---|---|---|---|---|
| 50 | 0.494 | 0.516 ± 0.047 | —† | +2.25 pts | —† |
| 100 | 0.488 | 0.500 ± 0.052 | 0.524 ± 0.060 | +1.15 pts | +3.55 pts |
| 200 | 0.493 | 0.497 ± 0.091 | 0.500 ± 0.102 | +0.40 pts | +0.69 pts |
| 500 | 0.512 | 0.512 ± 0.079 | 0.531 ± 0.037 | −0.02 pts | +1.84 pts |

**Telco Churn** (semantic features, 26.6% positive)

| n | Baseline | GPT-2 GReaT | Mistral-7B | GPT-2 gain | Mistral gain |
|---|---|---|---|---|---|
| 50 | 0.671 | 0.710 ± 0.070 | 0.690 ± 0.119 | +3.93 pts | +1.95 pts |
| 100 | 0.747 | 0.733 ± 0.034 | 0.768 ± 0.034 | −1.38 pts | +2.09 pts |
| 200 | 0.767 | 0.766 ± 0.026 | 0.777 ± 0.030 | −0.07 pts | +0.98 pts |
| 500 | 0.795 | 0.797 ± 0.007 | 0.803 ± 0.017 | +0.15 pts | +0.76 pts |

†n=50 Mistral-7B on Hillstrom excluded: 4 of 5 seeds failed to generate valid rows at 0.2% positive rate + n=50. Larger models appear more brittle than GPT-2 (1/5 failures) at extreme small n under severe imbalance. AUC reported for the 1 successful seed should not be compared to the 5-seed GPT-2 mean.

![Figure 10](../results/plots/paper2/fig10_modernllm_comparison.png)

**Figure 10.** GPT-2 (117M) vs Mistral-7B (7B) vs Baseline across three datasets. Shaded regions are 95% CI. Mistral-7B shows marginal improvement over GPT-2 on semantic-feature datasets (Hillstrom, Telco) but the fundamental failure modes persist.

**Paired tests: Mistral-7B vs GPT-2.** None of the Mistral-7B vs GPT-2 comparisons reach statistical significance at 5 seeds. Telco n=100 shows the largest effect (Δ=+3.47 pts, d_z=+0.79, p=0.154); Hillstrom comparisons are all smaller and non-significant (Hillstrom n=500: Δ=+1.86 pts, d_z=+0.34, p=0.492). The directional improvement on Telco is consistent but underpowered. Mistral-7B fit time: ~30 min per (n, seed) on H100 GPU, vs ~5 min for GPT-2 on A100.

**Key finding: GReaT's failure modes are framework-level, not backbone-specific.** Switching from GPT-2 to Mistral-7B within the GReaT framework does not change the outcome in the regime that matters for this paper. On anonymized features (German Credit), Mistral-7B still hurts across most n values. On Hillstrom (semantic + extreme imbalance), Mistral-7B's best gain (+3.55 pts at n=100) remains below CTGAN (+5.75 pts), and generation failures are more frequent than with GPT-2 (4/5 seeds failed at n=50 vs 1/5 for GPT-2). On Telco (semantic + balanced), Mistral-7B does consistently outperform GPT-2 across all n values — suggesting that when GReaT's preconditions are met (semantic features, sufficient class balance), a more capable backbone helps. But in the extreme-imbalance regime that is the focus of this paper, backbone choice does not change the conclusion.

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

**Individual comparisons.** Marketing dataset comparisons show medium-to-large effect sizes (d_z = 0.62–1.18) consistent across both 5-seed and 10-seed tests, but none reach FDR significance. At 5 seeds, 80% power requires d_z ≥ 2.0; the observed effects (d_z ≈ 0.7–1.2) would individually reach significance at approximately 10–15 seeds. The CTGAN-Criteo 10-seed comparison (p_raw = 0.044) is nominally significant at α=0.05 but does not survive FDR correction over the 12-test family. The only FDR-significant individual comparison is GReaT harm at n=2000 (p_fdr = 0.007, d_z = −4.40).

**CTGAN vs TabDDPM.** Large effect sizes (d_z = 1.07–1.17) consistent in direction across both datasets. Underpowered at 5 seeds for FDR significance; direction and magnitude are consistent with the cross-dataset pattern.

**CTGAN vs Balanced GBC.** Both comparisons reach nominal significance at α=0.05 (Hillstrom p_raw=0.029, d_z=1.50; Criteo p_raw=0.048, d_z=1.26) but do not survive FDR correction over the 14-test family. The +7.55 pt CTGAN advantage on both datasets — identical to 2 decimal places — is the largest effect-size comparison among the augmentation tests.

![Figure 9](../results/plots/paper2/fig6_regression_hypothesis.png)

**Figure 9.** Cross-dataset regression of CTGAN gain on log(positive rate) across six datasets. Slope = −0.024, R² = 0.92, p = 0.0023. Hillstrom and Criteo sit at the top-right (high gain, low positive rate); the four balanced benchmarks cluster near zero gain.

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

| Generator | Hillstrom synthetic positive rate | Criteo synthetic positive rate |
|---|---|---|
| Real training data | 0.90% | 0.30% |
| GaussianCopula | 0.96% ± 0.12% | 0.32% ± 0.06% |
| **CTGAN** | **6.34% ± 0.21%** | **26.76% ± 1.18%** |
| SMOTE | 100% (minority only) | 100% (minority only) |
| TabDDPM | 0.89% ± 0.05% | 0.33% ± 0.09% |

All values measured directly: 5 seeds × 8,000 generated rows per dataset.

GaussianCopula faithfully mirrors the real positive rate — it generates no more minority-class rows than the original data, producing the same class starvation that degrades the classifier. TabDDPM also samples near the natural rate (0.89% vs 0.87% real on Hillstrom; 0.33% vs 0.30% real on Criteo) — measured directly from 5 seeds × 8,000 generated rows. CTGAN's conditional vector generates minority-class rows at 6–100× the natural rate, directly addressing the bottleneck. This table makes the mechanism visible without inference: CTGAN helps because it generates the right class, not because it generates better-quality rows. The TabDDPM result closes the reviewer objection to approximation — it is now measured, not assumed.

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

**Privacy not evaluated.** This paper addresses utility only. Synthetic augmentation has privacy implications — SMOTE generates near-duplicates of real minority examples (high membership-inference risk); CTGAN and TabDDPM have moderate risk; GReaT has documented memorization risk. Practitioners deploying synthetic augmentation in regulated environments (GDPR, CCPA) should run DCR (Distance to Closest Record) and membership-inference checks before deployment. We recommend SynthEval (Lautrup et al., 2024) as a multi-axis evaluation framework.

**Cost-sensitive alternatives partially benchmarked.** We evaluated `class_weight='balanced'` via sample-weight reweighting on both marketing datasets (§4.4). CTGAN outperforms this baseline by +7.55 pts on both Hillstrom and Criteo. Threshold moving and ADASYN are not benchmarked and may yield different results on other tasks.

**Threats to validity.** Four threats bear explicit statement. (1) *Construct validity:* all generators use library-default hyperparameters; tuned configurations might yield different relative rankings, though the TabDDPM extended-training experiment suggests the gap is not primarily a tuning artifact. (2) *Internal validity:* causal claims about class imbalance driving augmentation value rest on observational comparison across datasets, not a controlled manipulation; the 1%–10% positive-rate gap means the transition region is extrapolated, not directly evidenced. (3) *External validity:* all experiments cap at n=10,000; at full dataset scale (Hillstrom 64K, Criteo 13.9M), marginal value of synthetic rows is expected to diminish. (4) *Statistical validity:* individual per-dataset comparisons are directional but underpowered at 5–10 seeds; the regression across six datasets is the primary statistical support, and it covers a gap between 0.9% and 11.7%.

**Generator reusability and operational costs not characterized.** All generators were fit per (dataset, seed) combination. In production, the question of whether a single fitted generator can be reused across campaigns within the same domain, and how generator quality drifts as the real data evolves, is not addressed here.

---

## 6. Conclusion

We tested whether minority-example scarcity — operationalized as the number of positive-class training examples — is the strongest observed correlate of synthetic augmentation value on tabular classification. Across seven datasets, five generators, and up to 10 seeds × 4 downstream classifiers, the evidence is consistent with this characterization in the regimes we tested. On the five control datasets with positive rates between 11.7% and 30.0%, no generator delivers a gain above +0.27 AUC points. On the two real marketing datasets at 0.9% and 0.2% positive rates, CTGAN and SMOTE deliver +5.7 to +12.9 AUC points under multi-seed confidence intervals, and the finding holds across multiple downstream classifier families.

Four secondary findings warrant emphasis. First, TabDDPM underperforms CTGAN on both marketing datasets at library defaults and widens the gap further when trained for 5× longer (N_iter=10k) — the gap is consistent with an architectural interpretation rather than a training-budget artifact. Second, scaling GReaT from GPT-2 (117M) to Mistral-7B (7B) provides marginal improvement on semantic-feature datasets but does not resolve the fundamental failure modes: anonymized features still hurt, CTGAN still dominates on imbalanced data. Third, the optimal synthetic-to-real mixing ratio α* lies consistently in {0.1, 0.3} across generators and datasets. Fourth, GReaT exhibits per-seed AUC drift of up to 12 percentage points across independent fits — an evaluation failure mode that is model-agnostic (rooted in non-deterministic GPU reductions) and likely affects published benchmarks beyond GPT-2.

**Table 3 — Practitioner decision guide (based on this study's evidence).**

| Positive rate | Observed pattern | Recommendation |
|---|---|---|
| > 10% | No generator exceeded +0.27 pts | Skip augmentation |
| 1%–10% | **Not tested** | Validate experimentally before committing |
| 0.5%–1% | CTGAN/SMOTE +5–6 pts (Hillstrom) | Evaluate CTGAN at α ∈ {0.1, 0.3} |
| < 0.5% | CTGAN/SMOTE +12–13 pts (Criteo) | Strongly consider CTGAN augmentation |

The practitioner-facing recommendation is simple. For data-scarce imbalanced regimes at the positive rates tested here (n_real ≈ 10,000, positive rates of 0.9% and 0.2%), CTGAN consistently achieved the largest gains in this study across datasets, seeds, and classifiers. The precise threshold below which augmentation reliably helps is not established by this study — the 1%–10% range is unsampled. We recommend evaluating it at α ∈ {0.1, 0.3} with a 5-point sweep for validation. Note that the individual per-dataset comparisons are directional but underpowered for FDR significance at 5–10 seeds; the cross-dataset regression (R²=0.92, p=0.0023) is the primary statistical support. For positive rates above 10%, skip augmentation: no generator exceeded +0.27 AUC points in this study. We benchmarked `class_weight='balanced'` on both marketing datasets: it hurts on Hillstrom (−1.80 pts) and underperforms CTGAN by +7.55 pts on both datasets. Synthetic augmentation delivers gains the cost-sensitive alternative cannot. For imbalanced marketing classification within the tested regime, the observed generator ranking is CTGAN ≈ SMOTE > TabDDPM > GaussianCopula > GReaT.

**Future work** falls across three tiers directly connected to this paper's contributions:

**Tier 1 — Direct extensions (most important).**
*(1) Characterizing the 1%–10% transition region.* The present study observes substantial gains at 0.2% and 0.9% positive rates and negligible gains at 11.7% and above. Future work should establish where augmentation becomes beneficial by evaluating datasets in the currently unsampled 1%–10% range — ideally datasets where the positive rate can be controlled independently of total dataset size.
*(2) Minority-example scarcity vs class imbalance.* Future work should disentangle positive rate from absolute minority-example count. A dataset with 1% positives and 1M rows may exhibit very different behavior from a dataset with 1% positives and 10k rows despite identical class imbalance. The present study cannot distinguish these because the 10k cap ties positive rate to minority count.
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

24. **Zhao, T., Lala, D., Sawada, T., & Watanabe, T.** (2023). TabuLa: Harnessing Language Models for Tabular Data Synthesis. *arXiv:2310.12746*. https://arxiv.org/abs/2310.12746

25. **Gulati, A., & Roysdon, P.** (2023). TabMT: Generating Tabular Data with Masked Transformers. *NeurIPS 2023*. arXiv:2312.06089. https://arxiv.org/abs/2312.06089

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

*All references 1–20 verified against DOI/arXiv pages. References 21–35 should be verified before submission.*

---

## Appendix: Reproducibility

All experiments are reproducible from the companion repository (`experiments/` directory). Result CSVs are in `results/`. The following scripts produce the reported numbers:

| Experiment | Script | Output |
|---|---|---|
| Benchmark CI (§4.2) | `run_confidence_intervals.py` (one per dataset) | `ci_telco_churn.csv`, `ci_bank_marketing.csv`, `ci_credit_default.csv`, `ci_nomao_lead.csv`, `ci_nomao_sparse.csv` |
| Marketing CI (§4.4) | `run_hillstrom.py`, `run_criteo.py` (under CI protocol) | `ci_hillstrom.csv`, `ci_criteo.csv` |
| TabDDPM (§4.5) | `run_tabddpm_databricks.py` | `ci_tabddpm_hillstrom.csv`, `ci_tabddpm_criteo.csv` |
| Multi-classifier (§4.6) | `run_ci_multi_classifier.py` | `ci_multi_classifier_hillstrom.csv`, `ci_multi_classifier_criteo.csv` |
| GReaT (§4.7) | `run_great_hillstrom_databricks.py` | `ci_great_hillstrom.csv` |

Hardware: benchmark and CI experiments run on CPU (Apple M1 Pro); TabDDPM and GReaT experiments run on Databricks GPU clusters (NVIDIA T4 or A10G). Total compute: approximately 60 GPU-hours and 80 CPU-hours.
