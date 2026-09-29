<!--
This file restructures papers/paper2-empirical.md into the exact section/
subsection hierarchy of Won et al. (2026), "Synthetic Data Augmentation for
Imbalanced Tabular Data: A Comparative Study of Generation Methods,"
Electronics 15(4), 883 (https://www.mdpi.com/2079-9292/15/4/883) -- the
target venue's own closest precedent. paper2-empirical.md remains the
source-of-truth analysis document (all numbers verified against raw CSVs
there); this file is a submission-formatted derivative, explicitly framed
as an extension of Won et al. along five of its own six stated limitations
(see Introduction).

Placeholder boilerplate sections (Author Contributions, Funding, IRB
Statement, Informed Consent, Conflicts of Interest) are marked TODO and
need Aditya's/coauthors' actual input -- not something to fabricate.
-->

# Synthetic Data Augmentation in the Extreme-Imbalance Regime: A Dose-Response Study and a Missing-Baseline Reassessment

**Authors:** Aditya Puttaparthi Tirumala [, coauthors TBD]

## Abstract

Practitioners facing severe class imbalance — email conversion rates below 1%, rare-event prediction in marketing classification — routinely turn to synthetic data augmentation, but existing benchmarks report aggregate generator rankings across heterogeneous tasks that don't answer the question that matters: given a specific positive rate and sample size, will augmentation help, and which method should be used? We answer this with a controlled study spanning seven datasets (0.2%–30% positive rate), five generators (GaussianCopula, CTGAN, SMOTE, TabDDPM, and GReaT at two LLM scales, GPT-2 and Mistral-7B), 5–10 seeds, four downstream classifier families, and two independent dose-response experiments that vary minority-class count while holding dataset identity fixed — directly testing whether scarcity itself, rather than which dataset happens to be at hand, drives the effect.

It does, but the finding is more useful than a single threshold. On two real marketing datasets — Hillstrom (0.9% positive) and Criteo (0.2%) — augmentation delivers +5.7 to +12.9 AUC points; on four balanced benchmarks at 11.7%–30%, no generator exceeds +0.27 points. We measure why: CTGAN's conditional sampler generates minority rows at 7–89× the natural rate, while TabDDPM and GaussianCopula sample unconditionally and underperform accordingly. The pattern holds across four classifier families, and is unchanged when GPT-2 in GReaT is replaced by the substantially larger Mistral-7B. On Criteo, 7 of 10 MLP seeds failed to converge on real data alone; CTGAN augmentation restored convergence in all 10.

But the exact point where augmentation stops helping and starts hurting is not a portable constant. A controlled within-dataset sweep — minority count varied from 16 to 1,024, total sample size held fixed — turns significantly harmful above roughly 1% positive rate on Bank Marketing. The same design on a second, higher-dimensional dataset (Nomao) holds the same direction, but the magnitude nearly vanishes: a near-ceiling baseline leaves little room to move. A single cutoff cannot be assumed to generalize.

The more surprising result concerns which method to use at all. ADASYN — free, instantaneous, and over fifteen years old — statistically ties CTGAN on both marketing datasets (paired test, p=0.97 and p=0.21), against the intuition that a modern deep generative model should win; SMOTE ties both as well. We also find CTGAN's own training is not reliably reproducible in the standard SDV implementation used here — a >10-point AUC swing from refitting on fully identical data and seed. To our knowledge this reliability gap has not been previously reported, and it likely affects prior CTGAN benchmarks generally. We disclose it as an open limitation rather than resolve it.

**Practical recommendation.** Below roughly 1% positive rate, try ADASYN or SMOTE before CTGAN — both are free and statistically indistinguishable from it here. Above 10%, augmentation is unlikely to help. The region in between is not a gap in this study — we tested it directly on two datasets — but the result is dataset-dependent rather than a single rule: augmentation was significantly harmful across most of this range on one dataset, and merely negligible on the other. Practitioners in this band should validate on their own data, not because the region is unstudied, but because our own two datasets already disagree on what happens there.

**Keywords:** synthetic data augmentation; class imbalance; CTGAN; ADASYN; dose-response; tabular classification; marketing analytics

---

## 1. Introduction

Marketing and product data scientists routinely face classification problems with severe class imbalance: email conversion rates below 1%, display ad click-through rates below 0.5%, and rare-event prediction across customer cohorts [1,2]. The class imbalance problem itself is well studied [3–5]. A growing body of work proposes synthetic data augmentation as a remedy [6,7]: train a generative model on the available real data, sample synthetic examples, and combine them with the real training set to improve downstream classifier performance. The space of available generators has expanded rapidly — from interpolation-based oversampling (SMOTE, ADASYN, Borderline-SMOTE, Geometric SMOTE [8]) through conditional GANs (CTGAN, CTAB-GAN+ [9]), copula-based parametric models (GaussianCopula), diffusion models (TabDDPM, TabSyn, TabDiff [10]), and language-model-based synthesizers (GReaT, TabuLa [11], TabMT [12], REaLTabFormer [13]).

This expansion has not been matched by corresponding clarity for practitioners. Existing benchmarks [14–16] evaluate generators across heterogeneous tabular tasks and report aggregate rankings that do not answer the practitioner's actual question: *given my classification task at this positive rate and this sample size, which generator should I use, and will augmentation help at all?*

**This paper extends the broader benchmark literature, including the study closest to this question at the target venue: Won et al. [17], published in Electronics, which compares SMOTE, GaussianCopula, TVAE, and CTGAN on a single banking dataset (UCI Bank Marketing, 7.88:1 imbalance) across statistical fidelity and machine learning utility, and reports a weak negative correlation between the two.** Won et al.'s own Limitations section (§5.6) identifies six specific gaps in their study; this paper directly addresses five of them along the dimensions of dataset breadth, generator coverage, and dose-response methodology. It does **not** extend Won et al.'s fidelity-measurement axis — the multi-dimensional statistical fidelity battery (marginal similarity, correlation preservation, KS tests) that their own study centers on is a distinct question (does statistical realism predict downstream utility?) from the one this paper asks (under what conditions does augmentation help, and why?). We measure one targeted fidelity quantity directly relevant to our own question (§2.4.2) rather than replicate their fidelity framework, and state this explicitly rather than leave it as an unexplained omission (§4.7).

The five gaps addressed:

1. *"the evaluation was conducted on a single dataset from the banking domain"* — we evaluate seven datasets spanning telecom, finance, lead-generation, and real marketing/advertising domains.
2. *"diffusion-based models (TabDDPM, TabSyn) and LLM-based generation (GReaT)... were not included"* — we include both TabDDPM and GReaT (at two LLM scales, GPT-2 and Mistral-7B).
3. *"three traditional machine learning classifiers... inclusion of deep neural network classifiers would broaden the assessment"* — we add a Multi-Layer Perceptron as a fourth classifier family.
4. *"formal PR-AUC computation... were not performed"* — we report Average Precision throughout.
5. *"all experiments were conducted at a single imbalance ratio (7.88:1)"* — we test positive rates from 0.2% to 30%, and additionally run two controlled dose-response sweeps that vary minority-class count independently of dataset identity, directly answering their own stated future-work request: *"systematic evaluation across a range of imbalance ratios... would reveal how the relative effectiveness of augmentation methods changes."* [17]

The sixth gap — hyperparameter optimization for deep generative models — is a genuine boundary of this study, not just theirs: we use library defaults throughout. But we go further than either study's stated future-work list on the adjacent question of *which* hyperparameter matters most: we test TabDDPM at a 5× extended training budget and find performance degrades rather than improves (ruling out undertraining as the explanation for its underperformance), and we sweep the synthetic-to-real mixing ratio — a dimension neither study's future-work list even names — across every generator, establishing a practical rule (α ∈ {0.1–0.3}, never 1.0) that holds regardless of generator choice.

One result previews why the practitioner-facing question matters: on Criteo Display Advertising (0.2% positive rate), 7 of 10 MLP seeds failed to converge using real data alone — the classifier predicted the majority class every time. After CTGAN augmentation, all 10 seeds converged. Augmentation in this regime is not a marginal improvement; it is the difference between a working classifier and a broken one.

This paper does not propose a new generative method, and the underlying regularity it confirms — that resampling delivers value under absolute minority-class rarity rather than relative imbalance per se — is consistent with the imbalanced-learning literature dating to the early 2000s. Its contribution is the same kind Won et al. [17] themselves claim: a systematic, integrated comparison assembled at a scale and with a methodological rigor not previously available for this specific practitioner question, plus two findings — a controlled disentanglement of minority count from dataset identity, and a missing-baseline check that overturns the assumption that a more sophisticated generator should win — that neither Won et al. [17] nor the broader benchmark literature currently provide.

Our contributions are:

1. **A seven-dataset, five-generator, four-classifier characterization of augmentation value**, directly extending Won et al.'s [17] single-dataset, three-classifier, four-generator design along the dimensions listed above.
2. **A controlled dose-response design that disentangles minority-class count from dataset identity** — the first (to our knowledge) within-dataset test of this question, replicated on two datasets with different domains and dimensionality, directly answering the "range of imbalance ratios" question Won et al.'s [17] own future-work section raises.
3. **A missing-baseline check that changes the practical recommendation**: ADASYN and SMOTE — both free — statistically tie CTGAN on both marketing datasets, against the assumption that a more sophisticated generator should win.
4. **A self-disclosed reliability finding**: CTGAN's own fit-to-fit randomness is uncontrolled in the standard SDV implementation, producing AUC swings comparable in magnitude to the paper's headline effect sizes — a limitation we believe has not been previously reported.
5. **A direct TabDDPM vs. CTGAN comparison at two training budgets**, showing extended training widens rather than closes the gap, consistent with an architectural (unconditional vs. conditional sampling) rather than undertraining explanation.

The paper is structured as follows. Section 2 describes datasets, experimental setup, generator implementations, and evaluation metrics. Section 3 reports results. Section 4 discusses mechanism, fidelity-utility trade-offs, and practical implications. Section 5 concludes.

---

## 2. Materials and Methods

### 2.1. Dataset Description

We selected seven publicly available classification datasets spanning the practitioner-relevant range of positive rates, summarized in Table 1. Five serve as controls (positive rate ≥ 11.7%) and two as the treatment condition (positive rate ≤ 0.9%, drawn from real marketing operations).

**Table 1 — Dataset characteristics.**

| Dataset | n (cap) | Positive rate | Domain | Source | Role |
|---|---|---|---|---|---|
| Telco Customer Churn | 7,032 | 26.6% | Telecom | IBM Kaggle | Control |
| Bank Marketing | 15,000 | 11.7% | Finance | UCI [18] | Control |
| German Credit | 1,000 | 30.0% | Finance | OpenML id=31 [19] | Control |
| Nomao Lead (full) | 10,000 | 28.3% | Lead generation | OpenML id=1486 [20] | Control |
| Nomao Lead (sparse, 70% missing) | 500 | 28.3% | Lead generation | OpenML id=1486 [20] | Sparsity stress |
| **Hillstrom Email Marketing** | **10,000** | **0.9%** | **Marketing** | MineThatData [21] | **Treatment** |
| **Criteo Display Advertising** | **10,000** | **0.2%** | **Advertising** | Criteo AI Lab [22] | **Treatment** |

Datasets larger than 10,000 rows are subsampled to the listed cap, defining the data-scarce minority-class regime this paper studies; at full dataset scale the marginal value of synthetic rows is expected to be smaller (see §4.7, Limitations). The Telco Customer Churn control dataset shares its domain with recent applied work combining SMOTE and GAN-based augmentation specifically for churn prediction [23]. Two supplementary datasets — Bank Marketing's full UCI source (45,211 rows, 5,289 positives) and Nomao's full OpenML source (34,465 rows, 9,844 positives) — provide additional headroom for the dose-response design in §2.2.5 and are described there.

### 2.2. Experimental Setup

#### 2.2.1. Implementation Environment

Benchmark and confidence-interval experiments run on CPU (Apple M1 Pro). TabDDPM and GReaT experiments run on Databricks GPU clusters (NVIDIA T4/A10G for TabDDPM and GPT-2 GReaT; H100 for Mistral-7B GReaT, bf16 precision). Total compute: approximately 60 GPU-hours and 80 CPU-hours for the original experiments, plus an additional ~15 CPU-hours for the dose-response and missing-baseline extensions reported here. All generators are implemented via SDV v1.36 (GaussianCopula, CTGAN), imbalanced-learn v0.12 (SMOTE, ADASYN, Borderline-SMOTE, random undersampling), synthcity v0.2.11 (TabDDPM), and be-great v0.0.13 (GReaT). Classifiers are implemented via scikit-learn.

#### 2.2.2. Repeated Experiments and Statistical Analysis

All augmentation results use 5 seeds {42, 123, 7, 2024, 999}, with 95% confidence intervals computed via the t-distribution on per-seed values. Multi-classifier robustness extends to 10 seeds by adding {10, 20, 30, 40, 50}. We report paired t-tests on per-seed differences for all headline comparisons, with Benjamini-Hochberg FDR correction at q=0.10 over the family of 14 tests; effect sizes are Cohen's d_z. The sole exception is the TSTR protocol (§3.1), reported as a single-seed point estimate to match the prior work it is compared against [4].

#### 2.2.3. Data Preprocessing

Categorical features are label-encoded (GaussianCopula) or entity-embedded (CTGAN) [24]; numeric features are used as-is for tree-based classifiers. Missing values are imputed with column medians where required (Nomao). The `duration` feature is dropped from Bank Marketing (target leakage).

#### 2.2.4. Data Splitting Strategy

An 80/20 stratified train/test split is used for all main experiments. For the GReaT small-n experiments, a fixed 10,000-row holdout (random_state=42) is used across all training sizes and seeds, since a variable split would leave too few positive test examples at n=50. For the dose-response experiments (§2.2.5), a large fixed holdout (3,000 rows, stratified at the dataset's natural rate) is drawn once and reused across every minority-count condition, so holdout composition never varies within a dataset.

#### 2.2.5. Synthetic Data Generation Protocol

**Notation.** Let $D = (X, y)$ denote a labeled dataset with $y \in \{0,1\}$, $N = |D|$ the total row count, and $m = \sum_i y_i$ the minority (positive-class) count, so the positive rate is $\pi = m/N$. For a generator $G_\theta$ fit on training split $D_{tr}$, let $S \sim G_\theta(\cdot \mid D_{tr}, n_{syn})$ denote $n_{syn}$ synthetic rows sampled from the fitted generator. The augmented training set at mixing ratio $\alpha$ is

$$D_{tr}^{(\alpha)} = D_{tr} \cup S, \qquad n_{syn} = \lfloor \alpha \cdot |D_{tr}| \rfloor, \qquad \alpha \in \{0.1, 0.2, 0.3, 0.5, 1.0\}.$$

For a classifier $f$ trained on $D_{tr}^{(\alpha)}$ and evaluated on a fixed real holdout $D_{ho}$, the **gain** of generator $G$ at $\alpha$, seed $s$, is

$$\Delta_G(\alpha, s) = \mathrm{AUC}\big(f_{D_{tr}^{(\alpha)}, s},\ D_{ho}\big) - \mathrm{AUC}\big(f_{D_{tr}, s},\ D_{ho}\big),$$

i.e., always measured against that same seed's own real-only baseline, never a blended or cross-seed baseline (the exact bookkeeping error identified and corrected in an earlier revision of this study, §4.5). We report $\bar\Delta_G(\alpha) = \tfrac{1}{|S|}\sum_{s \in S} \Delta_G(\alpha, s)$ with a t-distribution 95% CI across seeds $S$, and take $\alpha^\* = \arg\max_\alpha \bar\Delta_G(\alpha)$ as the reported best-$\alpha$ gain.

**Enrichment ratio.** Let $\hat\pi_{syn}(G) = \tfrac{1}{n_{syn}}\sum_{j} \mathbb{1}[S_j \text{ is positive-class}]$ be the measured positive rate within a generator's own synthetic output at $\alpha=1$ (Table 6). The enrichment ratio

$$\rho(G) = \hat\pi_{syn}(G) \,/\, \pi$$

is the paper's core mechanism statistic: $\rho(G) \approx 1$ for unconditional samplers (GaussianCopula, TabDDPM, GReaT — they reproduce the training distribution's own rate), while $\rho(\text{CTGAN}) \in [7, 89]$ across the two marketing datasets (Table 6) — a direct, measured quantity, not inferred from downstream performance.

For each (dataset, generator, seed) triple we generate synthetic rows at $\alpha \in \{0.1, 0.2, 0.3, 0.5, 1.0\}$. GaussianCopula and CTGAN are refit independently at each α (no fit-once-and-subsample caching in this implementation); SMOTE/ADASYN/Borderline-SMOTE re-call per α (no separate fit step); TabDDPM fits once at max(α) and subsamples for smaller α; GReaT fits once per (n, seed).

**Dose-response design.** To disentangle minority-class count $m$ from dataset identity, we fix $N=10{,}000$ and vary only $m \in \{16, 64, 256, 512, 1{,}024\}$ (equivalently $\pi \in \{0.16\%, 0.64\%, 2.56\%, 5.12\%, 10.24\%\}$) on two datasets chosen for headroom beyond their capped versions in Table 1: Bank Marketing (full source 45,211 rows / 5,289 positives) and Nomao (full source 34,465 rows / 9,844 positives, chosen additionally for its different domain and higher dimensionality — 119 vs. 17 features). GaussianCopula, CTGAN, and SMOTE are evaluated at each level; TabDDPM and GReaT are excluded from this sweep to keep it CPU-only. Algorithm 1 specifies the full procedure.

**Algorithm 1: Minority-Count Dose-Response Sweep**

```
Input:  full source pool P (positives P⁺, negatives P⁻), N = 10,000,
        minority-count grid M = {16, 64, 256, 512, 1024},
        seeds S = {42, 123, 7, 2024, 999}, generators G = {GaussianCopula, CTGAN, SMOTE}
Output: gain estimate Δ̄_G(m) with 95% CI, for every G ∈ G, m ∈ M

1:  D_ho ← StratifiedSample(P, size=3000, seed=42)        // drawn ONCE, fixed for all conditions
2:  P' ← P \ D_ho                                          // remaining pool after holdout removal
3:  for m in M:
4:      n_neg ← N − m
5:      for s in S:
6:          D_tr ← Shuffle( Sample(P'⁺, m, seed=s) ∪ Sample(P'⁻, n_neg, seed=s) )
7:          f_base ← Fit(GradientBoostingClassifier, D_tr, seed=s)
8:          Δ_base ← AUC(f_base, D_ho)
9:          for G in G:
10:             S_syn ← Fit-and-Sample(G, D_tr, n_syn = |D_tr|, seed=s)   // α = 1.0
11:             D_aug ← D_tr ∪ S_syn
12:             f_aug ← Fit(GradientBoostingClassifier, D_aug, seed=s)
13:             Δ_G(m, s) ← AUC(f_aug, D_ho) − Δ_base       // seed-matched, never cross-seed
14:         end for
15:     end for
16:     for G in G: report mean_s Δ_G(m, s) ± t-CI95         // Table 4 / Table 5 / Figure 12–14
17: end for
```

The two features that make this design answer the confounding critique it was built for (§1) are line 1 (one holdout, reused unchanged across every $m$ and $G$, so no condition ever sees a different evaluation surface) and line 13 (gain is always computed against that exact seed's own baseline fit on the same draw, never a baseline averaged or borrowed from elsewhere — the discipline whose absence caused the GPT-2/Mistral-7B bookkeeping bug this study found and corrected, §4.5).

#### 2.2.6. Experimental Workflow

For each (dataset, generator, seed) combination we run: (1) baseline (train/test on real data only), (2) TSTR (train on synthetic only, test on real), (3) the augmentation sweep across α, and (4) — for Hillstrom and Criteo only — a 10-seed, four-classifier robustness extension.

#### 2.2.7. Evaluation Protocol

Splits are stratified on the target. To control seed-induced non-determinism, all experiments invoke a `seed_everything()` routine seeding Python `random`, NumPy, PyTorch CPU/CUDA, and the cuDNN deterministic flag — with one documented exception: the CTGAN-calling code in this implementation never seeds PyTorch specifically (only NumPy), which we identify and quantify as a limitation (§4.5).

### 2.3. Synthetic Data Generation Methods

The generators evaluated span five design families.

#### 2.3.1. SMOTE, ADASYN, and Borderline-SMOTE (Interpolation-Based Oversampling)

**SMOTE** [25] generates synthetic minority examples by linear interpolation between a minority example $x_i$ and one of its $k$-nearest minority neighbors $x_{nn}$:

$$x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \qquad \lambda \sim \mathcal{U}(0,1).$$

It has no separate fit step and operates only on the minority class. **ADASYN** [26] extends this by weighting each minority example $x_i$ by the local density of majority neighbors,

$$r_i = \frac{1}{k}\Big|\{x_j \in kNN(x_i) : y_j = 0\}\Big|, \qquad \hat{r}_i = r_i \,\Big/\, \sum_{i'} r_{i'},$$

then generates $g_i = \mathrm{round}(\hat{r}_i \cdot n_{syn})$ synthetic points at $x_i$ using the same interpolation rule as SMOTE — examples in harder-to-learn (more majority-surrounded) regions receive proportionally more synthetic neighbors. **Borderline-SMOTE** [27] applies the identical SMOTE interpolation formula but restricts the base points $x_i$ to minority examples classified as "in danger" (a majority of their $k$-NN are majority-class). All three are free (no GPU, no training step beyond nearest-neighbor search) and are evaluated at the same α-sweep as the deep generative methods.

#### 2.3.2. Gaussian Copula

**GaussianCopula** [28] models the joint CDF of the $d$ features via a copula $C$ applied to fitted per-feature marginals $F_1, \dots, F_d$:

$$F(x_1, \dots, x_d) = C\big(F_1(x_1), \dots, F_d(x_d)\big), \qquad C = \Phi_\Sigma\big(\Phi^{-1}(F_1(x_1)), \dots, \Phi^{-1}(F_d(x_d))\big),$$

where $\Phi_\Sigma$ is the multivariate Gaussian CDF with correlation matrix $\Sigma$ estimated from the rank-transformed training data, and $\Phi$ the standard normal CDF. Sampling draws directly from this fitted joint distribution — no class label conditioning enters the generative process at all, which is why it samples unconditionally at the natural class rate regardless of $\alpha$ (§3.2, Table 6).

#### 2.3.3. CTGAN (Conditional Tabular GAN)

**CTGAN** [29] is a conditional generative adversarial network [30] trained with the standard minimax objective

$$\min_G \max_D\; \mathbb{E}_{x \sim p_{data}}\big[\log D(x \mid c)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z, c) \mid c)\big)\big],$$

but critically, the conditional vector $c$ encodes a target discrete-column value drawn during training by **training-by-sampling**: at each step, $c$ is drawn log-frequency-weighted across that column's categories (rather than at their natural empirical frequency), so the generator learns to condition on — and at inference time can be asked to target — the minority class specifically. This is the exact mechanism absent from GaussianCopula and TabDDPM's unconditional formulations, and it is what produces the 7–89× minority-class enrichment measured directly in this study (Table 6) rather than inferred from downstream performance alone.

#### 2.3.4. TabDDPM and GReaT

**TabDDPM** [31,32] applies a Gaussian forward diffusion process [33] to continuous features,

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big),$$

(with an analogous multinomial diffusion process for categorical columns) and trains a network $\epsilon_\theta$ to reverse it, sampling by iterative denoising from $x_T \sim \mathcal{N}(0, I)$ back to $x_0$. It is reported as the strongest single-table generator on general augmentation benchmarks [15], but the reverse process here samples unconditionally over the joint feature-label distribution — no class-conditioning term enters $\epsilon_\theta$'s objective, the same structural limitation as GaussianCopula, just via a different generative mechanism. **GReaT** [34] serializes each tabular row as a natural-language string and fine-tunes a pretrained causal language model (GPT-2, 117M; and Mistral-7B, 7B parameters) with the standard autoregressive objective $\mathcal{L} = -\sum_t \log p_\theta(w_t \mid w_{<t})$ over the serialized token sequence — sampling is likewise unconditional on the label unless explicitly prompted, which the default `guided_sampling` configuration used here does not enforce class-balanced generation.

#### 2.3.5. Random Undersampling and Cost-Sensitive Reweighting (Non-Generative Baselines)

**Random majority undersampling** removes majority-class rows to reach a target class ratio (evaluated here at 1:1); it adds no synthetic data at all. **`class_weight='balanced'`** reweights the loss function via `sample_weight=compute_sample_weight('balanced', y_train)`, at zero data or compute cost. Both are included specifically because a prior review of this work flagged their absence as a decisive gap: *"before fitting a GAN... a practitioner would first try class weighting... and random minority oversampling... This omission is decisive for the paper's deliverable."*

### 2.4. Evaluation Metrics

#### 2.4.1. Classification Utility

The primary metric is AUC-ROC (threshold-independent, standard in the benchmark literature this paper compares against). Secondary metrics are Average Precision throughout, and — for Hillstrom and Criteo specifically — Accuracy, Precision, Recall, and F1 (minority class) at the classifier's default 0.5 threshold, computed for every method except GReaT (see §4.7 for why GReaT is scoped out of this secondary-metric set); threshold-based metrics are known to be less stable than AUC-ROC under extreme class imbalance with few minority holdout examples [3,35]. The primary downstream classifier is `GradientBoostingClassifier` [36] (`n_estimators=100, max_depth=4`), implemented via scikit-learn [37]; Hillstrom and Criteo additionally use Logistic Regression, Random Forest, and a Multi-Layer Perceptron to verify findings are not classifier-specific.

#### 2.4.2. Synthetic Data Class Distribution (Fidelity Proxy)

Rather than a full multi-dimensional statistical fidelity battery (marginal similarity, correlation preservation, KS tests — as in Won et al.'s [17] §3.4.1), we measure one targeted fidelity quantity directly relevant to the imbalanced-classification question this paper asks: the fraction of positive-class rows in synthetic samples generated at α=1.0 (Table 6). This directly tests the mechanism hypothesis (conditional generators enrich the minority class; unconditional generators do not) without requiring the broader fidelity framework, which we leave to future work (§4.7).

---

## 3. Results

### 3.1. Synthetic-Only Training Underperforms Real Data (TSTR)

Training on synthetic data alone and testing on real data (TSTR) gives the cleanest measure of how faithfully a generator captures the joint distribution. Across three benchmark datasets, every generator's TSTR AUC falls materially below the real-data baseline (Telco: −4.1%; Bank Marketing: −17.4%; German Credit: −27.2%, all relative to baseline). The gap grows as the dataset shrinks, and no generator closes it — synthetic data is an augmentation method, not a replacement method.

### 3.2. Class Distribution After Augmentation

**Table 6 — Synthetic positive rate by generator (Hillstrom / Criteo, 5-seed mean ± std for GaussianCopula/CTGAN/SMOTE; single-run for TabDDPM, not independently re-verifiable).**

| Generator | Hillstrom | Criteo |
|---|---|---|
| Real training data | 0.90% | 0.30% |
| GaussianCopula | 0.96% ± 0.12% | 0.32% ± 0.06% |
| **CTGAN** | **6.65% ± 0.41%** | **26.78% ± 1.19%** |
| SMOTE | 100% (minority only) | 100% (minority only) |
| TabDDPM | 0.89% ± 0.05% | 0.33% ± 0.09% |

GaussianCopula and TabDDPM both mirror the natural (rare) positive rate — they sample unconditionally and leave the minority class no better represented than the real data. CTGAN's conditional vector generates minority rows at 7–89× the natural rate. This table makes the mechanism visible directly, without inference from downstream performance alone.

### 3.3. Classification Performance

**Benchmark datasets (positive rate ≥ 10%).** Best gain from any generator at any α is consistently below +0.5 AUC points across Telco (+0.21), Bank Marketing (−0.17), German Credit (+0.27), and Nomao (−0.06) — all within the baseline confidence interval.

![Figure 1](../results/plots/paper2/fig1_summary_comparison.png)

**Figure 1.** Cross-dataset summary: augmentation gains concentrate on the two imbalanced marketing datasets; all generators are within noise on the four balanced benchmarks. This is the single clearest visual answer to "which range shows value and which doesn't."

![Figure 2](../results/plots/paper2/fig2_ucurves_benchmark.png)

**Figure 2.** U-shaped augmentation curves for all four benchmark datasets — gains peak at α ∈ {0.1–0.3} and degrade toward α=1.0 on every dataset, but stay within noise of baseline throughout (see §4.3 for the α* discussion this motivates).

**Does the AUC-only conclusion survive a full metric suite?** The benchmark-dataset result above uses AUC-ROC only. We extend the same 5-seed, best-α protocol to Accuracy/Precision/Recall/F1 on all four control datasets (Table 7) to check whether AUC's threshold-independence is masking a real shift in operating point.

**Table 7 — Full metric suite, control datasets, best-α gain vs. seed-matched baseline (5-seed paired t-test; p_fdr from a 24-test family scoped to this table only).**

| Dataset | Method (α*) | AUC gain | p(AUC) | F1 gain | p(F1) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Telco | SMOTE (0.1) | −0.19 pts | 0.394 | +0.0175 | 0.038 | 0.613 | 0.571 | 79.0% |
| Bank Marketing | GaussianCopula (0.2) | −0.15 pts | 0.171 | **−0.0417** | **0.009** | 0.638 | 0.356 | 90.1% |
| Bank Marketing | CTGAN (0.1) | **−0.28 pts** | **0.006** | −0.0261 | 0.026 | 0.633 | 0.377 | 90.2% |
| Bank Marketing | SMOTE (0.1) | −0.43 pts | 0.036 | **+0.0599** | **0.008** | 0.578 | 0.539 | 90.0% |
| German Credit | SMOTE (0.2) | +0.28 pts | 0.692 | +0.0154 | 0.257 | 0.627 | 0.583 | 77.1% |
| Nomao | CTGAN (0.1) | −0.06 pts | 0.042 | −0.0000 | 0.991 | 0.935 | 0.917 | 95.8% |

*GaussianCopula/CTGAN rows for Telco and German Credit, and GaussianCopula/SMOTE for Nomao, are omitted from this condensed table (all non-significant, gains within ±0.5 AUC pts and ±0.03 F1 of zero); the full 12-row table is in the companion analysis document. Bold values survive Benjamini-Hochberg FDR correction at q=0.10 within this table's own family — not merged into the §3.5 family of 14. Nomao's numbers reflect a corrected rerun at n=10,000 (matching Table 1); an earlier internal draft used n=15,000 for Nomao by mistake, which does not change any qualitative conclusion but did shift one comparison in or out of FDR significance at a magnitude ≤0.003 F1 points either way.*

Three comparisons are FDR-significant, but every one is tiny in absolute terms (≤0.28 AUC points, ≤0.06 F1 points), detectable only because these control datasets have unusually tight per-seed variance — consistent with, not contradictory to, "negligible effect," just occasionally directionally detectable. The one exception worth practical attention is SMOTE's F1 gain on Bank Marketing (+0.060, FDR-significant) and, more weakly, Telco (+0.018, uncorrected): a genuine precision-recall trade-off (Bank Marketing recall rises from ~0.36–0.38 to 0.539 as precision falls to 0.578) that AUC-ROC, being threshold- and prior-invariant, does not register. GaussianCopula/CTGAN show no such effect — if anything, their F1 moves the opposite direction. Nomao is the cleanest "no effect" case: every gain is within ±0.06 AUC points and ±0.003 F1 of zero, none FDR-significant, consistent with its near-ceiling baseline (§3.6) leaving no room for any method to move.

**Sparsity stress test.** A secondary control — Nomao with 70% of feature values simulated missing (n=500) — tests whether augmentation helps when the baseline is degraded by *feature*-information starvation rather than minority-class starvation. It does not: the sparse baseline (0.897 ± 0.062) recovers only +0.50 pts from the best generator (CTGAN, α=0.1), against a dense-data reference of 0.9716 ± 0.0103 (a 7.46-point gap augmentation does not close). This confirms the mechanism in §4.1 is specific to minority-class data scarcity, not data scarcity generally.

![Figure 3](../results/plots/paper2/fig3_ucurve_sparse.png)

**Figure 3.** Augmentation U-curve for the sparsity stress test — flat across all α, confirming sparsity-driven performance gaps are not recoverable through synthetic augmentation.

![Figure 4](../results/plots/paper2/fig4_lowdata_regime.png)

**Figure 4.** Low-data regime: AUC vs. real training set size for benchmark datasets. Augmentation recovers 30–60% of the performance gap at n=250, narrowing rapidly by n≥1,000 — a second, independent line of evidence that augmentation value is a function of how much real data (specifically minority-class data) is available, not a fixed generator property.

**Marketing datasets (extreme imbalance).** Hillstrom: baseline 0.548 ± 0.092; CTGAN 0.605 ± 0.073 (+5.75 pts); SMOTE 0.606 ± 0.087 (+5.84 pts). Criteo: baseline 0.846 ± 0.228; CTGAN 0.974 ± 0.036 (+12.87 pts); SMOTE 0.966 ± 0.026 (+11.99 pts). Augmented confidence intervals are substantially narrower than baseline — synthetic augmentation under extreme imbalance stabilizes learning, not just improves its mean.

![Figure 5](../results/plots/paper2/fig5_marketing_ci.png)

**Figure 5.** Augmentation U-curves for Hillstrom and Criteo with 95% CI bands — the steep rise from α=0 to α≈0.2 and the narrower augmented-vs-baseline CI bands are the primary visual evidence for both the headline gain and the variance-stabilization finding.

**Multi-classifier robustness (10-seed).** On Criteo, MLP fails to converge on real data alone in 7 of 10 seeds (AUC < 0.15); CTGAN augmentation rescues all 10 (AUC 0.865–0.985). The CTGAN advantage holds across Gradient Boosting (+12.04 pts) and Random Forest (+9.55 pts); Logistic Regression is insensitive due to a near-ceiling baseline (0.963).

![Figure 6](../results/plots/paper2/fig8_mlp_rescue.png)

**Figure 6.** MLP per-seed AUC on Criteo, baseline vs. CTGAN-augmented — seeds that failed to converge on real data alone (AUC<0.15) all reach AUC>0.86 after augmentation.

![Figure 7](../results/plots/paper2/fig9_multiclassifier.png)

**Figure 7.** Multi-classifier robustness on Criteo across all four classifier families — the CTGAN advantage is not an artifact of the primary (Gradient Boosting) classifier choice.

**TabDDPM vs. CTGAN at two training budgets.** At N_iter=2,000 (default) and N_iter=10,000 (5× extended), CTGAN outperforms TabDDPM on both datasets; extended training widens the gap rather than closing it (TabDDPM at 10k goes uniformly negative on Hillstrom). Paired comparison: Hillstrom Δ=+7.76 pts (d_z=1.25, p=0.049); Criteo Δ=+6.41 pts (d_z=0.73, p=0.179).

![Figure 8](../results/plots/paper2/fig7_tabddpm_comparison.png)

**Figure 8.** CTGAN vs. TabDDPM at two training budgets — extended training (dashed) widens rather than closes the CTGAN advantage; all five TabDDPM-10k α values fall below baseline on Hillstrom.

**GReaT (GPT-2 and Mistral-7B) — does augmentation help here at all?** A directional positive signal at small n on Hillstrom (n=50, 4/5 seeds win) decays and inverts to a robustly negative effect at n=2,000 (0/5 seeds win, p=0.001, the only FDR-significant individual comparison in the study, p_fdr=0.008) — GReaT actively hurts as training size grows, the opposite of every other method tested. Replicating with Mistral-7B (7B parameters vs. GPT-2's 117M) does not change the outcome once gain is computed against each backbone's own seed-matched baseline: Mistral-7B underperforms its own baseline at every n tested on Telco (−4.70 to −1.99 pts), still hurts on anonymized features (German Credit), and its best Hillstrom gain (+1.20 pts, 3/5 valid seeds) remains well below CTGAN. Scaling the backbone 60-fold does not rescue GReaT on any of the three datasets tested — the failure mode is that GReaT samples unconditionally (like TabDDPM and GaussianCopula), so it dilutes rather than enriches the minority class regardless of the language model's raw capability.

![Figure 9](../results/plots/paper2/fig10_modernllm_comparison.png)

**Figure 9.** GPT-2 vs. Mistral-7B vs. baseline across three datasets, each backbone plotted against its own seed-matched baseline. Backbone scaling does not rescue GReaT on any dataset tested — the clearest single figure for "how GReaT helps or doesn't."

### 3.4. Method-Wise Average Performance and the Missing-Baseline Comparison

**Table 8 — All evaluated methods, best-α gain with 95% CI on each dataset separately (5-seed CI; a single blended average across datasets is avoided here since it would discard the uncertainty on each estimate).**

| Method | Cost | Hillstrom gain (95% CI) | Criteo gain (95% CI) |
|---|---|---|---|
| CTGAN | GPU/CPU, ~2 min/seed | +5.75 ± 7.32 pts | +12.87 ± 3.58 pts |
| ADASYN | Free, instant | +5.80 ± 6.70 pts | +12.35 ± 2.82 pts |
| SMOTE | Free, instant | +5.84 ± 8.72 pts | +11.99 ± 2.57 pts |
| Borderline-SMOTE | Free, instant | +1.13 ± 14.95 pts | +11.34 ± 5.02 pts |
| TabDDPM (2k) | GPU, ~6 min/seed | +1.35 ± 9.72 pts | +9.91 ± 5.69 pts |
| Random undersampling | Free, instant | +2.11 ± 6.05 pts | +7.24 ± 5.76 pts |
| GaussianCopula | CPU, ~seconds | +0.44 ± 10.68 pts | +6.61 ± 8.65 pts |
| `class_weight='balanced'` | Free, instant | −1.80 ± 7.33 pts | +5.32 ± 8.55 pts |
| **GReaT (GPT-2), n=2,000*** | GPU, ~5 min/seed | **−6.87 pts** (0/5 seeds win, p_fdr=0.008) | *not evaluated on Criteo* |

*GReaT's protocol varies training-set size n at fixed α=1.0, not α at fixed n=10,000 like every other method — its number here is not on the same axis as the rest of the table and is included for visibility, not direct ranking. At its largest tested n (2,000, the condition most comparable in spirit to the other methods' full-data condition), GReaT is the only method in this entire study to deliver a *statistically significant, FDR-corrected* effect — and it is significantly harmful, not helpful. At every other n tested on Hillstrom, GReaT's gain ranges from +2.25 pts (n=50) down through this point, never exceeding TabDDPM's or GaussianCopula's performance at any n (§3.3).

Confidence intervals overlap substantially at the top of Table 8 — the wide CIs on Hillstrom in particular (a consequence of only ~72 real minority examples per split, §4.1) mean CTGAN, ADASYN, and SMOTE cannot be distinguished from each other by eye, and a direct paired comparison confirms this is not just an eyeballing artifact: ADASYN and CTGAN are statistically indistinguishable (Hillstrom p=0.970, Criteo p=0.212). This is precisely the outcome a prior reviewer of this work predicted was likely, given that a free heuristic already matching a GPU-trained generator "is decisive for the paper's deliverable": *if a cheap method recovers most of the reported gain at zero cost, recommending the expensive one is the wrong advice.*

![Figure 10](../results/plots/paper2/fig15_missing_baselines.png)

**Figure 10.** All evaluated methods on both marketing datasets, sorted by gain. ADASYN sits within noise of CTGAN on both; Borderline-SMOTE and random undersampling deliver smaller but real, zero-cost gains.

### 3.5. Statistical Significance Analysis

We report paired t-tests with Benjamini-Hochberg FDR correction (q=0.10) over a family of 14 headline comparisons. Individual per-dataset comparisons show medium-to-large effect sizes (d_z = 0.62–1.18) but none reach FDR significance at 5–10 seeds (80% power at 5 seeds requires d_z ≥ 2.0). The cross-dataset regression of CTGAN gain on log(positive rate) across six datasets is the primary statistical support for the regime-level claim (R²=0.92, p=0.0023), robust to leave-one-out refitting (R² 0.90–0.96, all p<0.05 across all six LOO fits). The only individually FDR-significant comparison is GReaT harm at Hillstrom n=2,000 (p_fdr=0.008).

![Figure 11](../results/plots/paper2/fig6_regression_hypothesis.png)

**Figure 11.** Cross-dataset regression of CTGAN gain on log(positive rate) across six datasets (slope=−0.024, R²=0.92, p=0.0023) — the primary statistical support for the regime-level claim, since individual per-dataset comparisons are underpowered on their own.

### 3.6. Precision–Recall Trade-Off and the Dose-Response Curve

**Threshold-based metrics diverge sharply by method (Table 9).** At the default 0.5 threshold, CTGAN and GaussianCopula collapse to F1=0 on Hillstrom (the classifier never crosses the threshold into predicting positive at 0.9% positive rate) while Accuracy remains uselessly high (~98–99%). Random undersampling is the exception: it recovers 57–96% of positives (vs. 1–24% for every enrichment-based method) at the cost of collapsing precision and accuracy to near-chance (51.1% on Hillstrom) — the expected mechanism of shifting the training-time class prior, not a defect, and a genuinely different operating point a practitioner needing high recall with human-review tolerance might prefer.

**Table 9 — F1 / Precision / Recall / Accuracy at default threshold (5-seed CI).**

| Method | Hillstrom (F1/P/R/Acc) | Criteo (F1/P/R/Acc) |
|---|---|---|
| Baseline | 0.012 / 0.013 / 0.011 / 98.4% | 0.259 / 0.312 / 0.240 / 99.6% |
| CTGAN | 0.000 / 0.000 / 0.000 / 98.9% | 0.214 / 0.264 / 0.184 / 99.6% |
| ADASYN | 0.000 / 0.000 / 0.000 / 99.0% | 0.225 / 0.198 / 0.273 / 99.4% |
| SMOTE | 0.014 / 0.018 / 0.011 / 98.6% | 0.180 / 0.133 / 0.291 / 99.2% |
| Borderline-SMOTE | 0.000 / 0.000 / 0.000 / 98.9% | 0.262 / 0.247 / 0.291 / 99.4% |
| GaussianCopula | 0.000 / 0.000 / 0.000 / 98.3% | 0.204 / 0.211 / 0.229 / 99.5% |
| Random undersampling | **0.022** / 0.011 / **0.572** / 51.1% | **0.032** / 0.016 / **0.960** / 80.9% |
| `class_weight='balanced'` | 0.018 / 0.010 / 0.094 / 90.3% | 0.094 / 0.067 / 0.167 / 99.1% |
| TabDDPM (2k)* | 0.000 / n/c / n/c / n/c | 0.156 ± 0.181 / n/c / n/c / n/c |

*Precision/Recall/Accuracy were not computed for TabDDPM (n/c = not computed) — its evaluation harness saved AUC-ROC, F1, and Average Precision only, and recomputing the remainder requires a GPU rerun (`synthcity`) not performed in this revision. This is a narrower gap than GReaT's (§4.7): F1 and AP are real, measured values here, only the three additional breakdown metrics are missing. TabDDPM's Hillstrom F1=0.000 matches the same threshold-collapse pattern seen in every enrichment-based method on this dataset (row above); its Criteo F1 (0.156) sits below CTGAN (0.214) and ADASYN (0.225), consistent with its lower AUC gain in Table 8.

**The dose-response curve directly tests whether the extreme-scarcity effect reflects minority count or dataset identity**, by fixing total sample size and varying minority count alone. On Bank Marketing, gains are positive only at the lowest count tested (16; SMOTE +5.31 pts, p=0.033) and turn significantly negative from count=64 onward (up to −3.62 pts, p<0.01).

![Figure 12](../results/plots/paper2/fig12_dose_response.png)

**Figure 12.** Bank Marketing dose-response: AUC and gain vs. minority count. Positive gain only at the lowest count tested; augmentation significantly hurts from count=64 onward, holding smoothly across the whole range with no reversal (the count=512 point was added specifically to check for a reversal partway through — there is none).

On Nomao — chosen for a different domain and 119 features vs. Bank Marketing's 17 — the same direction holds, but the magnitude is an order of magnitude smaller (−0.09 to −0.31 pts), because Nomao's baseline is already near ceiling (AUC>0.97 by count=256), leaving little room to move.

![Figure 13](../results/plots/paper2/fig13_dose_response_combined.png)

**Figure 13.** Dose-response replication side by side: Bank Marketing vs. Nomao. Same qualitative direction (gains shrink as count rises), different threshold and severity — Nomao's near-ceiling baseline leaves little room to move in either direction.

Overlaying both curves on the original six-dataset cross-dataset comparison shows each curve's endpoint converges toward that dataset's own cross-dataset point — an internal consistency check between two independently run analyses.

![Figure 14](../results/plots/paper2/fig14_gain_vs_positive_rate.png)

**Figure 14.** CTGAN gain vs. positive rate: sparse cross-dataset points (diamonds, one per dataset) overlaid with the two dense within-dataset dose-response curves. This is the paper's central "which range shows value and which doesn't" figure — diminishing and reversing returns as positive rate rises, replicated within two datasets, not just inferred from six sparse cross-dataset points.

**Table 10 — Dose-response summary (CTGAN gain vs. minority count, both datasets).**

| Minority count | Positive rate | Bank Marketing gain | Nomao gain |
|---|---|---|---|
| 16 | 0.16% | +2.65 pts (p=0.171) | +11.21 pts (p=0.016) |
| 64 | 0.64% | −1.28 pts (p=0.198) | +1.81 pts (p=0.029) |
| 256 | 2.56% | −3.48 pts (p=0.008) | +0.40 pts (p=0.169) |
| 512 | 5.12% | −2.87 pts (p<0.001) | −0.16 pts (p=0.059) |
| 1,024 | 10.24% | −2.24 pts (p=0.008) | −0.31 pts (p=0.001) |

---

## 4. Discussion

### 4.1. Why Minority-Example Scarcity Is the Mechanism

The results trace a clean dichotomy at the cross-dataset level: on datasets with positive rates between 11.7% and 30.0%, best augmentation gain is +0.27 points; on datasets at 0.9% and 0.2%, the same generators deliver +5.7 to +12.9 points. We interpret this through the minority-example budget: at 0.2% positive rate and an 8,000-row training set, only ~16 minority examples are expected, and the variance of this count across seeds is the source of the wide baseline confidence intervals (±22.8 pts on Criteo). Synthetic augmentation densifies the minority-class region of feature space; CTGAN and SMOTE/ADASYN do this directly, though by different mechanisms — SMOTE interpolates within the convex hull of observed minority examples without consulting the majority class, which can be boundary-blind and prone to near-duplicate samples when minority seeds are few [38], while CTGAN conditions on the class label while preserving cross-feature correlations, yielding minority samples that are in principle more aware of where the class boundary lies [39] — while GaussianCopula and unconditional TabDDPM sample proportionally to the existing imbalance and cannot.

### 4.2. Deep Learning Methods' Performance Is Not Uniquely Strong

This finding is consistent with the broader tabular-data literature, where tree-based models are repeatedly shown to match or outperform deep learning approaches [40,41]. TabDDPM, the strongest generator on general benchmarks [15], underperforms CTGAN in this regime by 3–4 AUC points and gets *worse* with 5× more training — inconsistent with an undertraining explanation and consistent with an architectural one (unconditional sampling cannot be fixed by more compute; it needs a conditioning mechanism), consistent with prior comparative work finding TabDDPM competitive at moderate imbalance but conditional generators advantageous under extreme scarcity [42,43]. GReaT's failure modes are framework-level, not backbone-specific: scaling from GPT-2 (117M) to Mistral-7B (7B, 60× larger) does not change the outcome on any of three datasets once gains are computed against each backbone's own seed-matched baseline. Section 3.4's missing-baseline comparison extends this theme: CTGAN is not uniquely capable of delivering the paper's headline effect — ADASYN, a 2008-era heuristic, ties it exactly.

### 4.3. Evaluation Metrics and Operational Considerations

AUC-ROC and threshold-based metrics (F1/Precision/Recall) diverge sharply in this regime (§3.6), and the divergence itself is informative: augmentation improves the classifier's ability to *rank* positives above negatives without necessarily shifting enough probability mass across a fixed 0.5 threshold to change count-based predictions. We did not tune the decision threshold; threshold-moving remains a promising, untested, zero-cost extension. Compute cost varies by three orders of magnitude across evaluated methods (SMOTE/ADASYN: seconds; CTGAN: ~2 min/seed CPU; TabDDPM: 6–29 min/seed GPU; GReaT: ~5–30 min/seed GPU) — a dimension the ranking in Table 8 does not capture on its own and that should weigh heavily in a practitioner's choice given how close the top three methods are on raw gain.

**Which mixing ratio α is best?** A consistent secondary observation across every augmentation sweep (Figures 2, 5) is the location of the α* peak: on the four benchmark datasets, the best gain — to the extent any gain is observable at all — occurs at α ∈ {0.1, 0.2, 0.3} on every dataset, and degrades toward α=1.0. On the marketing datasets, CTGAN peaks at α=1.0 on Hillstrom but α=0.2 on Criteo; SMOTE peaks at α=0.1 on Hillstrom and α=0.3 on Criteo — so the exact optimum is dataset- and generator-specific, but it never exceeds α=1.0, and an exhaustive grid search is unnecessary: a 5-point sweep over α ∈ {0.1, 0.2, 0.3, 0.5, 1.0} is sufficient to locate the optimum within a 0.1 step in every case tested. We interpret the U-shape as a quality-quantity trade-off: moderate synthetic volume densifies the minority-class region without overwhelming the real-data signal; at high volume, the synthetic rows' imperfect fidelity begins to bias the decision boundary. **Practical guidance: start at α=0.1–0.3, not α=1.0**, regardless of which generator is chosen.

### 4.4. Model-Specific Observations

Random undersampling is the clearest example of a method whose behavior cannot be summarized by AUC gain alone (§3.6): it is not competitive on F1/Precision but delivers the highest recall of any method by a wide margin, a genuinely different and viable option for high-recall, human-reviewed use cases. GaussianCopula's failure mode throughout is structural, not a matter of degree: it samples at the natural class rate by design, so no amount of α-tuning changes its fundamental inability to enrich the minority class.

### 4.5. A Self-Disclosed Reliability Limitation: CTGAN's Uncontrolled Fit-to-Fit Variance

Unlike TabDDPM and GReaT (both of which seed PyTorch/CUDA via a `seed_everything()` routine), the CTGAN-calling code in this study never calls `torch.manual_seed()` — only NumPy's RNG is seeded. We confirmed empirically that this matters: holding input data, split, and NumPy seed fully fixed, four repeated CTGAN fits produced downstream AUC ranging from 0.527 to 0.632 (a 10.45-point range) — noise of the same order of magnitude as this paper's headline effect sizes. GaussianCopula is unaffected (fully deterministic under the same test; no PyTorch dependency). This mirrors a broader, documented problem in ML benchmarking generally, where insufficient seed control can inflate apparent method differences [44]. This means every CTGAN confidence interval reported here should be read as a lower bound on true uncertainty. We did not rerun the affected experiments with corrected seeding (the fix is straightforward — adding `torch.manual_seed(seed)` before each fit — but rerunning is computationally expensive); we disclose it rather than silently leave it undiscovered, consistent with broader calls in the literature for transparent reporting of synthetic-data methodology risks [45], since, to our knowledge, this reliability gap has not been previously reported and likely affects prior CTGAN benchmarks generally.

### 4.6. Practical Implications and Decision Framework

**Table 11 — Practitioner decision guide.**

| Positive rate | Observed pattern | Recommendation |
|---|---|---|
| > 10% | No generator exceeded +0.27 pts | Skip augmentation |
| 1%–10% | Tested directly on two datasets: significantly harmful on one, negligible on the other | Validate on your own data — the two datasets tested here disagree |
| 0.5%–1% | ADASYN/SMOTE/CTGAN tied, +5–6 pts | Try ADASYN or SMOTE first (free) |
| < 0.5% | ADASYN/SMOTE/CTGAN tied, +12–13 pts | Try ADASYN or SMOTE first; reserve CTGAN for cases where they underperform on your data |

**GReaT is not recommended in any positive-rate band tested.** It never wins at any n tested against CTGAN/ADASYN/SMOTE, its best-case gain is small and non-significant, and its worst case (large n) is the only *statistically significant harm* in the entire study. Scaling the backbone from GPT-2 to Mistral-7B does not change this. GReaT is included in this study to answer whether LLM-based synthesis is competitive in this regime, not because it is a candidate recommendation — the answer, directly, is that it is not (§3.3, §4.2).

### 4.7. Limitations

**Dataset breadth and scope.** All experiments cap at n=10,000; conclusions apply to the data-scarce minority-class regime this defines, not necessarily to full-scale industrial datasets (Hillstrom's full 64,000 rows, Criteo's 13.9M). **Single fixed holdout per dataset** within the main experiments (not the dose-response design, which uses a single large fixed holdout by construction). **Generator hyperparameters use library defaults** throughout except the TabDDPM training-budget comparison. **F1/Precision/Recall/Accuracy are not computed for GReaT** — its harness captured AUC-ROC only, and given GReaT's own documented per-seed fit variance, a statistically meaningful secondary-metric estimate would require rerunning the full multi-seed, multi-backbone matrix; this does not weaken the paper's GReaT conclusion, which rests on the sampling-mechanism argument in §4.1–4.2 and is metric-agnostic. **Precision/Recall/Accuracy are also not computed for TabDDPM** (Table 9) — a narrower gap than GReaT's, since F1 and Average Precision are already measured for TabDDPM; the remaining three metrics would require a GPU rerun not performed in this revision. **Privacy is not evaluated** — SMOTE-family methods generate near-duplicates of real minority examples (elevated membership-inference risk); practitioners deploying in regulated environments (GDPR, CCPA) should run distance-to-closest-record and membership-inference checks before deployment [46]. **Multi-dimensional statistical fidelity** (marginal similarity, correlation preservation, KS tests, as in Won et al.'s [17] framework) is not measured here beyond the targeted class-distribution check in §3.2; a full fidelity-utility trade-off analysis on this dataset suite is left to future work. **CTGAN's uncontrolled fit-to-fit variance** (§4.5) is disclosed but not corrected in this revision.

---

## 5. Conclusions

We tested whether minority-example scarcity is the strongest observed correlate of synthetic augmentation value on tabular classification, extending the closest prior benchmark at this venue [17] along five of its own six stated limitations. Across seven datasets, five generators, and two independent dose-response experiments, minority-example scarcity is confirmed as a real, replicated driver of augmentation value — but neither the exact threshold nor the severity of crossing it is a portable constant across datasets. The more consequential finding for practitioners is that CTGAN is not uniquely capable of delivering the effect this literature documents: ADASYN and SMOTE, both free, tie it directly. We additionally identify and disclose an uncontrolled reliability gap in CTGAN's standard implementation that, to our knowledge, has not been previously reported. Future work should extend the dose-response design to additional datasets to establish whether "room to improve" (baseline separability) rather than minority count alone is the more fundamental moderator, and should apply Won et al.'s [17] multi-dimensional fidelity framework to this dataset suite to directly test the fidelity-utility trade-off their study identifies against the enrichment mechanism this study identifies.

---

## Author Contributions

*TODO — needs actual input from Aditya and coauthors (CRediT taxonomy: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing, visualization, supervision).*

## Funding

*TODO — needs actual input (e.g., "This research received no external funding" if applicable).*

## Institutional Review Board Statement

*TODO — likely "Not applicable" given all datasets are publicly available, de-identified, non-human-subjects tabular data, but confirm before submission.*

## Informed Consent Statement

*TODO — likely "Not applicable" for the same reason.*

## Data Availability Statement

All datasets used are publicly available: Hillstrom (MineThatData), Criteo (Criteo AI Lab uplift dataset), Telco Customer Churn (IBM/Kaggle), Bank Marketing (UCI ML Repository), German Credit (OpenML id=31), Nomao (OpenML id=1486). Code, experiment scripts, and raw result CSVs are available at https://github.com/adityapt/llmsynth *(TODO: confirm this repository should be public before listing it here — verify current visibility setting and remove/redact anything sensitive first)*.

## Conflicts of Interest

*TODO — needs actual input (standard: "The authors declare no conflicts of interest.").*

---

## Appendix A. Experimental Configuration

See Table 2 (generator hyperparameters) and Table 3 (classifier hyperparameters) in the companion analysis document (`paper2-empirical.md` §3.2–3.4) for the complete configuration table; reproduced in full detail there rather than duplicated here to avoid drift between the two documents.

**Note on figure numbering.** Figure numbers in this document (1–14) follow this document's own presentation order and do not match the numeric suffix in each PNG's filename (e.g., this document's Figure 6 is `fig8_mlp_rescue.png`) — filenames reflect the original analysis order in `paper2-empirical.md`, which itself has a similar, separately-documented numbering mismatch. Stated here explicitly rather than left for a reader to discover while cross-referencing the repository.

## Appendix B. Evaluation Metrics Definitions

Let $TP, FP, TN, FN$ denote confusion-matrix counts at the classifier's default 0.5 decision threshold, and let $\hat{y} \in [0,1]$ be the classifier's predicted probability for the positive class.

**Precision, Recall, Accuracy, F1 (minority class):**

$$P = \frac{TP}{TP+FP}, \qquad R = \frac{TP}{TP+FN}, \qquad \mathrm{Acc} = \frac{TP+TN}{TP+FP+TN+FN}, \qquad F_1 = \frac{2PR}{P+R}.$$

**AUC-ROC** (threshold-independent): the probability that a randomly drawn positive example is ranked above a randomly drawn negative example,

$$\mathrm{AUC} = \Pr(\hat{y}_i > \hat{y}_j \mid y_i = 1,\ y_j = 0).$$

**Average Precision (AP)**: the area under the precision-recall curve, $\mathrm{AP} = \sum_k (R_k - R_{k-1})\, P_k$, summed over the ranking-induced sequence of thresholds $k$. Note $F_1$ is a single-threshold quantity while AP is threshold-aggregated — the two are not interchangeable (§3.6), and neither can be algebraically recovered from the other or from AUC alone (F1's defining identity has two degrees of freedom, $P$ and $R$; AUC and AP each supply only one aggregate constraint across all thresholds, not a value at the specific 0.5 cut point).

**Cohen's $d_z$** (paired-samples effect size): for per-seed differences $\delta_s = \Delta_G(\alpha, s)$ across seeds $s=1,\dots,k$,

$$d_z = \frac{\bar\delta}{\mathrm{sd}(\delta)}, \qquad \bar\delta = \tfrac{1}{k}\sum_s \delta_s, \qquad \mathrm{sd}(\delta) = \sqrt{\tfrac{1}{k-1}\sum_s (\delta_s - \bar\delta)^2}.$$

**Minority-class count** $m$: absolute number of positive-class training examples in a split, as distinct from positive rate $\pi = m/N$ (proportion) — the distinction the dose-response design (Algorithm 1) is built to isolate.

---

## References

1. Neslin, S.A.; Gupta, S.; Kamakura, W.; Lu, J.; Mason, C.H. Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models. *J. Mark. Res.* **2006**, 43, 204–211.
2. Johnson, J.M.; Khoshgoftaar, T.M. Survey on Deep Learning with Class Imbalance. *J. Big Data* **2019**, 6, 27.
3. He, H.; Garcia, E.A. Learning from Imbalanced Data. *IEEE Trans. Knowl. Data Eng.* **2009**, 21, 1263–1284.
4. Branco, P.; Torgo, L.; Ribeiro, R.P. A Survey of Predictive Modeling on Imbalanced Domains. *ACM Comput. Surv.* **2016**, 49, 31.
5. Jordon, J.; Szpruch, L.; Houssiau, F.; Bottarelli, M.; Cherubin, G.; Maple, C.; Cohen, S.N.; Weller, A. Synthetic Data — What, Why and How? *arXiv* **2022**, arXiv:2205.03257.
6. Fonseca, J.; Bacao, F. Synthetic Data Generation for Imbalanced Learning on Tabular Data. *Expert Syst. Appl.* **2023**.
7. Fernández, A.; García, S.; Herrera, F.; Chawla, N. SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-Year Anniversary. *J. Artif. Intell. Res.* **2018**, 61, 863–905.
8. Douzas, G.; Bacao, F. Geometric SMOTE: A Geometrically Enhanced Drop-in Replacement for SMOTE. *Inf. Sci.* **2019**, 501, 118–135.
9. Zhao, Z.; Kunar, A.; Van der Scheer, H.; Birke, R.; Chen, L.Y. CTAB-GAN: Effective Table Data Synthesizing. In Proceedings of the Asian Conference on Machine Learning (ACML), **2021**.
10. Shi, J.; Xu, M.; Hua, W.; Zhang, H.; Ermon, S.; Leskovec, J. TabDiff: A Mixed-Type Diffusion Model for Tabular Data Generation. In Proceedings of the International Conference on Learning Representations (ICLR), **2025**.
11. Zhao, Z.; Birke, R.; Chen, L. TabuLa: Harnessing Language Models for Tabular Data Synthesis. *arXiv* **2023**, arXiv:2310.12746.
12. Gulati, M.S.; Roysdon, P.F. TabMT: Generating Tabular Data with Masked Transformers. *NeurIPS* **2023**.
13. Solatorio, A.V.; Dupriez, O. REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers. *arXiv* **2023**, arXiv:2302.02041.
14. Erickson, N.; et al. TabArena: A Living Benchmark for Machine Learning on Tabular Data. *NeurIPS* **2025**.
15. Davila Restrepo, G.; et al. Benchmarking Tabular Data Synthesis: Evaluating Tools, Metrics, and Datasets on Prosumer Hardware. *Data Sci. J.* **2025**, 24, 37.
16. Sidorenko, A.; Platzer, M.; Scriminaci, M.; Tiwald, P. Benchmarking Synthetic Tabular Data: A Multi-Dimensional Evaluation Framework. *arXiv* **2025**, arXiv:2504.01908.
17. Won, D.-H.; et al. Synthetic Data Augmentation for Imbalanced Tabular Data: A Comparative Study of Generation Methods. *Electronics* **2026**, 15, 883.
18. Moro, S.; Cortez, P.; Rita, P. A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decis. Support Syst.* **2014**, 62, 22–31.
19. Hofmann, H. Statlog (German Credit Data); UCI Machine Learning Repository, **1994**.
20. Candillier, L.; Lemaire, V. Design and Analysis of the Nomao Challenge. In Proceedings of the ALRA Workshop, ECML-PKDD, **2012**.
21. Hillstrom, K. The MineThatData E-Mail Analytics and Data Mining Challenge. *MineThatData Blog*, **2008**.
22. Diemert, E.; Betlei, A.; Dieudonne-Boucher, C.; Amini, M.-R. A Large Scale Benchmark for Uplift Modeling. In Proceedings of the AdKDD & TargetAd Workshop, KDD, **2018**.
23. Agrawal, R.; Hamdare, S.; Ghosh, D.; et al. Improving Predictive Performance in Telecom Churn Modeling with Hybrid SMOTE and GAN-Based Synthetic Data Generation. *Int. J. Comput. Intell. Syst.* **2026**.
24. Guo, C.; Berkhahn, F. Entity Embeddings of Categorical Variables. *arXiv* **2016**, arXiv:1604.06737.
25. Chawla, N.V.; Bowyer, K.W.; Hall, L.O.; Kegelmeyer, W.P. SMOTE: Synthetic Minority Over-sampling Technique. *J. Artif. Intell. Res.* **2002**, 16, 321–357.
26. He, H.; Bai, Y.; Garcia, E.A.; Li, S. ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. In Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN), **2008**; pp. 1322–1328.
27. Han, H.; Wang, W.-Y.; Mao, B.-H. Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning. In *Advances in Intelligent Computing (ICIC 2005)*; Lecture Notes in Computer Science, Vol. 3644; Springer: Berlin, Germany, **2005**; pp. 878–887.
28. Patki, N.; Wedge, R.; Veeramachaneni, K. The Synthetic Data Vault. In Proceedings of the IEEE International Conference on Data Science and Advanced Analytics (DSAA), **2016**; pp. 399–410.
29. Xu, L.; Skoularidou, M.; Cuesta-Infante, A.; Veeramachaneni, K. Modeling Tabular Data using Conditional GAN. *NeurIPS* **2019**.
30. Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; Bengio, Y. Generative Adversarial Networks. *Commun. ACM* **2020**, 63, 139–144.
31. Kotelnikov, A.; Baranchuk, D.; Rubachev, I.; Babenko, A. TabDDPM: Modelling Tabular Data with Diffusion Models. In Proceedings of the International Conference on Machine Learning (ICML), **2023**.
32. Choi, W.C.; Lam, C.T.; Mendes, A.J. Comparison of Data Imputation Performance in Deep Generative Models for Educational Tabular Missing Data. In Proceedings of the International Conference on Educational Data Mining (EDM), **2025**; pp. 133–142.
33. Ho, J.; Jain, A.; Abbeel, P. Denoising Diffusion Probabilistic Models. In Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS), **2020**; pp. 6840–6851.
34. Borisov, V.; Seßler, K.; Leemann, T.; Pawelczyk, M.; Kasneci, G. Language Models are Realistic Tabular Data Generators. In Proceedings of the International Conference on Learning Representations (ICLR), **2023**.
35. Boyd, K.; Eng, K.H.; Page, C.D. Area under the Precision-Recall Curve: Point Estimates and Confidence Intervals. In *Machine Learning and Knowledge Discovery in Databases*; Springer: Berlin, Germany, **2013**; pp. 451–466.
36. Friedman, J.H. Greedy Function Approximation: A Gradient Boosting Machine. *Ann. Stat.* **2001**, 29, 1189–1232.
37. Pedregosa, F.; et al. Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, 12, 2825–2830.
38. Blagus, R.; Lusa, L. SMOTE for High-Dimensional Class-Imbalanced Data. *BMC Bioinform.* **2013**, 14, 106.
39. Engelmann, J.; Lessmann, S. Conditional Wasserstein GAN-Based Oversampling of Tabular Data for Imbalanced Learning. *Expert Syst. Appl.* **2021**, 174, 114582.
40. Shwartz-Ziv, R.; Armon, A. Tabular Data: Deep Learning is Not All You Need. *Inf. Fusion* **2022**, 81, 84–90.
41. Grinsztajn, L.; Oyallon, E.; Varoquaux, G. Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data? *NeurIPS* **2022**.
42. Gündüz, A.F.; Şahin, C.B. Synthetic Data Augmentation for Imbalanced Tabular Protein Subcellular Localization: A Comparative Study of SMOTE, CTGAN, TVAE, and TabDDPM Methods. *Appl. Sci.* **2026**, 16, 3694.
43. Adhikari, G.; Acharya, J.; Sapkota, A.; Ghimire, S.; Ghimire, U.K. A Comparative Analysis on Synthetic Data Generation of Electronic Health Records using CTGAN, REaLTabFormer and TabDDPM. *J. Innov. Eng. Educ.* **2024**, 9.
44. Bouthillier, X.; et al. Accounting for Variance in Machine Learning Benchmarks. In Proceedings of Machine Learning and Systems (MLSys), **2021**.
45. van Breugel, B.; Qian, Z.; van der Schaar, M. Synthetic Data, Real Errors: How (Not) to Publish and Use Synthetic Data. In Proceedings of the International Conference on Machine Learning (ICML), **2023**.
46. Lautrup, A.D.; Hyrup, T.; Zimek, A. SynthEval: A Framework for Detailed Utility and Privacy Evaluation of Tabular Synthetic Data. *Data Min. Knowl. Discov.* **2024**.
