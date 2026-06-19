# Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification

*Short paper submitted to INFORMS Workshop on Data Science 2026*
*[BLINDED FOR REVIEW — remove this line before submission]*

---

## Abstract

Class imbalance is a pervasive challenge in business classification — conversion rates below 1%, fraud rates below 0.5%, and rare-event prediction across customer cohorts. In these settings, classifiers trained on real data alone frequently fail due to extreme minority-example scarcity. Synthetic data augmentation is widely proposed as a remedy, but practitioners lack actionable guidance on when it helps and which generator to use. We evaluate five generators — GaussianCopula, CTGAN, SMOTE, TabDDPM, and GReaT (at GPT-2 and Mistral-7B scales) — across seven datasets spanning positive rates from 0.2% to 30%, with up to 10 seeds and four downstream classifiers. We find that large, consistent augmentation gains appear only when fewer than ~100 minority examples are available in training. We measure why: CTGAN generates 7–89× more minority-class rows than the natural distribution; TabDDPM and GaussianCopula mirror the real imbalance and provide no enrichment. Neither LLM backbone resolves this architectural limitation. The 1%–10% positive-rate transition region is untested and remains an open empirical question.

---

## 1. Introduction

Class-imbalanced datasets are pervasive across business domains. Fraud detection, medical diagnosis, insurance claims, and customer marketing all operate with rare positive events — sometimes below 1% of the dataset. When minority examples are scarce, standard classifiers default to predicting the majority class, yielding poor recall on the events that matter most. In marketing specifically, the cost of misclassification is direct and measurable: failing to identify a converting customer means lost revenue; missing a churning customer means reactive rather than preventive intervention (Neslin et al., 2006). Across financial services, healthcare, and digital marketing, accurate minority-class modeling has significant operational value.

Synthetic data augmentation — generating artificial training examples and mixing them with real data — is one of the most commonly used remedies for class imbalance (Chawla et al., 2002; He & Garcia, 2009; Johnson & Khoshgoftaar, 2019). The space of available generators has grown substantially: from interpolation-based oversampling (SMOTE), to conditional GANs (CTGAN), diffusion models (TabDDPM), and LLM-based synthesizers (GReaT). Aggregate benchmarks rank these generators on heterogeneous tabular tasks (Erickson et al., 2025; Davila et al., 2025), but do not answer the question practitioners actually face: *given my task at this positive rate, will augmentation help, and which generator should I use?*

This paper addresses that question through a controlled empirical study on seven datasets spanning positive rates from 0.2% to 30%. Table 1 makes the bottleneck concrete: in our experiments, datasets with fewer than ~100 minority training examples (positive rate ≤ 0.9%) show large augmentation gains of up to +12.9 AUC points; datasets with more than ~240 minority examples (positive rate ≥ 11.7%) show negligible gains — no generator exceeded +0.27 AUC points. This is not a class-imbalance story per se; it is a **minority-example scarcity** story. We note explicitly that the 1%–10% positive-rate region is not represented in our evaluation, and general claims about that transition region cannot be drawn from this study.

**Table 1 — Minority-example budget and baseline AUC by dataset**

| Dataset | Positive Rate | Train Rows | Minority Examples | Baseline AUC* |
|---|---|---|---|---|
| Criteo Display | 0.2% | 8,000 | **16** | 0.846 ± 0.228 |
| Hillstrom Email | 0.9% | 8,000 | **72** | 0.548 ± 0.092 |
| German Credit | 30.0% | 800 | 240 | 0.794 ± 0.044 |
| Bank Marketing | 11.7% | 12,000 | 1,404 | 0.928 ± 0.004 |
| Telco Churn | 26.6% | 5,626 | 1,497 | 0.844 ± 0.015 |
| Nomao Lead | 28.3% | 8,000 | 2,264 | 0.991 ± 0.001 |


\* Baseline AUC is calculated using a GradientBoosting classifier approach
## 2. Experimental Setup

**Generators.** We evaluate five synthetic data generators: GaussianCopula (Patki et al., 2016), CTGAN (Xu et al., 2019), SMOTE (Chawla et al., 2002), TabDDPM (Kotelnikov et al., 2023), and GReaT (Borisov et al., 2023) at two LLM backbone scales (GPT-2 117M and Mistral-7B 7B). All generators use library-default hyperparameters.

**Experimental pipeline.** For each (dataset, generator, seed) combination, we run three evaluation conditions:

1. *Baseline (TRTR):* Train on 80% real data, evaluate on 20% real holdout. This establishes the no-augmentation performance floor.
2. *TSTR (Train on Synthetic, Test on Real):* Train on fully synthetic data, evaluate on the same real holdout. This measures how faithfully a generator captures the real distribution. A large TSTR gap indicates synthetic data cannot replace real data.
3. *Augmentation sweep*†: Fit the generator on the real training set, generate synthetic rows, and train on the combined real+synthetic set. Evaluate on the real holdout and compare to baseline.

†**α (synthetic fraction)** controls how many synthetic rows are added relative to the real training set: α = n_synthetic / n_real. We sweep α ∈ {0.1, 0.2, 0.3, 0.5, 1.0}. For example, α=0.2 adds 20% as many synthetic rows as real rows; α=1.0 doubles the training set. We report the best-α result per generator.

**Downstream model.** Primary: GradientBoostingClassifier with n_estimators=100, max_depth=4 (Friedman, 2001; Pedregosa et al., 2011). Extended to Logistic Regression, Random Forest, and MLP for robustness. Each classifier is re-trained from scratch on the augmented training set; synthetic rows are only used for training, never for evaluation.

**Metric.** Primary metric: AUC-ROC (area under the receiver operating characteristic curve), which measures a classifier's ability to rank positive examples above negative ones, independent of a classification threshold. AUC ranges from 0 to 1; random guessing yields 0.5. We report gains in AUC points (×100 for readability): a gain of +0.1287 AUC is reported as +12.87 AUC points.

**Seeds and holdout design.** We use two holdout strategies depending on the experiment:

- *Augmentation sweep (CTGAN, SMOTE, TabDDPM):* Each seed determines an independent 80/20 stratified split — both the training set and the test set vary per seed. This standard approach estimates generalization across different data partitions and is appropriate when the full dataset (8,000–12,000 rows) provides a reliable holdout regardless of split.

- *GReaT small-n experiments:* The holdout is fixed once (random_state=42) before the seed loop and does not change across seeds or training sizes. Only the small training sample (n ∈ {50, 100, 200, 500, 1000, 2000}) varies per seed. This design is necessary because at n=50 training rows, a variable holdout would yield ~12 test rows — far too small for stable AUC estimation at 0.9% positive rate. A fixed holdout of 10,000 rows guarantees ~90 positive test examples, enabling reliable comparison across all training sizes and seeds on the same evaluation surface.

We report 95% confidence intervals (t-distribution on per-seed AUC values). For pairwise comparisons, we use a **paired t-test** on per-seed AUC differences — pairs are matched by seed, so the same random partition is used for both methods being compared. This removes cross-seed variability and isolates the generator effect. Effect size is reported as Cohen's d_z (mean paired difference / standard deviation of paired differences; d_z ≥ 0.8 is large). Multiple comparisons are corrected with Benjamini-Hochberg FDR at q=0.10. Compute: augmentation experiments run on a MacBook Pro M1 Pro 32 GB (CPU); TabDDPM and GReaT run on an NVIDIA H100 GPU cluster.

---

## 3. Core Finding: Minority-Example Scarcity Drives Augmentation Value

Across the five datasets with a positive rate of 11.7% or higher, each of which contains at least 240 minority examples, we find that no generator improves on the baseline by more than +0.27 AUC points at any value of α. The picture changes on the two marketing datasets. On Hillstrom (72 minority examples) and Criteo (16 minority examples), CTGAN and SMOTE recover between +5.7 and +12.9 AUC points. We did not evaluate datasets in the 1%–10% positive-rate range, and we caution against extrapolating our results into that region.

Criteo offers a useful illustration of what is at stake in this regime. Trained on real data alone, 7 of 10 MLP seeds failed to converge (AUC < 0.15). After CTGAN augmentation, all 10 seeds converged, with a mean AUC of 0.940 (95% CI). In this setting, augmentation is not a marginal refinement of an already-working model; it is what separates a classifier that trains from one that does not.

**Table 2 — Synthetic positive rate by generator (5 seeds × 8,000 generated rows)**

| Generator | Hillstrom synthetic rate | Criteo synthetic rate | Observation |
|---|---|---|---|
| Real training data | 0.90% | 0.30% | — |
| GaussianCopula | 0.96% ± 0.12% | 0.32% ± 0.06% | Mirrors real — no enrichment |
| TabDDPM | 0.89% ± 0.05% | 0.33% ± 0.09% | Mirrors real — no enrichment |
| **CTGAN** | **6.34% ± 0.21%** | **26.76% ± 1.18%** | **7–89× minority enrichment** |
| SMOTE | 100% (minority only) | 100% (minority only) | Minority targeted by design |

Table 2 points to a direct explanation for this performance gap. GaussianCopula and TabDDPM both sample at the natural positive rate of the data, so at a 0.2% positive rate roughly 99.8% of the rows they generate belong to the negative class, and the minority class is left no better represented than before. CTGAN behaves differently: its conditional vector produces 7–89× more positive-class rows than the training distribution contains. Because we observe these synthetic positive rates directly, the mechanism is measured rather than inferred.

For comparison, we also evaluated simple class reweighting (`class_weight='balanced'`) under the same 5-seed protocol, and CTGAN outperforms it by +7.55 AUC points on both datasets.

---

## 4. TabDDPM vs CTGAN

TabDDPM is currently the state-of-the-art generator on general tabular benchmarks (Davila et al., 2025), which makes it a natural point of comparison. We evaluated it at two training budgets: $N_{iter}$=2,000, the library default, and $N_{iter}$=10,000, a fivefold increase. On both Hillstrom and Criteo, CTGAN outperformed TabDDPM at each budget. Extending the training budget did not help TabDDPM and in fact widened the gap: at 10,000 iterations its performance on Hillstrom fell below baseline at every value of α. At this extended budget the CTGAN advantage on Hillstrom reaches an effect size of $d_z$=1.25 (p=0.049).

Table 2 again suggests why. TabDDPM samples unconditionally, at the natural positive rate of the data, and increasing the training budget did not change the sampling distribution we observed. This leads us to suspect that the limitation is architectural rather than a matter of insufficient optimization. The two generators also differ substantially in cost: CTGAN fits in roughly 2 minutes on CPU, whereas TabDDPM requires between 6 and 29 minutes on GPU.

** What CPU (M1 Pro 2022 32GB)

** What GPU (NVIDIA H100 x8)

---

## 5. LLM-Based Synthesis (GReaT)

We next turn to GReaT (Borisov et al., 2023), which takes a different approach by fine-tuning a language model on serialized tabular rows. We evaluated it at two backbone scales, GPT-2 (117M parameters) and Mistral-7B (7B parameters), in order to ask whether a larger language model helps. We tested GReaT on three datasets chosen to separate two conditions: anonymized features, where feature names carry no meaning (German Credit), and semantic features at different levels of class balance (Hillstrom, with extreme imbalance, and Telco, which is balanced).

**Table 3 — GReaT vs CTGAN: AUC gain over baseline ± 95% CI (5 seeds)**

| Dataset | Condition | GPT-2 GReaT | Mistral-7B | CTGAN reference |
|---|---|---|---|---|
| German Credit | Anonymized, n=100 | −7.02 ± 3.33 pts | −5.59 ± 5.31 pts | +0.27 pts |
| German Credit | Anonymized, n=500 | −3.00 ± 1.90 pts | −0.38 ± 2.52 pts | +0.27 pts |
| Hillstrom | Semantic + extreme imbalance, n=50 | +2.25 ± 4.69 pts | —† | +5.75 pts |
| Hillstrom | Semantic + extreme imbalance, n=100 | +1.15 ± 5.25 pts | +1.20 ± 5.99 pts‡ | +5.75 pts |
| Hillstrom | Semantic + extreme imbalance, n=2000 | **−6.87 ± 4.61 pts** | −1.45 ± 4.35 pts | +5.75 pts |
| Telco | Semantic + balanced, n=100 | −1.38 ± 3.44 pts | −2.15 ± 3.43 pts | +0.28 pts |

†n=50 Mistral-7B on Hillstrom: 4/5 seeds failed to generate parseable rows (extreme imbalance + very small n). ‡Only 3 valid seeds; 2 seeds failed entirely.

Three findings emerge from Table 3. First, on the anonymized dataset (German Credit), both backbone scales hurt performance consistently. This is what we would expect if the value of a language-model prior comes from the meaning of feature names, since that meaning is absent here. Second, on Hillstrom, where imbalance is extreme, GReaT does not merely fail to help but actively harms performance as n grows (GPT-2: −6.87 points at n=2,000, $d_z$=−4.40, FDR-significant, p=0.006). The most plausible reading is that, at a 0.9% positive rate, additional LLM-generated rows dilute the minority class rather than enrich it. Third, the larger backbone offers little: Mistral-7B is marginally less harmful than GPT-2 on German Credit at large n (−0.38 vs −3.00 points at n=500), but on Hillstrom and Telco both models fall below baseline in most conditions. On the extreme-imbalance datasets that matter for marketing, neither scale comes close to CTGAN.

We read this result through the same lens as Table 2. Like TabDDPM, GReaT samples from the joint distribution without any conditioning on the minority class, so the synthetic positive rate it produces tends to mirror the training distribution (about 0.9% on Hillstrom) whether the backbone is GPT-2 or Mistral-7B. Scaling up the language model leaves this sampling behavior unchanged, which is consistent with the null effect we observe.

---

## 6. Recommendation and Limitations

**Practitioner guidance:**

| Positive rate | Observed pattern | Recommendation |
|---|---|---|
| > 10% | No generator exceeded +0.27 pts | Skip augmentation |
| 1%–10% | **Not tested in this study** | - |
| 0.5%–1% | CTGAN/SMOTE +5–6 pts | Validate experimentally |
| < 0.5% | CTGAN/SMOTE +12–13 pts | Strongly consider CTGAN |

**Limitations:** (1) Only two extreme-imbalance datasets tested (Hillstrom, Criteo); the 1%–10% transition region is unsampled. (2) All experiments cap at n=10,000; at full dataset scale the minority-example budget is larger and gains may diminish. (3) Generator hyperparameters use library defaults; tuned TabDDPM may narrow the CTGAN gap. (4) Privacy and operational costs not evaluated.

**Future direction — context-conditioned LLM synthesis.** Our LLM results use GReaT, which fine-tunes on serialized rows and samples *unconditionally*; the null effect of scaling (GPT-2 → Mistral-7B) suggests added prior knowledge alone cannot overcome unconditional sampling under extreme imbalance. A different lever is untested here: prompting an instruction-tuned LLM to generate **specifically minority-class** rows, supplied in-context with schema-level metadata (marginal statistics, feature correlations, domain semantics, signal sparsity). This reframes the LLM's role from learning the joint to a CTGAN-like conditional generator, and is the form of "context engineering" most likely to close the gap — a hypothesis we leave to future work.

---

## References

Borisov, V., et al. (2023). Language models are realistic tabular data generators. Proceedings of ICLR 2023.

Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.

Dávila Restrepo, G., et al. (2025). Benchmarking tabular data synthesis. Data Science Journal.

Erickson, N., et al. (2025). TabArena. Advances in Neural Information Processing Systems.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5).

Kotelnikov, A., et al. (2023). TabDDPM. Proceedings of ICML 2023.

Patki, N., et al. (2016). The Synthetic Data Vault. Proceedings of IEEE DSAA.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Xu, L., et al. (2019). Modeling tabular data using conditional GAN. Advances in Neural Information Processing Systems.

Hillstrom, K. (2008). MineThatData e-mail analytics challenge.

Diemert, E., et al. (2018). A large scale benchmark for uplift modeling. Proceedings of the KDD AdKDD Workshop.
