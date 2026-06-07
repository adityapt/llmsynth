# Peer Review Panel — `paper2-empirical.md`

**Manuscript:** *When Class Imbalance Dominates: A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification*
**Review date:** 2026-06-07
**Reviewers:** EIC + R1 (statistical methods) + R2 (synthetic data) + R3 (applied marketing ML) + DA (Devil's Advocate)
**Target venue:** KDD Applied Data Science / NeurIPS Datasets & Benchmarks

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict:** Major revision required. The paper has a clear, testable hypothesis and a non-trivial empirical contribution (CTGAN > TabDDPM on marketing imbalance). Recommended for the applied track if revisions land cleanly. **Not currently a NeurIPS D&B candidate** — it's neither a new dataset nor a new benchmark protocol; it's an evaluation of existing methods on existing datasets. Target KDD Applied.

**Top-line issues that must be addressed before resubmission:**
1. **Statistical rigor gap** — claims like "the finding is not specific to a single downstream classifier" are made without paired tests, effect sizes, or multiple-comparison correction. The 5-seed CIs are reported but never used to test anything.
2. **Generality of the marketing finding is overstated** — n=2 marketing datasets (Hillstrom, Criteo) is not a basis for the abstract's "imbalance is the switch" framing. The hypothesis test, as stated, is underpowered.
3. **TabDDPM hyperparameters are at library defaults** — the headline "TabDDPM loses to CTGAN" claim is conditional on this. A reviewer will ask: did you try `N_iter=5000` or `num_timesteps=2000`? If not, the comparison is unfair.
4. **Single-seed §4.2 results are presented as if they were CIs** — the table shows "0.844 ± 0.015" but these are not 5-seed CIs; they appear to be cross-validation folds within a single random seed. The paper must be explicit.
5. **GReaT is evaluated only on Hillstrom** — yet the GReaT-fit variance finding (contribution #4) is framed as a general claim about LLM-based synthesis benchmarks. Reviewers will demand at least one additional dataset.

**Strengths to preserve:** clear hypothesis framing, honest reporting of artifacts (MLP Criteo), reproducibility infrastructure, the surprising and counterintuitive CTGAN-over-TabDDPM finding.

---

## REVIEWER 1 — Statistical Methods Reviewer

**Summary:** The paper claims a controlled empirical study but does not run controlled statistical tests on the central comparisons. The numerical results are correct (I spot-checked five of them against the CSVs) but the inferential apparatus is missing.

**Major points:**

**R1.1 (Critical) — No paired tests on the headline comparisons.**
The TabDDPM-vs-CTGAN claim (§4.5) reports differences of +4.40 pts (Hillstrom) and +2.96 pts (Criteo) with no statistical test. With 5 seeds, a paired t-test on per-seed AUC differences would directly establish whether the gap is significant. As reported, the reader has no way to distinguish "CTGAN reliably beats TabDDPM" from "CTGAN happened to win in this run." This must be addressed; add per-seed Δ tables, paired t-statistics, and Cohen's d_z.

**R1.2 (Critical) — Multiple comparison correction is absent.**
The paper makes roughly 30 distinct numerical claims (5 generators × 5 α × 2 datasets at the marketing core; 4 classifiers × 5 α × 3 generators × 2 datasets at the robustness layer). At α=0.05 per test, 30 tests yield roughly 1.5 expected false positives. The selection of "best α" per (generator, dataset, classifier) cell is exactly the kind of selection that inflates Type I error. Apply Benjamini-Hochberg FDR over an explicitly defined family of comparisons.

**R1.3 (Important) — The hypothesis test is informal.**
The abstract claims to "test the hypothesis that class imbalance is the primary and sufficient condition." But no formal hypothesis is stated and no test is run. A reasonable formalisation: H₀: gain | positive rate < 5% ≤ gain | positive rate ≥ 10%. The paper has the data to run a properly stratified comparison; it should.

**R1.4 (Important) — §4.2 confidence intervals are mislabeled.**
"0.844 ± 0.015" in Table §4.2 — is this 5-seed CI on per-seed AUC, or 5-fold CV CI within one seed? The paper says §6.2-6.4 of the prior draft used "single random seed," and §3.3 of this paper says "Benchmark dataset results (§4.1–4.3) ... are point estimates rather than confidence-interval estimates." But the table shows ± values. Resolve this contradiction.

**R1.5 (Important) — TSTR results are single-seed point estimates.**
Section 4.1 reports "TSTR gap −4.1%, −17.5%, −27.2%" without confidence intervals. The reader cannot judge whether the gap variance overlaps zero. Add 5-seed CIs to §4.1 or move it to an appendix as illustrative single-seed observations.

**Minor points:**

- §5.1 "the variance in [minority] count across splits is the primary driver of classifier instability" — this is a mechanistic claim that deserves an empirical check. You have the data: regress baseline AUC on minority count per split.
- The α* claim (§5.3) is descriptive across many cells but never tested as a model. A one-way ANOVA across α values per (dataset, generator) would establish the U-curve formally.

**Recommendation:** Major revision. The paper has the data to address every R1 issue; what's missing is the statistical analysis.

---

## REVIEWER 2 — Synthetic Data / Tabular ML Reviewer

**Summary:** The CTGAN-over-TabDDPM finding on imbalanced marketing data is genuinely interesting and could be a contribution worth publishing. However, the conditions under which this claim is established are too narrow for the strong framing the paper uses.

**Major points:**

**R2.1 (Critical) — TabDDPM hyperparameters are library defaults.**
The synthcity defaults (N_iter=2000, num_timesteps=1000) are reasonable for general tabular tasks but the Kotelnikov et al. paper itself uses different settings for class-imbalanced data. A reviewer would expect: (a) at least one TabDDPM run with extended training (N_iter=5000, 10000); (b) at least one run with class-conditional sampling enabled. As currently reported, the headline "TabDDPM loses to CTGAN" is conditional on a specific (and possibly suboptimal) configuration. Either tune, or substantially soften the claim.

**R2.2 (Critical) — Two marketing datasets is too thin a basis for the imbalance hypothesis.**
The paper uses Hillstrom (0.9%) and Criteo (0.2%) as the treatment group. Both are real, but n=2 is not sufficient to claim the imbalance condition is *necessary and sufficient*. The benchmark datasets all sit at positive rates ≥ 11.7%, leaving the 1%–10% range completely empty. The claim would be much stronger with one or two intermediate-imbalance datasets (e.g., KDD Cup 2009 Appetency at 1.8%, or a subsampled Bank Marketing at 5%).

**R2.3 (Important) — TabSyn omission is significant.**
Davila et al. (2025) report TabSyn at or above TabDDPM on augmentation utility. The paper acknowledges this in §6 ("TabSyn may still be worth evaluating") but the practitioner-facing conclusion (default to CTGAN) cannot survive if TabSyn changes the picture. At minimum, run TabSyn on one of the marketing datasets to bracket the claim.

**R2.4 (Important) — The mechanism explanation in §5.2 is speculative.**
"CTGAN's conditional vector explicitly targets the minority class during generation" is correct; the inference that this is why CTGAN beats TabDDPM is not directly tested. An ablation — running CTGAN without the conditional vector, or running TabDDPM with class-conditional sampling — would convert the speculation into evidence.

**R2.5 (Moderate) — Privacy and fidelity are not evaluated.**
For practitioners, the "default to CTGAN" recommendation has implications beyond utility. SMOTE generates near-duplicates of real minority examples and has known privacy risks. The paper should at least acknowledge this dimension; ideally a basic DCR (Distance to Closest Record) measurement would strengthen the practitioner recommendation.

**Minor points:**

- The 10K row cap is mentioned but the implications for the imbalance hypothesis at scale are not discussed. At full Criteo (13.9M rows), is the minority-class budget still the bottleneck?
- "≈ 45 min GPU" for TabDDPM and "≈ 2 min CPU" for CTGAN — these are anecdotal; report the actual wall-clock measurements from the experiment runs in a table.
- GReaT evaluation uses GPT-2 (117M). Recent LLM-based synthesizers use larger models (TabuLa, REaLTabFormer with GPT-3-tier backends). The "GReaT is unreliable" claim should be scoped to GPT-2.

**Recommendation:** Major revision. The contribution is real but currently overclaimed.

---

## REVIEWER 3 — Applied Marketing ML Reviewer

**Summary:** This paper is the kind of empirical study marketing data scientists actually need. The framing is right, the datasets are real, and the practical recommendation is actionable. My concerns are about generality and about making the recommendation maximally useful.

**Major points:**

**R3.1 (Important) — The recommendation should account for cost-sensitive alternatives.**
Before recommending synthetic augmentation, practitioners typically try cost-sensitive learning (class weighting) and threshold moving. These are cheaper and often deliver comparable gains. The paper should report (or at least cite) the gain from `class_weight='balanced'` on Hillstrom and Criteo as a baseline. If CTGAN-augmented GBC gains +12.9 pts on Criteo but `class_weight='balanced'` GBC gains +10 pts, the synthetic-data overhead may not be justified for many practitioners.

**R3.2 (Important) — n=2 marketing datasets is not enough to ground "marketing classification" claims.**
The paper's title and recommendation generalise to "marketing classification" but the evidence base is two datasets: one email response (Hillstrom 2008) and one display ad conversion (Criteo 2018). Both are old. Both are from anonymised or weakly-supervised settings. Consider adding (a) a customer churn dataset at low positive rate (e.g., a sub-1% subscription churn cohort), (b) at least one more recent marketing dataset, or (c) softening the recommendation to "Hillstrom-like and Criteo-like tasks."

**R3.3 (Important) — Uplift modeling is mentioned in future work but practitioners need it now.**
The marketing teams reading this paper care about uplift, attribution, and CLV — not just classification. The synthetic-data caveat for uplift (association-based generators corrupt causal structure) deserves a more prominent placement, possibly its own subsection. As written it's buried in §6.3 future work.

**R3.4 (Moderate) — The "20× compute cost" claim needs specifics.**
Practitioners care about deployment cost as well as fit cost. Add: (a) sampling time per α (not just fit time), (b) memory footprint at inference, (c) whether the trained generator needs to be retained for re-sampling or can be discarded.

**R3.5 (Moderate) — Operational concerns are not addressed.**
For production deployment, practitioners need to know: (a) can the same generator be reused across campaigns within the same domain, or must it be refit per task? (b) how does generator quality drift as the real data evolves? Brief mention in limitations would strengthen practical applicability.

**Minor points:**

- The "default to CTGAN at α ∈ {0.1, 0.3}" recommendation is too compressed. Practitioners need a recipe: which α to try first, how to validate the choice on holdout, whether to re-validate per task.
- The MLP Criteo artifact is honestly reported but could be re-run on cloud GPU to close the gap; the disclosure should mention this.

**Recommendation:** Major revision, lean toward accept. Of the three reviewers, I am the most positive on the contribution.

---

## DEVIL'S ADVOCATE

**The paper's central claim is a tautology dressed as an empirical finding.**

The paper says: *augmentation helps when there is class imbalance*. This is precisely the regime SMOTE (Chawla et al., 2002) and CTGAN (Xu et al., 2019) were designed for — both papers explicitly motivate the method by imbalanced classification. Finding that they help on imbalanced data and don't help on balanced data is not a discovery; it is a reproduction of the original motivation. The paper would be stronger if it claimed a sharper, more surprising result.

**Specific challenges:**

**DA.1 — Is the imbalance hypothesis novel?**
Chawla et al. (2002), Fonseca & Bacao (2023), Won et al. (2026), and the Davila et al. (2025) benchmark all establish that augmentation gains concentrate on imbalanced data. The paper cites these but does not credit them as the source of the hypothesis. As framed, contribution #1 is presented as new; it is at best a multi-generator confirmation.

**DA.2 — The CTGAN-vs-TabDDPM "head-to-head" is on a configuration that disadvantages TabDDPM.**
The fit-once-and-subsample strategy is fine for CTGAN (which generates conditionally and can handle the subsample efficiently). For TabDDPM (which generates unconditionally), subsampling at α < 1.0 means the practitioner is throwing away most of the diffusion model's output — but is paying full compute cost. A fair comparison would either (a) re-generate from TabDDPM at the smaller volume, or (b) report TabDDPM's per-α cost-adjusted performance.

**DA.3 — The GReaT-fit variance finding may not generalise.**
The GReaT-fit variance documentation (contribution #4) is interesting but is established on one dataset (Hillstrom) with one base model (GPT-2 117M). The paper extrapolates to "published GReaT benchmarks more broadly" but provides no evidence that the same variance pattern holds for TabuLa, REaLTabFormer, or larger LLM-based synthesizers. Recent work may have already addressed this with proper seeding.

**DA.4 — The "practitioner-facing decision rule" oversimplifies.**
"If positive rate < 5%, use CTGAN; otherwise skip" ignores the dataset-size dimension. At full Criteo (13.9M rows), the practitioner has hundreds of thousands of minority examples; the bottleneck is no longer minority data starvation. The rule should be scoped to the data-scarce regime the paper actually studies (cap at 10K).

**DA.5 — The MLP Criteo artifact is suspicious.**
The paper says the MLP baseline on Criteo "collapsed due to Metal Performance Shaders failures." But the paper accepts that MLP gained +65.6 pts under CTGAN (Table §4.6). If MLP training is broken, why is the augmented MLP result trusted? Either the artifact applies to all MLP runs (in which case all four MLP-on-Criteo rows should be excluded) or it doesn't (in which case the baseline should be re-evaluated). The current treatment is internally inconsistent.

**DA.6 — Existing CTGAN-over-TabDDPM evidence may already exist.**
The paper claims "first direct TabDDPM vs CTGAN head-to-head on real marketing data with 5-seed CI." Has the author done an exhaustive prior-art search? In particular, the Davila et al. (2025) benchmark — which the paper cites as motivation — may include the relevant comparison on some of their datasets. The novelty claim needs verification.

**Recommendation:** Reject in current form. The paper has a valid observation but is overclaiming its novelty and significance.

---

## SYNTHESIS — Recommended Revisions

Combining all five reviews, the manuscript needs:

### Must-have (critical for resubmission)

1. **Statistical apparatus** (R1.1, R1.2, R1.3): Add paired t-tests on per-seed Δ for every headline claim, with Cohen's d_z and BH-FDR correction over the explicit comparison family. Recompute confidence labels carefully (R1.4, R1.5).
2. **TabDDPM tuning** (R2.1, DA.2): At minimum, one TabDDPM run with extended training; ideally a class-conditional sampling variant. If results don't change, the claim survives; if they do, soften the abstract accordingly.
3. **MLP Criteo treatment** (DA.5): Either re-run MLP on cloud GPU, or remove all MLP-on-Criteo rows. The current half-included treatment is indefensible.
4. **Single-seed labeling** (R1.4): Make explicit which results are 5-seed CIs and which are single-seed point estimates. Cannot show "± 0.015" if it's not from 5 seeds on per-seed AUC.

### Should-have (substantially strengthens)

5. **Intermediate imbalance dataset** (R2.2): Add at least one dataset at 1%–10% positive rate to fill the gap between Bank Marketing (11.7%) and Hillstrom (0.9%).
6. **Cost-sensitive baseline** (R3.1): Report `class_weight='balanced'` on Hillstrom and Criteo as a comparison baseline; this is what practitioners try first.
7. **TabSyn comparison** (R2.3): At least one TabSyn run on a marketing dataset to bracket the diffusion-model claim.
8. **Mechanism ablation** (R2.4): CTGAN without conditional vector OR TabDDPM with conditional sampling, on one dataset.
9. **Soften novelty framing** (DA.1): Credit prior work (Chawla et al., Fonseca & Bacao, Won et al.) for the imbalance hypothesis; reframe the contribution as "multi-generator, multi-classifier verification with TabDDPM head-to-head."

### Nice-to-have (polish)

10. **Prior-art search** (DA.6): Document that no direct CTGAN-vs-TabDDPM 5-seed CI exists on Hillstrom or Criteo prior to this work.
11. **Operational details** (R3.4, R3.5): Add sampling time per α, memory footprint, generator reusability.
12. **GReaT scope** (R2 minor): Scope the variance claim to GPT-2 117M; do not extrapolate to larger LLM synthesizers.

---

## OVERALL VERDICT

**Score:** 4/10 (current form) → estimated 7/10 after must-have revisions

**Aggregate recommendation:** Major revision, lean toward accept-after-revisions at KDD Applied Data Science track if revisions are completed thoroughly. NOT ready for NeurIPS Datasets & Benchmarks (paper is not a benchmark or dataset contribution). NOT ready for the main NeurIPS/ICML track (no novel method or theoretical contribution).

The paper has a real finding worth publishing. The current draft does not yet do the finding justice — statistically, in scope, or in framing. With the revisions above, it is publishable in an applied track.

---

## Reviewer Disagreements

The five reviewers diverge on these points; the editor must adjudicate:

1. **Severity of the n=2 marketing dataset issue.** R2 sees this as critical; R3 sees it as important; DA sees it as fatal in combination with novelty concerns. Editor view: critical but addressable by either adding datasets or scoping claims explicitly.
2. **Novelty of the imbalance hypothesis.** DA argues it's tautological; R3 argues it's a useful confirmation; R2 is neutral. Editor view: contribution should be reframed as "imbalance-driven generator comparison" rather than "imbalance-as-driver discovery."
3. **TabDDPM fairness.** R2 and DA both demand more thorough TabDDPM tuning; R3 is less concerned. Editor view: at least one tuned run is non-negotiable for the comparison claim to survive review.
