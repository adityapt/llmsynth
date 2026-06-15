# Peer Review — `paper3-llm-vs-other.md`

**Manuscript:** *Scale Does Not Fix LLM-Based Tabular Synthesis: A Controlled Evaluation of GReaT Across Model Sizes and Data Regimes*
**Review date:** 2026-06-14
**Target venue:** NeurIPS Datasets & Benchmarks / NeurIPS Table Representation Learning Workshop
**Reviewers:** EIC + R1 (statistical methods) + R2 (LLM/NLP) + R3 (tabular ML / applications) + DA

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict: Weak accept at NeurIPS Table Representation Learning Workshop; major revision for NeurIPS Datasets & Benchmarks.**

The paper's central question — does LLM scale fix GReaT's failure modes? — is timely and the answer (no, not on imbalanced data) is useful to the field. However, the evidence base for the scale claim is weaker than the writing suggests, and the architectural explanation (conditional vs unconditional sampling) is compelling but untested. The paper needs either the ablation or a substantially softened claim.

**Score: 5.5/10 in current form. Estimated 7/10 after the ablation experiment (one experiment away from a strong NeurIPS D&B submission).**

**Critical issues:**
1. **Scale is confounded with architecture** — GPT-2 and Mistral-7B are different model families, not just different sizes. A reviewer will immediately flag this.
2. **Central architectural claim has no ablation** — the conditional vs unconditional argument is the most interesting part of the paper and is entirely speculative.
3. **3 datasets × 5 seeds for the scale comparison** — thin basis for a "scale does not fix" conclusion.
4. **No statistical tests on GPT-2 vs Mistral-7B** — the comparisons in §4.1 are reported without any paired tests.

---

## REVIEWER 1 — Statistical Methods

**Summary:** The paper makes headline claims about model scale without running a single statistical test on the scale comparison. The §4.1 tables report means and CIs but no paired tests of GPT-2 vs Mistral-7B at any (dataset, n) cell.

**R1.1 (Critical) — No statistical tests on the scale comparison.**
Table §4.1 shows GPT-2 gain vs Mistral-7B gain but never asks: is the difference statistically distinguishable from zero? With 5 seeds, the paired t-test has limited power, but you have the data. For the headline finding "scale does not fix failure modes," you need to show either (a) the Mistral-7B improvement is not statistically significant, or (b) even when significant, it falls short of CTGAN by a statistically distinguishable margin.

**R1.2 (Critical) — The n=50 Mistral-7B Hillstrom result is reported with NaN CI.**
"n=50 Mistral-7B on Hillstrom: only 1 valid seed (4/5 sampling failures)." This is arguably the most striking data point in the paper — larger model fails MORE at small n — but it's presented as a footnote with an imputed number. Either report it properly (with caveat on n=1) or exclude it and discuss the failure rate separately.

**R1.3 (Important) — Fit variance claim for Mistral-7B is inferred, not measured.**
§4.4 says "This is independent of model size: the same non-determinism affects Mistral-7B." This is an architectural inference, not an empirical observation. You ran GReaT twice with GPT-2. You did not run GReaT twice with Mistral-7B. The claim should be scoped to GPT-2 with the inference noted.

**R1.4 (Minor) — Mixed precision across model scales.**
GPT-2 was trained with fp16; Mistral-7B with bf16. This is a hardware-driven difference (H100 requires bf16) but it potentially affects training dynamics. Should be acknowledged.

**Recommendation:** Major revision — R1.1 is blocking.

---

## REVIEWER 2 — LLM/NLP Reviewer

**Summary:** The paper conflates model scale with model architecture. GPT-2 and Mistral-7B are not points on the same scaling curve — they have different architectures, tokenizers, training data, and alignment procedures. The "scale" framing is not supported by the experimental design.

**R2.1 (Critical) — Scale and architecture are confounded.**
GPT-2 uses a basic transformer decoder (2019). Mistral-7B uses grouped-query attention, sliding window attention, and was trained on ~6 trillion tokens with instruction tuning. The improvement on Telco and marginal improvement on Hillstrom could be attributed to any of: scale, architecture, training data diversity, or instruction following. A valid scale comparison requires holding architecture constant — e.g., GPT-2 (117M) → GPT-2 XL (1.5B) → GPT-Neo (2.7B, same family) → GPT-J (6B, same family). Using a completely different model family (Mistral) conflates everything.

**Consequence:** The framing "Scale Does Not Fix GReaT's Failure Modes" is not supportable. What the data supports is: "Mistral-7B (a different, larger model) provides marginal inconsistent improvement over GPT-2 in the GReaT framework." That's a weaker and more honest claim.

**R2.2 (Critical) — The architectural explanation is untested.**
§4.3 argues the failure is due to unconditional sampling: at 0.2% positive rate, LLMs sample 99.8% negative-class rows. This is the most interesting scientific contribution in the paper — but it is not tested. The obvious ablation: prompt Mistral-7B or GPT-2 with "target is 1" and see if performance improves. If class-conditional prompting fixes Mistral-7B on Hillstrom, the architecture claim is confirmed; if not, something else explains the gap. Without this ablation, the claim is an unfalsified hypothesis.

**R2.3 (Important) — Missing baseline: class-conditional LLM sampling.**
The paper argues CTGAN's conditional vector is the advantage. The natural next step is to test class-conditional LLM synthesis. GReaT's `guided_sampling` does partial conditioning — you can prompt with partial row specifications including "target is 1". This experiment takes one afternoon and would substantially strengthen or qualify the architectural claim.

**R2.4 (Important) — REaLTabFormer and TabuLa are named but not evaluated.**
The related work mentions these as extensions of GReaT but they are not in the experiments. A reviewer at NeurIPS D&B will ask why the comparison is GPT-2 vs Mistral-7B when the actual frontier of LLM-based tabular synthesis has moved to these models.

**Recommendation:** Major revision — R2.1 and R2.2 are the two most important issues in the paper. R2.1 requires reframing the claim; R2.2 requires one ablation experiment.

---

## REVIEWER 3 — Tabular ML / Applications

**Summary:** The practical contribution is clear and useful. The data is real. The finding that LLMs underperform CTGAN on imbalanced marketing data is actionable. But the paper is trying to be both an AI methods paper and an applied evaluation paper, and the tension shows.

**R3.1 (Important) — Three datasets is too thin for the scale claim.**
The scale comparison covers German Credit, Hillstrom, Telco. The pattern is:
- German Credit: both fail (anonymized features)
- Hillstrom: Mistral-7B marginally better (inconsistent)
- Telco: Mistral-7B consistently better

From three datasets, the paper cannot establish "scale doesn't help" — at best it establishes "scale doesn't help on German Credit and Hillstrom; it does help on Telco." The title overclaims relative to the evidence.

**R3.2 (Important) — The positive finding on Telco undermines the headline.**
Mistral-7B outperforms GPT-2 consistently on Telco Churn. This is actually evidence that scale DOES help in the right conditions (semantic features + balanced classes). The paper buries this finding and treats it as a footnote. A balanced framing would be: "scale helps when LLM priors are operative (semantic features, balanced classes) but not in the extreme imbalance regime." That's a more nuanced and ultimately more interesting finding.

**R3.3 (Moderate) — Hillstrom n=50 Mistral failures need explanation.**
4/5 seeds failed to generate parseable rows with Mistral-7B at n=50. This is striking — why does the larger model fail more at very small training sizes? The paper attributes it to "more brittle" behavior but doesn't explain the mechanism. Possible explanation: Mistral-7B's longer token sequences and different tokenizer interact badly with be-great's guided sampling at very short training data. This deserves a paragraph.

**Recommendation:** Major revision — reframe the headline claim, give the Telco positive finding its due.

---

## DEVIL'S ADVOCATE

**The paper's central conclusion may be backwards.**

The data shows:
- Semantic + balanced (Telco): Mistral-7B consistently outperforms GPT-2 → **scale helps**
- Semantic + imbalanced (Hillstrom): Mistral-7B marginally better at some n → **scale partially helps**
- Anonymized + balanced (German Credit): both fail → scale irrelevant

The regime where scale doesn't help is the extreme imbalance regime — and the paper's own analysis in §4.3 correctly identifies that this is an architectural problem (unconditional sampling), not a scale problem. An accurate title would be: **"Unconditional Sampling, Not Scale, Is the Bottleneck for LLM-Based Tabular Synthesis Under Extreme Imbalance."**

"Scale Does Not Fix GReaT" is catchy but misleading — it implies scale is irrelevant, when the data shows it is relevant in the right conditions.

**DA.1 — The confound between scale and architecture (R2.1) is the fundamental problem.**
Without a same-family scale comparison (GPT-2 → GPT-J → GPT-NeoX or similar), the paper cannot attribute results to scale. This is not a minor limitation; it invalidates the framing.

**DA.2 — The paper has three distinct contributions that deserve separate framing:**
1. LLM-based synthesis vs statistical generators in the imbalanced regime → belongs in paper2 (empirical evaluation)
2. Scale comparison within GReaT → interesting but needs better experimental design
3. Fit variance documentation → methodological contribution to benchmark reproducibility

Trying to package all three as a single "scale doesn't fix" narrative creates incoherence.

**Recommendation:** Reject current framing. Rewrite as either (a) a focused benchmark reproducibility paper (fit variance as the primary contribution) or (b) accept the conditional framing ("scale helps in the right conditions, not in imbalanced regime") and add the architectural ablation.

---

## SYNTHESIS

### Critical issues (must address for any venue)

1. **R2.1 / DA.1 — Scale-architecture confound.** Either (a) run same-family scale comparison (GPT-2 → GPT-Neo 2.7B same architecture), or (b) reframe: "Mistral-7B vs GPT-2 in GReaT framework" not "scale doesn't fix." The reframe is cheaper.

2. **R2.2 — Architectural claim untested.** Run GReaT with class-conditional prompting on Hillstrom. One afternoon experiment. If it works, the paper gets much stronger. If it doesn't, the claim needs hedging.

3. **R1.1 — No paired tests on GPT-2 vs Mistral-7B.** Add paired t-tests for each (dataset, n) comparison. Data exists; tests take 5 minutes.

4. **R3.2 / DA — The Telco positive finding contradicts the headline.** Either add it to the abstract as "scale helps when priors are operative, but not under extreme imbalance" or explain why Telco is out of scope.

### Important but not blocking

5. **R1.2 — n=50 Mistral Hillstrom NaN CI.** Exclude from table with explanation, or report properly.
6. **R1.3 — Fit variance inference for Mistral-7B.** Scope claim to GPT-2 or run second Mistral fit.
7. **R3.1 — Three datasets.** Either add one more or scope claims explicitly.
8. **R3.3 — n=50 Mistral failure mechanism.** One paragraph.

### What would make this a strong NeurIPS D&B paper

The single most impactful addition: **class-conditional prompting ablation** (R2.2 / R2.3). If prompting with "target is 1" improves Mistral-7B on Hillstrom to near-CTGAN performance, the paper becomes:

> *"The failure of GReaT in the extreme imbalance regime is not a scale problem but a sampling protocol problem. Class-conditional prompting recovers most of the gap with CTGAN, while scaling the backbone alone does not."*

That is a genuinely novel and important finding that NeurIPS D&B would publish.

---

## OVERALL VERDICT

**Score: 5.5/10 (current) → estimated 7.5–8/10 after class-conditional ablation + reframing.**

**As NeurIPS D&B:** Major revision required. The scale-architecture confound and untested architectural claim are blocking.
**As NeurIPS Table Representation Learning Workshop:** Weak accept. The fit variance and LLM-vs-CTGAN comparison are novel contributions even without the ablation.
**As a companion to paper2 (empirical evaluation paper):** The LLM-scale results fit naturally into paper2 §4.7 and do not need a standalone paper. The standalone paper needs the ablation to earn its place.

**The one experiment that changes everything:** run GReaT with `guided_sampling=True` and a partial row conditioning "target is 1" on Hillstrom. If Mistral-7B with class-conditional prompting approaches CTGAN, you have a strong NeurIPS D&B paper. If it doesn't, you have strong evidence that the failure is deeper than sampling protocol, which is also a publishable finding.
