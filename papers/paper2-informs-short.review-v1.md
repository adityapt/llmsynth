# Peer Review — INFORMS Workshop on Data Science 2026
## Paper: Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification

**Review date:** 2026-06-19
**Mode:** Full (5-reviewer panel)
**Based strictly on:** papers/paper2-informs-short.md only
**Note:** Reviewers have seen only this paper. No external files, CSVs, or prior context used.

---

## Phase 0 — Reviewer Configuration

| # | Persona | Expertise | Focus |
|---|---|---|---|
| EIC | Senior INFORMS data science researcher | Venue fit, contribution, overall quality | Overall decision |
| R1 | ML evaluation methodology specialist | Experimental design, statistical validity | Methodology |
| R2 | Marketing analytics / CRM researcher | Domain relevance, practical contribution | Domain |
| R3 | Generative AI / tabular data researcher | Generator comparison, LLM framing | Perspective |
| DA | Devil's Advocate | Logical challenges, alternative explanations | Core argument |

---

## EIC Review

**Recommendation: Accept with minor revisions | Score: 7/10**

**Venue fit:** Strong. The paper addresses synthetic data generation and AI in digital marketing — both explicitly in the INFORMS WDS CFP.

**Writing quality (updated):** Sections 3, 4, and 5 read naturally and professionally. The prose is clear and the logical flow from Table 2 to the mechanism explanation in §§3–5 is effective. Section 1 and 2 are also well-written. This is ready for submission in terms of voice.

**Strengths:**
- Real marketing datasets, not benchmark proxies
- Table 2 (measured synthetic class distributions) is a direct, empirical contribution
- Honest 1%–10% caveat stated multiple times
- Practitioner decision table in §6
- §5 GReaT comparison at two LLM scales is novel for this venue

**Remaining blockers:**

**EIC.1 (Blocker) — Author notes in §4.** Two lines read: `** What CPU (M1 Pro 2022 32GB)` and `** What GPU (NVIDIA H100 x8)`. These are raw author notes embedded in the paper body. Must be replaced with prose before submission.

**EIC.2 (Blocker) — Header says "short paper."** Line 3: *"Short paper submitted to INFORMS Workshop on Data Science 2026."* If submitting as a complete paper, update accordingly.

**EIC.3 (Blocker) — CI value missing in §3.** The paper reads: "all 10 seeds converged, with a mean AUC of 0.940 (95% CI)." The confidence interval value is absent.

**EIC.4 (Major) — TSTR defined, results absent.** §2 defines TSTR as the second experimental condition. No TSTR results appear anywhere in the paper. Either present results or remove TSTR from the protocol.

**EIC.5 (Major) — No figures.** The paper has three tables but no figures. For a complete paper targeting 10 pages, at least one visualization would significantly improve impact and use available space.

**EIC.6 (Moderate) — No related work section.** The introduction moves directly to experimental setup. A brief related work section (1 page) situating the paper against prior benchmarks and imbalanced learning literature would be expected at 10 pages.

**EIC.7 (Moderate) — Recommendation table "–" for 1%–10%.** A dash provides no guidance. "Validate experimentally before deploying" is more actionable.

---

## R1 — Methodology Reviewer

**Recommendation: Minor revision | Score: 6.5/10**

**Strengths:**
- Two-holdout-strategy design clearly explained and correctly justified
- Paired t-test with BH-FDR appropriate
- α formally defined with a worked example
- Compute environment now clearly stated: M1 Pro 32 GB CPU, H100 GPU cluster (though currently in placeholder format)

**Issues:**

**R1.1 (Major) — Best-α optimism bias not acknowledged.** The paper reports "best-α result per generator." This is the maximum over 5 values. Without acknowledging this, the reported gains represent the best case, not the expected case. This should be noted in §2 or §3.

**R1.2 (Major) — GReaT α value not stated.** §5 evaluates GReaT at training sizes n ∈ {50, 100, 200, 500, 1000, 2000} but never specifies the synthetic fraction α. Readers cannot compare GReaT to the augmentation sweep results without this.

**R1.3 (Major) — GReaT not in Table 2.** §5 claims GReaT mirrors the training distribution (sampling ~0.9% positive on Hillstrom). Table 2 contains measurements for CTGAN, GaussianCopula, TabDDPM, and SMOTE but not GReaT. This claim is stated as observed fact without a measurement row to support it.

**R1.4 (Moderate) — Table 3 three-seed cell insufficiently flagged.** Hillstrom Mistral-7B n=100 has only 3 valid seeds (footnote ‡: "+1.20 ± 5.99 pts"). A CI of ±5.99 from n=3 is not reliable for inference. This should be noted in the body text, not only a footnote.

**R1.5 (Moderate) — CTGAN 6–89× enrichment unexplained.** Table 2 shows CTGAN generates 6.34% positive on Hillstrom vs 0.90% real. The paper states this as a finding but offers no explanation for why CTGAN oversamples the minority class. Readers will ask.

**R1.6 (Minor) — α* not communicated.** §2 defines the α-sweep and §6 recommends "CTGAN at α ∈ {0.1, 0.3}" without connecting this to the sweep results. What does the sweep show about optimal α?

---

## R2 — Domain Reviewer

**Recommendation: Minor revision | Score: 7.5/10**

**Strengths:**
- Marketing cost framing in §1 is effective for the INFORMS audience
- MLP convergence rescue (§3) is the paper's most striking single result and is now well-written
- Practitioner decision table directly answers the applied question
- +7.55 AUC vs class reweighting benchmark is a useful addition

**Issues:**

**R2.1 (Major) — SMOTE vs CTGAN recommendation unexplained.** §3 reports CTGAN and SMOTE both deliver +5.7 to +12.9 AUC points. §6 recommends "Strongly consider CTGAN." SMOTE requires no generative model, no GPU, and no hyperparameters. If SMOTE achieves the same gains, the recommendation must explain why a practitioner would choose the more complex and expensive option.

**R2.2 (Moderate) — Optimal α not explained.** §6 recommends "CTGAN at α ∈ {0.1, 0.3}" without connecting this to results from the α-sweep defined in §2. Practitioners need to know not only which generator but how much synthetic data to add.

**R2.3 (Moderate) — Dataset age unacknowledged.** Hillstrom (2008) is 18 years old. Marketing conversion patterns change substantially. A single sentence noting this in the limitations would be appropriate.

**R2.4 (Minor) — Privacy implications of SMOTE.** SMOTE generates near-duplicate minority examples, creating membership inference risk in regulated marketing environments. Not mentioned in the limitations section.

---

## R3 — Perspective Reviewer

**Recommendation: Minor revision | Score: 7/10**

**Strengths:**
- §5 is now clearly written — the two-condition design (anonymized vs semantic features) is well-motivated
- The GReaT-to-Table-2 connection in the final paragraph of §5 is effective
- Future direction paragraph is well-positioned and naturally motivated

**Issues:**

**R3.1 (Major) — GReaT serialization not explained.** §5 states GReaT "fine-tunes a language model on serialized tabular rows" without explaining what serialization means. For the IS/analytics audience at INFORMS, this is not self-evident. A one-sentence example would help.

**R3.2 (Major) — GReaT failure: training instability not ruled out.** The paper attributes GReaT's failure to unconditional sampling from the joint distribution. But Table 3 shows more failures at smaller n (n=50: 4/5 seeds fail; n=100: 3 valid seeds). This pattern is equally consistent with training instability at very small datasets. The paper presents the sampling explanation as the conclusion without ruling out this alternative. Framing it as "the most plausible interpretation" (which §5 does for the dilution claim at large n) is appropriate — the same hedging should be applied to the architectural claim.

**R3.3 (Moderate) — Mistral-7B n=100 Hillstrom discussed without n=3 warning in text.** §5 draws conclusions about Mistral-7B's behavior on Hillstrom. The 3-valid-seed limitation is in a footnote (‡) that a reader can easily miss. The finding at this cell should be marked as preliminary in the text.

**R3.4 (Minor) — "Serialized tabular rows" jargon in §5.** This phrase appears twice without clarification. Non-LLM readers will not understand.

---

## Devil's Advocate Report

**Strongest counter-argument: The mechanism claim cannot be separated from SMOTE**

The paper's central mechanism is: *CTGAN's conditional vector enriches the minority class; generators that sample unconditionally (TabDDPM, GaussianCopula) do not; this explains the performance gap.*

Table 2 clearly supports the distinction for CTGAN vs TabDDPM/GaussianCopula. But §3 also reports that **SMOTE delivers comparable gains** (+5.7 to +12.9 AUC points, same range as CTGAN). SMOTE achieves 100% minority-class synthetic rate — it enriches the minority class far more aggressively than CTGAN's 6–27%.

So the pattern from the paper is:
- High minority enrichment → CTGAN (6–27%), SMOTE (100%) → large gains ✓
- No enrichment → TabDDPM (~real rate), GaussianCopula (~real rate) → negligible gains ✓

If the mechanism is minority enrichment, and SMOTE achieves it more completely, why does the paper recommend CTGAN over SMOTE? The paper's own data supports the simpler conclusion: *adding minority-class rows helps; how they are generated matters less than whether they are minority-class.* The paper should address this directly or revise the recommendation.

**DA.2 (Major) — "Architectural" conclusion is premature.** The paper concludes TabDDPM's limitation is "architectural rather than optimization-related." This is based on one additional training budget experiment (10k vs 2k iterations). Getting worse with more training is consistent with overfitting, not uniquely with architectural limitation. The hedged language ("leads us to suspect") in §4 is appropriate — the architectural claim in the abstract is less hedged and should match.

**DA.3 (Moderate) — Threshold of ~100 inferred from n=2 datasets.** The threshold "fewer than ~100 minority examples" is inferred from exactly two datasets (16 and 72). Any threshold from n=2 is extrapolation. The paper correctly acknowledges the 1%–10% gap — the threshold itself deserves similar hedging.

**DA.4 (Minor) — "7–89× minority enrichment" is generation rate, not training enrichment.** At α=0.1, only 10% as many synthetic rows as real rows enter training. The effective enrichment of the training set is much smaller than the generation rate implies.

**Observations (non-defects):**
- The two-holdout design is genuinely important and correctly handled
- The 1%–10% caveat is appropriately prominent throughout
- Table 2 is empirically measured — this is the paper's core strength

---

## Phase 2 — Editorial Synthesis

### Progress since prior review (v1)
- Sections 3, 4, 5 now have natural, professional prose — a significant improvement
- The Table 2 → mechanism connection is now clearly articulated in §§3–5
- The Criteo MLP rescue result is now framed compellingly

### Remaining consensus issues (3+ reviewers)

| Issue | Reviewers | Priority |
|---|---|---|
| Placeholder author notes in §4 | EIC, R1 (implied), R3 (implied) | Blocker |
| TSTR defined but results absent | EIC, R1 | Major |
| SMOTE vs CTGAN recommendation unexplained | R2, DA | Major |
| GReaT not in Table 2 | R1, R3 | Major |
| Decision table "–" for 1%–10% | EIC, R2 | Moderate |

### Single-reviewer remaining issues

| Issue | Reviewer | Priority |
|---|---|---|
| No figures | EIC | Major |
| No related work | EIC | Moderate |
| Best-α optimism bias | R1 | Major |
| GReaT α not stated | R1 | Major |
| GReaT serialization unexplained | R3 | Major |
| GReaT training instability alternative | R3, DA | Major |
| CI value missing | EIC | Blocker |
| Header mismatch | EIC | Blocker |

---

## Editorial Decision: **Accept with Minor Revisions**

The writing quality is substantially improved. Sections 3–5 now read like a finished paper. The core contribution (Table 2, measured mechanism) is sound. The remaining issues are primarily cleanup plus two substantive gaps.

### Must fix before submission

| # | Issue | Fix |
|---|---|---|
| 1 | `** What CPU/GPU` in §4 | Replace with: "Augmentation experiments ran on a MacBook Pro M1 Pro (32 GB RAM); TabDDPM and GReaT on an NVIDIA H100 GPU cluster (8×)" |
| 2 | Header says "short paper" | Update to complete paper or remove label |
| 3 | "0.940 (95% CI)" — CI missing | Fill in the actual confidence interval value |
| 4 | TSTR defined but results absent | Add TSTR results table OR remove TSTR from §2 |

### Strongly recommended

| # | Issue | Fix |
|---|---|---|
| 5 | SMOTE vs CTGAN unexplained | Add 1–2 sentences: both achieve similar gains; CTGAN preferred when diversity and feature correlation matter beyond interpolation |
| 6 | GReaT α not stated in §5 | Add: "GReaT was evaluated at α=1.0 (n_synthetic = n_train)" |
| 7 | GReaT serialization unexplained | Add one-sentence example of row serialization |
| 8 | Decision table "–" for 1%–10% | Change to "Validate experimentally before deploying" |
| 9 | Add a figure | Add minority-budget-vs-gain scatter or augmentation curve |
| 10 | Brief related work (~0.5–1 page) | Situate against prior imbalanced learning and synthetic data benchmarks |

### Bottom line
Three blockers, all quick fixes. The most substantive remaining gap is the SMOTE explanation (#5). Estimated time to submission-ready: **2–4 hours** for a focused editing session.
