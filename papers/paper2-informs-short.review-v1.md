# Peer Review — INFORMS Workshop on Data Science 2026
## Paper: Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification

**Review date:** 2026-06-19
**Mode:** Full (5-reviewer panel)
**Reviewer constraint:** Based strictly on papers/paper2-informs-short.md only. No external files, CSVs, or prior context used.

---

## Phase 0 — Reviewer Configuration

| # | Persona | Expertise | Focus |
|---|---|---|---|
| EIC | Senior INFORMS data science researcher | Venue fit, contribution, overall quality | Overall decision |
| R1 | ML evaluation methodology specialist | Experimental design, statistical validity | Methodology |
| R2 | Marketing analytics / CRM practitioner | Domain relevance, practical contribution | Domain |
| R3 | Generative AI / tabular data researcher | Generator comparison, LLM framing | Perspective |
| DA | Devil's Advocate | Core argument challenges | Stress test |

---

## EIC Review

**Recommendation: Accept with minor revisions | Score: 7.5/10**

**Venue fit:** Excellent. "Synthetic data generation" is explicitly listed in the INFORMS WDS CFP. The paper addresses AI in digital marketing with real datasets. Fit is among the strongest possible for this workshop.

**Writing quality:** Noticeably improved from a standard academic scaffold. The abstract is direct and practical. The introduction opens with the business problem immediately. Section 3 onward reads naturally and professionally. The "Spoiler: it can't" aside in §2 is appropriately conversational for a workshop paper. Overall tone matches the INFORMS practitioner-oriented audience well.

**Strengths:**
- Real marketing datasets (Hillstrom, Criteo), not generic UCI benchmarks
- Table 2 (measured synthetic class distributions) is the paper's strongest contribution — directly supports the mechanism claim
- Figure 1 (minority budget vs gain) makes the central thesis visually immediate
- Figure 2 (U-curves with CI bands) supports the α recommendation
- TSTR table correctly placed in §2 — answers the "can synthetic replace real?" question upfront
- Honest 1%–10% caveat stated in abstract, §1, §2, and §6

**Remaining issues:**

**EIC.1 (Blocker) — Blinding note still present.** Line 4: `*[BLINDED FOR REVIEW — remove this line before submission]*`. Must be removed before submission.

**EIC.2 (Major) — Table 0 numbering is unconventional.** A table numbered "0" before the main results is unusual and will confuse readers and formatting systems. Renumber as Table 1, shift subsequent tables to 2, 3, 4. Or rename it "Table — TSTR Results."

**EIC.3 (Major) — Figure 1 placed inside the introduction section.** Figure 1 appears after Table 1 and before §2. This is structurally awkward — figures supporting results should appear in the results section, not the introduction. Move Figure 1 to §3 where the results are discussed, or reference it from §3.

**EIC.4 (Major) — No related work section.** For a complete 10-page paper, a brief related work section (~0.5–1 page) situating the paper against prior work is expected. The introduction cites benchmarks but doesn't discuss the broader imbalanced learning or synthetic data literature in a structured way.

**EIC.5 (Moderate) — Paper is currently ~7.5 pages.** For a 10-page complete paper, 2.5 pages remain. This space should be used, not left empty. Related work, expanded discussion, or an additional figure would fill it appropriately.

---

## R1 — Methodology Reviewer

**Recommendation: Minor revision | Score: 7/10**

**Strengths:**
- TSTR condition now defined AND results shown — a significant improvement
- Two-holdout-strategy design clearly explained with correct justification
- Paired t-test with BH-FDR appropriate
- CTGAN enrichment mechanism now explained (log-frequency conditional vector)
- Best-α optimism bias explicitly noted in §3 — well done
- GReaT α=1.0 now stated; serialization explained with example

**Remaining issues:**

**R1.1 (Major) — GReaT still not in Table 2.** The paper claims "GReaT samples from the joint distribution without any conditioning on the minority class, so the synthetic positive rate it produces tends to mirror the training distribution (about 0.9% on Hillstrom)." This is stated as a fact in the text but Table 2 contains no GReaT row to verify it. Either add a measured GReaT row or qualify this as a hypothesis.

**R1.2 (Moderate) — Table 3, three-seed cell warning still primarily in footnote.** The bold "**Only 3 valid seeds**" in footnote ‡ is better than before but still below the table. The Hillstrom Mistral-7B n=100 result should be excluded from discussion in the body text, or the text should explicitly say "this cell has only 3 valid seeds and should be interpreted cautiously."

**R1.3 (Moderate) — TSTR table is single-seed only.** Table 0 is labeled "(single seed)" — this means it's a point estimate with no uncertainty quantification. This should be acknowledged in the text as a limitation: "TSTR results are single-seed point estimates; multi-seed CIs would be needed to establish TSTR gaps precisely."

**R1.4 (Minor) — "Best-α" note in §3 is correct but placement is mid-section.** The note about best-α optimism appears after Table 2, which is after the main results statement. Ideally it should appear in §2 (Experimental Setup) when the α-sweep is defined, not after results are presented.

---

## R2 — Domain Reviewer

**Recommendation: Minor revision | Score: 8/10**

**Strengths:**
- The writing now reads like it was written by a marketing analytics practitioner, not generated
- "A missed converting customer is lost revenue" — exactly the framing this audience needs
- CTGAN vs SMOTE guidance in §6 is now clear and actionable
- Decision table is the clearest element in the paper
- Privacy/GDPR note in limitations is appropriate for marketing context

**Issues:**

**R2.1 (Major) — SMOTE vs CTGAN comparison still not empirically supported in §3.** The §6 recommendation says "both achieve similar gains." Section 3 says "CTGAN and SMOTE recover between +5.7 and +12.9 AUC points" — but this range covers both methods together without breaking them out. A reader cannot verify from the paper that SMOTE achieves comparable gains to CTGAN. Add a sentence in §3 stating the individual gains for each generator on each dataset.

**R2.2 (Moderate) — Optimal α not directly supported by evidence in paper.** Section 6 recommends "α ∈ {0.1, 0.3}" but Figure 2 (U-curves) is placed in §3 without a caption that explicitly identifies the peak α. Connect the figure to the recommendation explicitly.

**R2.3 (Minor) — Dataset vintage.** Hillstrom (2008) is 18 years old. One sentence in limitations would be appropriate.

---

## R3 — Perspective Reviewer

**Recommendation: Minor revision | Score: 7.5/10**

**Strengths:**
- GReaT serialization is now explained with a concrete example sentence — excellent addition
- The architectural connection (Table 2 → GReaT failure) at the end of §5 is the paper's best theoretical contribution
- The future direction paragraph (context-conditioned LLM synthesis) is well-motivated and clearly written
- GPT-2 vs Mistral-7B comparison adds genuine novelty at this venue

**Issues:**

**R3.1 (Major) — GReaT training instability alternative not acknowledged.** Table 3 shows more generation failures at smaller n (4/5 seeds fail at n=50; 3/5 at n=100). This is consistent with GReaT being unstable at very small training sets, independent of the sampling mechanism. The paper presents the "unconditional sampling dilutes minority class" explanation as the conclusion without acknowledging this alternative. One sentence of hedging is needed.

**R3.2 (Moderate) — Figure 2 caption doesn't mention GaussianCopula explicitly.** The caption says "CTGAN and SMOTE gains peak at α ≈ 0.2–0.3... GaussianCopula stays near baseline throughout." But GaussianCopula is shown in Figure 2 — the caption should match what readers see in the figure.

**R3.3 (Minor) — Future direction paragraph references "schema-level metadata" without definition.** "Marginal statistics, feature correlations, domain semantics, signal sparsity" — these are jargon-dense. A simpler phrasing would improve readability for the IS/analytics audience.

---

## Devil's Advocate Report

**Strongest counter-argument: SMOTE achieves comparable gains to CTGAN, undermining the mechanism claim**

The paper's core mechanism is: *CTGAN's conditional vector enriches the minority class; Table 2 shows it generates 7–89× more positive rows; this explains the performance gap over TabDDPM and GaussianCopula.*

This is well-supported for the CTGAN vs TabDDPM/GaussianCopula comparison. But the paper also states SMOTE delivers gains in the same "+5.7 to +12.9 AUC points" range as CTGAN, and SMOTE generates 100% minority-class rows — even more aggressive enrichment than CTGAN.

**The paper's own evidence supports a simpler explanation:** *any method that adds minority-class training examples helps; the amount of minority enrichment matters, not the specific generation mechanism.* 

Under this simpler explanation:
- SMOTE works: 100% minority rows ✓
- CTGAN works: 6–27% minority rows (still enrichment) ✓
- TabDDPM doesn't work: ~0.3% minority rows (no enrichment) ✓
- GaussianCopula doesn't work: ~0.3% minority rows (no enrichment) ✓

The CTGAN mechanism story is consistent with this simpler account, but doesn't distinguish from it. The paper should acknowledge this alternative framing or add evidence that CTGAN's diversity/quality matters beyond simple minority enrichment.

**DA.2 (Major) — "Architectural" conclusion from two training budgets.** The paper concludes TabDDPM's failure is architectural ("limitation is architectural rather than a matter of insufficient optimization"). This is based on one additional training budget (10k vs 2k). Getting worse with more training is also consistent with overfitting. The hedged language in §4 ("we suspect") is appropriate — but the abstract states "neither LLM backbone resolves this architectural limitation" more confidently than the evidence supports.

**DA.3 (Moderate) — Threshold of ~100 minority examples from n=2 datasets.** The threshold is stated as established fact in the abstract and introduction. It's extrapolated from two data points (16 and 72 minority examples). The paper correctly acknowledges the 1%–10% gap — the threshold itself deserves the same hedging.

**Observations (non-defects):**
- The TSTR table addition directly answers a key reviewer concern
- The best-α optimism bias disclosure is honest and appropriate
- GReaT α=1.0 and serialization fixes are clean improvements

---

## Phase 2 — Editorial Synthesis

### Progress since prior review
- Writing quality substantially improved — natural, professional, practitioner voice
- TSTR results added
- CTGAN mechanism explained with log-frequency conditional sampling
- GReaT α and serialization fixed
- Best-α bias acknowledged
- Figure 1 and Figure 2 added
- All blockers from prior review resolved except: blinding note

### Consensus issues (3+ reviewers)

| Issue | Reviewers | Priority |
|---|---|---|
| Blinding note still present | EIC | Blocker |
| Table 0 numbering unconventional | EIC, R1 | Major |
| GReaT not in Table 2 | R1, R3 (implied) | Major |
| SMOTE individual gains not shown | R2, DA | Major |
| Figure 1 placement (intro, not results) | EIC | Major |

### Single-reviewer issues

| Issue | Reviewer | Priority |
|---|---|---|
| No related work section | EIC | Major |
| Paper under-uses 10-page budget | EIC | Moderate |
| TSTR single-seed caveat | R1 | Moderate |
| GReaT training instability alternative | R3, DA | Major |
| Best-α note placement | R1 | Minor |
| Dataset vintage | R2 | Minor |

---

## Editorial Decision: **Accept with Minor Revisions**

**Score: 7.5/10** — significant improvement from prior version.

The paper is now submission-quality in terms of writing voice and scientific content. The mechanism contribution (Table 2, measured not inferred) is the paper's core strength and is now properly explained and defended.

### Must fix before submission

| # | Fix |
|---|---|
| 1 | Remove blinding note (Line 4) |
| 2 | Renumber Table 0 → Table 1 (shift all subsequent tables +1) |
| 3 | Move Figure 1 to §3 where results are discussed |
| 4 | Add individual SMOTE and CTGAN gains in §3 (readers need to verify the claims) |
| 5 | Note GReaT training instability as an alternative explanation in §5 |

### Strongly recommended

| # | Fix |
|---|---|
| 6 | Add GReaT row to Table 2 (or qualify as hypothesis) |
| 7 | Add brief related work section (~0.5 page) |
| 8 | Use remaining ~2.5 pages of budget (related work + expanded discussion) |
| 9 | Connect Figure 2 explicitly to the α ∈ {0.1, 0.3} recommendation in §6 |

### Bottom line
Two substantive fixes needed (#4 individual SMOTE/CTGAN gains, #5 GReaT instability alternative), plus straightforward cleanup (#1 blinding note, #2 table renumbering, #3 figure placement). Estimated time: **1–2 hours**.
