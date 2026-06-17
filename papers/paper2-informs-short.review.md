# Peer Review Panel — `paper2-informs-short.md`

**Manuscript:** *Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification*
**Review date:** 2026-06-16
**Target venue:** INFORMS 10th Workshop on Data Science 2026 (5-page short paper)
**Reviewers:** EIC + R1 (statistical methods) + R2 (synthetic data / ML) + R3 (analytics / IS) + DA

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict: Accept after minor revisions.**

This is a tight, focused 5-page short paper that fits the INFORMS WDS venue well. The CFP explicitly lists "synthetic data generation," "AI in digital marketing," and "applications and adaptations of generative AI" — this paper hits all three. The two tables (minority budget + synthetic class distribution) do most of the argumentative work and are visually compelling. The mechanism story is concrete, measurable, and useful.

**Strengths:**
- Excellent topic-venue fit
- Tables 1 and 2 are self-explanatory and carry the central argument
- Real marketing datasets (Hillstrom, Criteo) — not just UCI benchmarks
- Practitioner-facing decision table in §6 is exactly what this audience wants
- Honest limitations including the 1%-10% gap

**Score: 7/10** for a short paper at this venue. Workshop reviewers are typically less harsh than journal reviewers, and the contribution is clear.

**Comparison to full paper:** This is a *better* paper than the 7,500-word version for this specific venue. The compression forces focus on the strongest results.

---

## REVIEWER 1 — Statistical Methods (5-page review)

**Status: Accept after minor revisions.**

The statistical reporting is appropriate for a short paper. The R²=0.92 is honestly hedged ("directional, n=6"). The CTGAN advantage at extended training (d_z=1.25, p=0.049) is the one inferential test reported, which is fine for the page budget.

**R1.1 (Minor) — The Criteo MLP rescue framing is the paper's most powerful single result and should appear earlier.**
Currently buried in §3 as a "striking illustration." Move to §1 as the motivating hook: "On Criteo, 7/10 MLP seeds fail to train on real data alone; CTGAN augmentation rescues all 10." That sentence alone earns reader attention.

**R1.2 (Minor) — Table 1's "Minority Examples" column should be sorted to make the pattern more visible.**
Currently the table is sorted by positive rate. Consider sorting by minority example count instead — this makes the threshold (~200 examples) jump out immediately to the reader.

**R1.3 (Minor — content correction) — §4 says "No amount of additional training changes the sampling distribution."**
This was a flagged overclaim in prior reviews of the full paper. Use "Additional training within the tested budgets did not change the sampling distribution" — same meaning, no overclaim.

**Recommendation: Accept with minor revisions.**

---

## REVIEWER 2 — Synthetic Data / Tabular ML

**Status: Accept.**

Table 2 is the strongest single piece of evidence in any version of this paper. Measuring TabDDPM's actual positive rate (0.33% vs 0.30% real) closes the most important reviewer attack — that the architectural claim is inferred. This is now empirically verified.

**R2.1 (Minor) — The CTGAN positive rates need brief explanation.**
CTGAN generates 6.34% (Hillstrom) and 26.76% (Criteo) — readers will wonder why these specific numbers rather than 50/50 or matching real distribution. One sentence on CTGAN's log-frequency conditional sampling would clarify, without needing a full mechanism explanation.

**R2.2 (Minor) — GReaT section could be sharper.**
Currently §5 reads as a series of qualitative claims. Consider a 2-row mini-table:

| Model | Hillstrom best gain | Criteo best gain |
|---|---|---|
| GPT-2 GReaT | +2.25 pts | not tested |
| Mistral-7B GReaT | +3.55 pts | not tested |
| **CTGAN reference** | **+5.75 pts** | **+12.87 pts** |

That communicates "LLMs at any scale lose to CTGAN" in 4 lines.

**R2.3 (Observation) — TabDDPM at N_iter=10k is the strongest comparative result.**
The fact that 5× more training makes TabDDPM WORSE on Hillstrom is genuinely interesting and not previously documented in the literature. Consider giving this a one-line emphasis in §4.

**Recommendation: Accept.**

---

## REVIEWER 3 — Analytics / Information Systems

**Status: Strong accept.**

The practitioner decision table in §6 is exactly what an INFORMS audience wants. Most synthetic-data papers I see at IS/analytics venues don't get to this level of actionable guidance. Combined with real marketing data (Hillstrom + Criteo, not just UCI), this paper has clear practical value.

**R3.1 (Minor) — The cost-sensitive baseline (`class_weight='balanced'`) comparison deserves slightly more prominence.**
Currently in §3 as one sentence. For IS/analytics readers, "we benchmarked against the free alternative and CTGAN wins by 7.55 pts" is a key practitioner finding. Consider a single line in §6's recommendation table or directly under the decision guide.

**R3.2 (Minor) — Reproducibility statement would strengthen the paper.**
One sentence at the end: "Code and result CSVs are available at [URL]" — a single GitHub link earns trust at IS venues.

**R3.3 (Observation) — The paper avoids overselling.**
This is rare and appreciated. The phrase "directional, n=6" in §3 is the right level of hedging for the n=6 regression.

**Recommendation: Strong accept.**

---

## DEVIL'S ADVOCATE

**Status: No critical objections.**

The compressed 5-page format eliminates most of the targets a Devil's Advocate would attack in the full paper. The remaining vulnerabilities:

**DA.1 — "Only two extreme-imbalance datasets" remains the most attackable claim.**
The limitations section acknowledges this explicitly, which is the right move. A reviewer might still note that "evidence from marketing classification" in the title is generalised from n=2 marketing datasets. This is appropriately scoped in §6.

**DA.2 — The §3 regression (R²=0.92, n=6) is correctly hedged.**
"Directional, n=6" is honest and the limitations section reinforces this. Nothing more needed.

**DA.3 — The Mistral-7B comparison is well-positioned.**
Framing as "GReaT at two LLM scales" rather than "modern LLMs vs GPT-2" avoids the scale-architecture confound issue. Good.

**No further critical objections at this length and venue.**

**Recommendation: Accept.**

---

## SYNTHESIS

### Critical issues: None.

### Minor revisions (all text-only, ~1 hour total):

1. **R1.1** — Move MLP rescue result to §1 introduction (motivating hook)
2. **R1.2** — Sort Table 1 by minority example count (not positive rate)
3. **R1.3** — Soften "no amount of training" → "within tested budgets"
4. **R2.1** — One sentence on why CTGAN generates at 6-25% (log-frequency sampling)
5. **R2.2** — Optional mini-table in §5 comparing GPT-2 / Mistral / CTGAN gains
6. **R2.3** — One line emphasising TabDDPM-10k getting WORSE on Hillstrom
7. **R3.1** — Surface cost-sensitive baseline finding in §6 recommendation
8. **R3.2** — Add reproducibility line at the end

### Items NOT to do

- Don't add more experiments
- Don't expand the paper beyond 5 pages
- Don't re-introduce the multi-classifier robustness, full statistical apparatus, or GReaT-fit variance details — those belong in the KDD full version

---

## OVERALL VERDICT

**Score: 7.5/10** for a workshop short paper.
**Aggregate recommendation: Accept after minor revisions.**

This is the cleanest piece of writing in the project. The compression forces the strongest results to the surface. The two tables alone tell the story.

**Timeline to submission (by July 1):**
- Apply minor revisions: 1 hour
- Author prose rewrite: 1-2 days
- Format in Word/Google Docs: 0.5 day
- Final pass + blinding: 0.5 day
- **Total: 3-4 days of focused work**

This is well within reach for July 1 if the author commits to writing this week.

**The reviewer panel signs off.** The short paper is ready for author rewrite and submission.

---

## External Reviewer Feedback (added 2026-06-16)

External reviewer (LLM-assisted) submitted independent assessment with the following score:

| Dimension | Score |
|---|---|
| Relevance | 5/5 |
| Novelty | 4/5 |
| Technical quality | 4/5 |
| Practical impact | 5/5 |
| **Overall** | **Accept** |

**Key points from external review:**

1. **Best version of the paper for this venue.** "The full paper became a good applied AI research paper. This short version became a good workshop paper. And importantly, the short version actually communicates the core idea more clearly than the long version."

2. **Table 2 is the centerpiece.** Considered the single most important change across all revisions — converting the mechanism from speculation ("CTGAN probably helps because...") to measurement ("Here are the measured synthetic positive rates").

3. **Two specific text tightenings** (applied 2026-06-16):
   - §4: "No amount of additional training changes the sampling distribution — it is an architectural property" → softened to "Increasing training budget did not alter the observed sampling distribution in our experiments, suggesting that the limitation is architectural rather than optimization-related."
   - §3: Added "The 1%–10% positive-rate region is not represented in this regression and should not be inferred from it." immediately after the regression description, to preempt reviewer criticism by placing the caveat adjacent to the claim.

4. **Venue fit assessment.** Concluded that this short version is better positioned for INFORMS DS than the full paper for KDD because the CFP topic areas (marketing application, synthetic data, practitioner guidance, AI benchmarking, real-world implications) all appear explicitly.

5. **Rejection probability:** "I would be surprised if this version were rejected unless the workshop becomes unusually competitive."

**Both internal panel and external reviewer converge on: accept after minor revisions, submit by July 1.**
