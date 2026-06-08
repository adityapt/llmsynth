# Peer Review Panel — Third Re-Review of `paper2-empirical.md`

**Manuscript:** *When Class Imbalance Dominates: A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification*
**Re-review date:** 2026-06-08
**Previous reviews:** `paper2-empirical.review.md` (first) · `paper2-empirical.rereview.md` (second)
**Mode:** Re-review v3 — verify resolution of remaining issues after statistical apparatus + TabDDPM extended training

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict:** Substantial progress. The four critical methodological issues from the first review are now resolved or addressed, leaving the paper at the threshold of acceptance at KDD Applied Data Science. The TabDDPM extended-training result is the single strongest revision since the first draft — it converts a defensive position ("we used library defaults, hope this is OK") into an offensive one ("we ran 5× extended training and the gap widens, demonstrating the issue is architectural").

**Status of critical issues from prior reviews:**

| Critical issue | First review | Second review | This re-review |
|---|---|---|---|
| Statistical apparatus | ❌ | ❌ | ✅ §4.8 with 12 paired tests + BH-FDR + cross-dataset regression R²=0.92 (p=0.0023) |
| TabDDPM at library defaults | ❌ | ❌ | ✅ N_iter=10k tested; extended training *widens* the gap |
| §3.3 vs §4.2 labeling contradiction | ⚠️ | ❌ | ✅ Fixed; §4.2 correctly labeled as 5-seed CI |
| MLP Criteo treatment | ❌ | ✅ | ✅ (stable; rescue finding remains prominent) |
| n=2 marketing datasets | ❌ | ❌ | ⚠️ Scoped in contribution language but no new dataset added |
| Tautology framing (DA.1) | ❌ | ❌ | ✅ Contribution #1 credits prior work explicitly |
| Prior-art search (DA.6) | ❌ | ❌ | ✅ One sentence added |
| Reference verification (Refs 14, 15) | ❌ | ⚠️ Flagged | ✅ Both verified against arXiv; Chia author name corrected |

**Score progression:** First review 4/10 → second 5/10 → this re-review **6.5–7/10**.

**The TabDDPM-10k result deserves emphasis.** This is the single most reviewer-defusing revision in the paper. Reviewer 2's first-round concern — that the CTGAN-over-TabDDPM result was unfair because TabDDPM was undertrained — is not just addressed but *inverted*. The Hillstrom paired test reaching p=0.049 (nominally significant) at 10k iterations means the paper now has a substantive negative-result finding about diffusion models in extreme-imbalance regimes that no prior work has documented. This is the kind of empirical contribution that gets cited.

**Recommendation: Conditional accept after the remaining items (prose rewrite + a small set of items below) are addressed.** Lean toward acceptance at KDD Applied Data Science.

---

## REVIEWER 1 — Statistical Methods (Re-Review v3)

**Summary:** All five of my first-round R1 issues are now addressed. The statistical reporting is appropriate for the sample sizes available and honest about the inferential limits. This is the cleanest single revision of the manuscript.

### Status of previous R1 issues:

**R1.1 Paired tests on headline comparisons — ✅ Resolved.**
§4.8 now reports 12 paired comparisons with Cohen's d_z, 95% CI, raw p, and FDR-adjusted p. The honest framing (medium-to-large effect sizes, individually underpowered at 5–10 seeds) is exactly what reviewers want to see.

**R1.2 Multiple-comparison correction — ✅ Resolved.**
BH-FDR at q=0.10 applied across the 12-test family. The single FDR-significant finding (GReaT n=2000, p_fdr=0.007) is reported with appropriate prominence.

**R1.3 Formal hypothesis test — ✅ Resolved.**
The cross-dataset regression of CTGAN gain on log(positive rate) — slope=−0.024, R²=0.92, p=0.0023 — is now the primary statistical claim. This is the formal test of the imbalance hypothesis I asked for, and the result is highly significant. The R² of 0.92 across six independent datasets is a stronger piece of evidence than any individual comparison would be.

**R1.4 §4.2 ± labels — ✅ Resolved.**
§3.3 now correctly states that §4.2 uses 5-seed CIs. The protocol section and the results section are now consistent.

**R1.5 TSTR confidence intervals — ⚠️ Partially.**
§4.1 still reports TSTR as single-seed point estimates with no CIs. §3.3 now correctly explains that only TSTR is single-seed (TSTR experiments do not use per-seed CI because the single-seed protocol matches prior work). The single-seed framing is now defensible but I would still prefer to see CIs added in a future revision.

### New R1 issue from this re-review

**R1.7 (New, minor) — The §4.8 paired tests on CTGAN vs TabDDPM-10k should be paired against the *same* baseline.**
The current §4.8 reports the CTGAN vs TabDDPM-10k comparison as Hillstrom Δ=+7.76 pts (d_z=+1.25, p=0.049). This is computed as CTGAN-from-5-seed-CI vs TabDDPM-10k from a different file. The seeds match (both use {42, 123, 7, 2024, 999}) but the practitioner should verify that the per-seed pairing is correctly preserved across the two CSVs before any final submission. Spot-check the per-seed values from `ci_hillstrom.csv` CTGAN α=1.0 against `ci_tabddpm_hillstrom_10000.csv` TabDDPM α=0.1 and confirm they are paired by seed (not by row index).

**Recommendation:** Accept after minor revisions (R1.5 ideal; R1.7 verification).

---

## REVIEWER 2 — Synthetic Data / Tabular ML (Re-Review v3)

**Summary:** R2.1 (the most critical first-round issue) is decisively resolved. R2.2–R2.5 remain in various states of partial resolution.

### Status of previous R2 issues:

**R2.1 TabDDPM hyperparameters at library defaults — ✅ Resolved, and the result strengthens the paper.**
This was my top critical issue and the headline comparative claim. The N_iter=10k results are exactly what I asked for. The result that extended training *hurts* — particularly the Hillstrom going uniformly negative — is more interesting than the original 2k-vs-CTGAN comparison. This is the strongest revision in the paper.

The compute estimate is now also corrected (6 min at 2k / 29 min at 10k vs CTGAN's 2 min CPU). The ratio is now reported as ~3× (2k) to ~15× (10k), which is more defensible than the previous "≈ 20×".

**R2.2 Two marketing datasets is too thin — ⚠️ Scoped.**
The contribution language now scopes claims more carefully. The cross-dataset regression (six datasets total) addresses this partially — it provides a statistical test that does not depend on individual marketing-dataset power. However, a reviewer at KDD will still note that the 1%–10% positive-rate gap between Bank Marketing and Hillstrom is unfilled. The author has indicated no new experiments will be run. **Recommended action:** add explicit language to §3.1 (Datasets) acknowledging the gap, and to the limitations section, scoping all claims to "the imbalanced regime as represented by Hillstrom-like and Criteo-like tasks." This is a text-only mitigation.

**R2.3 TabSyn omission — ❌ Not addressed.**
The TabDDPM-10k result substantially weakens this objection. If extended TabDDPM training widens rather than closes the gap, the burden of proof shifts to anyone claiming TabSyn would behave differently. The §6 future-work mention of TabSyn is now sufficient.

**R2.4 Mechanism ablation — ❌ Not addressed.**
The §5.2 architectural explanation (CTGAN's conditional vector vs TabDDPM's unconditional sampling) remains speculative. However, the new TabDDPM-10k result indirectly supports it: if the mechanism were training quality, more training would help; the fact that it hurts is consistent with the conditional-vs-unconditional explanation. I withdraw this as a critical issue and instead recommend the speculation be acknowledged more explicitly.

**R2.5 Privacy/fidelity not evaluated — ❌ Not addressed.**
Acknowledge in limitations.

**R2.6 MLP rescue finding deserves abstract mention — ⚠️ Partial.**
The MLP rescue is now in the contributions list (item 3) and prominently discussed in §4.6 and the §5.1 discussion of mechanism. The abstract still does not explicitly mention it. **Recommended action:** add one sentence to the abstract about the MLP rescue (this is the single most striking empirical result in the paper).

### New R2 issue

**R2.7 (New) — The TabDDPM-10k "overfitting" interpretation should be hedged.**
§4.5 currently says "the model overfits the training distribution and loses generalization." This is the most plausible explanation but it is not the only one — at α=0.1 on Hillstrom, the TabDDPM-10k model may also be generating systematically different synthetic distributions. The negative-result framing is correct; the mechanism attribution needs one more sentence acknowledging that we infer overfitting from the pattern, not directly measure it. Suggested edit: "Without measuring synthetic-data fidelity directly, we attribute this pattern to overfitting of the diffusion model to the training distribution."

**Recommendation:** Accept after minor revisions (R2.2 scoping, R2.6 abstract mention, R2.7 hedging).

---

## REVIEWER 3 — Applied Marketing ML (Re-Review v3)

**Summary:** The applied-track verdict is now clearly accept after minor revisions. The paper has crossed the threshold for KDD Applied; the remaining items are polish.

### Status of previous R3 issues:

**R3.1 Cost-sensitive baseline comparison — ❌ Not addressed.**
This remains my only unresolved critical issue. The practitioner-facing recommendation in §6 ("default to CTGAN") cannot fully account for `class_weight='balanced'` without empirical evidence. The author has indicated no new experiments will be run.

**Recommended mitigation (text-only):** add a sentence to §6 noting that "practitioners should compare CTGAN augmentation against cost-sensitive learning (`class_weight='balanced'`, threshold moving) as a cheaper baseline before deploying any synthetic-data augmentation; the present study does not benchmark against these alternatives." This protects against the most common practitioner objection without requiring new compute.

**R3.2 n=2 marketing datasets — ⚠️ Scoped.** Same status as R2.2.

**R3.3 Uplift/CLV scope — ❌ Not addressed.** Stay in future work.

**R3.4 "20× compute cost" specifics — ✅ Resolved.**
The §4.5 update gives actual logged times: "2 minutes per seed on CPU; TabDDPM at approximately 6 minutes (N_iter=2k) and 29 minutes (N_iter=10k) per seed on GPU." This is the level of detail practitioners need.

**R3.5 Operational concerns — ❌ Not addressed.** Acknowledge in limitations.

**R3.6 Figure 1 caption with takeaway — ❌ Not addressed.** Caption is still generic.

### New R3 observation

**R3.8 (Positive) — The new framing of contribution #1 is exactly what practitioners need.**
The credit to prior work (Chawla 2002, Fonseca & Bacao 2023, Won et al. 2026) and the reframing as "first TabDDPM comparator" makes clear what the paper *uniquely* contributes vs what it confirms. This is the right level of intellectual honesty for an applied-track paper.

**Recommendation:** Accept after minor revisions (R3.1 mitigation, R3.6 caption).

---

## DEVIL'S ADVOCATE — Re-Review v3

**The first-round tautology critique is now resolved, and the new TabDDPM-10k result substantially strengthens the case for the paper's primary novel claim.**

### Status of previous DA issues:

**DA.1 Tautology critique — ✅ Resolved.**
The new contribution #1 is properly hedged. The framing is "we extend prior imbalance findings to TabDDPM, confirm with multi-classifier robustness, and quantify with cross-dataset regression." This is fair representation.

**DA.2 TabDDPM unfair comparison — ✅ Resolved.**
N_iter=10k directly addresses this. The result that extended training hurts is more interesting than the original 2k comparison.

**DA.3 GReaT-fit variance scope — ⚠️ Partial.**
Contribution #4 now explicitly scopes the variance finding to GPT-2 117M. The body text in §4.7 still extrapolates somewhat. Suggested edit: "We document this as a GPT-2-based-synthesis-specific concern; whether larger or non-GPT LLM synthesizers exhibit similar fit variance is not tested here."

**DA.4 Decision rule oversimplifies — ⚠️ Partial.**
§6 still says "if positive rate < 5%, run CTGAN at α ∈ {0.1, 0.3}." The dataset-size scope (the 10K row cap) is acknowledged in §5.4 limitations but not in the rule itself. The rule should explicitly carry the scope: "for data-scarce imbalanced regimes (n_real ~ 10K, positive rate < 5%)."

**DA.5 MLP Criteo artifact — ✅ Resolved.** Stable from prior re-review.

**DA.6 Prior-art search for CTGAN-vs-TabDDPM — ✅ Resolved.**
The sentence "To our knowledge, no prior work reports a direct CTGAN vs TabDDPM 5-seed CI comparison on Hillstrom or Criteo" is now in contribution #1. This is the right level of claim.

**DA.7 §3.3 internal contradiction — ✅ Resolved.** Stable.

### New DA challenge

**DA.8 (New) — The TabDDPM-10k "overfitting" interpretation is too neat.**
The paper presents a clean story: "more training hurts → overfitting → architectural problem". But the data shows Criteo TabDDPM-10k at α=0.1 is +4.76 pts, at α=0.5 is +6.26 pts, at α=1.0 is +5.88 pts — the pattern is not monotonic in α. Pure overfitting would predict more degradation at higher synthetic-data volumes (more synthetic = more biased = worse), not the U-curve we see. The result might also reflect mode collapse, sampling variance, or the specific (dataset, seed) interaction. The interpretation needs a sentence acknowledging that the pattern is consistent with overfitting but not uniquely so.

**Recommendation:** Conditional accept. The remaining issues are minor and addressable in text. The TabDDPM-10k result removes my most substantive original objection.

---

## SYNTHESIS — Re-Review v3 Conclusions

### Issues resolved in this round (the third pass)

1. **Statistical apparatus** — §4.8 with full paired tests + FDR + regression. R1.1, R1.2, R1.3 closed.
2. **TabDDPM extended training** — N_iter=10k decisively addresses R2.1 and DA.2. The result that extended training *widens* the gap is the strongest single revision in the manuscript.
3. **§3.3 / §4.2 labeling consistency** — R1.4 closed.
4. **Tautology framing** — DA.1 closed.
5. **Prior-art documentation** — DA.6 closed.
6. **Reference verification** — Refs 14 and 15 verified, Chia author name corrected.

### Issues remaining as text-only fixes (~30 minutes total)

7. **R2.6 — MLP rescue in abstract** (1 sentence)
8. **R3.1 — Cost-sensitive baseline acknowledgment** (1 sentence in §6 or §5.4)
9. **R2.7, DA.8 — Hedge the TabDDPM-10k overfitting interpretation** (1 sentence in §4.5)
10. **DA.3 — GReaT scope to GPT-2 117M** (1 sentence in §4.7)
11. **DA.4 — Add n_real scope to §6 rule** (modify rule sentence)
12. **R3.6 — Figure 1 caption with takeaway** (rewrite caption)
13. **R3.5, R2.5 — Privacy/operational concerns in limitations** (1 paragraph in §5.4)
14. **R2.2/R3.2 — Acknowledge n=2 marketing data gap in §3.1 and §5.4** (2 sentences)
15. **R1.7 — Verify per-seed pairing across CSVs** (spot-check, 5 min)

### Issues that cannot be addressed without new experiments (acknowledged limitations)

16. **R1.5 — TSTR multi-seed CIs** (would require re-running TSTR with 5 seeds)
17. **R3.1 (full) — Cost-sensitive baseline empirical comparison**
18. **R2.3 — TabSyn empirical comparison**
19. **R2.4 — CTGAN-without-conditional-vector ablation**
20. **R3.3 — Uplift/CLV evaluation**

These remain as acknowledged limitations. Items 17–20 are the natural set for the author's future work.

---

## OVERALL VERDICT

**Score:** 6.5–7/10 (after this re-review). Estimated 7.5/10 after the 30-min polish list (items 7–15).

**Aggregate recommendation:** **Conditional accept at KDD Applied Data Science Track after minor revisions.** This is the first re-review to recommend acceptance. The remaining items are all addressable without new compute and could be completed in one sitting.

**Comparative assessment:**

| Review | Score | Verdict |
|---|---|---|
| First (full draft) | 4/10 | Major revision |
| Second (after MLP fix + figures + references) | 5/10 | Major revision |
| **Third (after stats + TabDDPM 10k + tautology fix)** | **6.5–7/10** | **Conditional accept** |

**Most important single revision since first review:** TabDDPM N_iter=10k. This converts the paper's defensive position into an offensive one. The Hillstrom paired result (p=0.049 nominal) on extended training is a result the field has not previously documented — that more diffusion-model training on extremely imbalanced data hurts rather than helps. This is the kind of negative result that gets cited.

**Most important remaining single revision:** The prose rewrite (author's responsibility, outside review scope). The paper currently reads as a structural draft; the technical content is solid, the writing voice needs to be the author's.

**Time-to-submission estimate:**
- Item-by-item text fixes (7–15): one focused work session, ~3 hours
- Prose rewrite: 1–2 weeks of author work
- Total: paper submittable to KDD Applied within 2 weeks if the author commits to the prose rewrite

---

## What the panel agrees on

All five reviewers agree:
1. The paper has a publishable finding (CTGAN > TabDDPM on imbalanced marketing; cross-dataset regression confirms the imbalance hypothesis).
2. The TabDDPM-10k result strengthens the paper materially.
3. The remaining critical issues are now text-only and addressable.
4. The paper is appropriate for KDD Applied Data Science Track; not for NeurIPS Datasets & Benchmarks (it is an evaluation paper, not a benchmark contribution) or NeurIPS/ICML main track (no new method).
5. The prose rewrite is the gating item, not the methodology.

## What the panel disagrees on

1. **R1 vs R2/DA on TSTR multi-seed CIs (item 16).** R1 sees this as ideal; R2/DA accept the single-seed framing. Editor: defer; can be a future-work item.
2. **R2/DA on the overfitting interpretation (item 9).** R2 wants one hedging sentence; DA wants explicit acknowledgment of alternative mechanisms. Editor: take the stronger hedging position to be safe.
3. **R3 on the cost-sensitive baseline (R3.1).** R3 sees this as critical; R1/R2/DA see it as recommended-not-required given the empirical scope. Editor: include the acknowledgment sentence (text-only mitigation) and treat the empirical comparison as future work.
