# Peer Review Panel — Fifth Re-Review of `paper2-empirical.md`

**Manuscript:** *When Class Imbalance Dominates: A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification*
**Re-review date:** 2026-06-08
**Previous reviews:** v1 · v2 · v3 · v4
**Mode:** Re-review v5 — post expert-review fixes

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict: Accept after prose rewrite. Strong recommendation.** Since v4, the manuscript has resolved every major attack the expert reviewer identified — most notably the cost-sensitive baseline omission, which was correctly flagged as "probably the single most dangerous omission." The result is now a paper that anticipates and rebuts its strongest expected criticism with empirical evidence rather than caveat language.

**The fifth-pass score is the highest of any review round.**

**Score progression across five reviews:**

| Pass | Score | Verdict |
|---|---|---|
| v1 (first draft) | 4/10 | Major revision |
| v2 (after MLP + figures) | 5/10 | Major revision |
| v3 (after stats + TabDDPM 10k) | 6.5–7/10 | Conditional accept |
| v4 (after 9 polish fixes) | 7.5–8/10 | Accept after prose rewrite |
| **v5 (after expert-review fixes)** | **8–8.5/10** | **Accept after prose rewrite, strong recommend** |

**Changes resolved since v4:**

| Expert-review attack | v4 status | v5 status |
|---|---|---|
| "Necessary and sufficient" overclaim | Open | ✅ Softened in 3 places (abstract, §2.3, §6) |
| Regression statistically fragile (n=6) | Open | ✅ LOO regression added (R² 0.90–0.96, all p<0.05) + Spearman with honest power note |
| Missing cost-sensitive baseline | Acknowledged limitation | ✅ Empirically benchmarked — CTGAN +7.55 pts vs balanced on both datasets |
| Conclusion overstates evidence | Open | ✅ "default to CTGAN" → "CTGAN consistently achieved the largest gains" |
| PR-AUC results missing | Open | ✅ Added in §4.4 with honest framing |

**The cost-sensitive baseline result is the most strategically valuable single addition in this revision.** It transforms the practitioner-facing recommendation from "use CTGAN (we didn't compare to the cheaper alternative)" to "use CTGAN — we benchmarked the cheaper alternative and CTGAN wins by ~7.5 pts." This is precisely the empirical answer reviewers will demand and most papers in this space cannot provide.

**Final blockers:** None requiring new experiments or analysis. The only remaining work is the author's prose rewrite.

---

## REVIEWER 1 — Statistical Methods (Re-Review v5)

**Status: Accept.**

The statistical apparatus is now methodologically complete for the available sample sizes. Three substantive additions since v4:

**LOO regression (§4.8).** The robustness check I implicitly wanted in earlier reviews is now explicit. Leave-one-out at all six datasets gives R² ranging 0.90–0.96 with p ranging 0.004–0.013. Every LOO fit remains significant at p < 0.05. This is a much stronger response to the "n=6, two influential points" challenge than any rhetorical hedging would be — the data themselves show no point is influential.

**Spearman correlation.** Honestly reported as ρ = −0.49, p = 0.33 with the correct explanation: rank-based tests at n=6 have insufficient power, the sign is consistent, and the LOO regression establishes robustness through a more powerful test. This is exactly the right way to report a non-significant secondary result — neither hide it nor overweight it.

**Cost-sensitive baseline statistics.** The new §4.4 comparison shows CTGAN advantage of +7.55 pts over `class_weight='balanced'` on both datasets. The per-seed values are saved in `ci_balanced_*.csv` for future paired-test extensions. Recommendation: add paired t-test of CTGAN vs Balanced to §4.8 in the prose rewrite — the per-seed data exists, the test takes 30 seconds to compute.

**Single remaining R1 wishlist item:** the §4.8 statistical summary could explicitly list the CTGAN-vs-Balanced comparison as a 13th row in the test family. The data exists; this is a copy-paste extension.

---

## REVIEWER 2 — Synthetic Data / Tabular ML (Re-Review v5)

**Status: Accept.**

The empirical contribution is now well-anchored against multiple alternative explanations:

1. **Vs TabDDPM at default settings**: CTGAN wins (§4.5).
2. **Vs TabDDPM at extended training (N_iter=10k)**: CTGAN wins more decisively (§4.5).
3. **Vs cost-sensitive baseline (`class_weight='balanced'`)**: CTGAN wins by +7.55 pts on both datasets (§4.4 new).
4. **Across classifier families**: CTGAN's advantage holds on GBC, RF, and especially MLP-rescue (§4.6).

This is the kind of multi-pronged empirical anchoring that distinguishes a strong applied paper from a single-comparison benchmark.

**Comparative observation (R2.8, new):** The balanced-GBC variance pattern on Criteo is itself an interesting finding worth a sentence. Seed 2024 (the seed that produced the collapsed baseline AUC=0.5670) gains +35.84 pts under balanced GBC; the other seeds show much smaller changes (or negative deltas on seed 123). This suggests that `class_weight='balanced'` only "fixes" the catastrophic-split case, not the typical-split case. CTGAN, by contrast, improves across all seeds. This per-seed pattern strengthens the variance-stabilisation argument from §4.4 and could be highlighted in the prose.

**No remaining R2 critical issues.**

---

## REVIEWER 3 — Applied Marketing ML (Re-Review v5)

**Status: Strong accept.**

The practitioner-facing recommendation is now empirically defensible. My principal first-round concern (R3.1) is fully resolved.

The §6 conclusion now reads as the kind of operational guidance practitioners want:
- *"CTGAN consistently achieved the largest gains in this study"* — appropriately hedged, claim-supported
- *"For positive rates above 10%, skip augmentation"* — actionable
- *"We benchmarked `class_weight='balanced'`: it hurts on Hillstrom and underperforms CTGAN by 7.55 pts on both datasets"* — addresses the cheapest-alternative question directly

**The paper now answers the practitioner's central question:**
> *"My positive rate is 0.2%. Should I bother with synthetic data, and if so which generator?"*

Answer (paraphrased from the paper):
- Yes, your regime is exactly where augmentation matters.
- Use CTGAN at α ∈ {0.1, 0.3}.
- Free alternatives (class weighting) deliver less than half the gain.
- Diffusion-based generators are not justified — they cost 10–15× more compute and lose.

No applied-track reviewer will block this paper on the question of practical value. The remaining concerns are all stylistic (prose voice).

---

## DEVIL'S ADVOCATE — Re-Review v5

**Status: No further critical objections.**

My v3 and v4 critiques have been resolved in stages. The cost-sensitive baseline addition in v5 was the missing piece — without it, a sceptical reviewer could always argue "CTGAN wins against the wrong baseline." With it, that line of attack is closed.

**One observation (DA.9):** The paper's strongest empirical finding is now arguably the **convergence rescue of MLP under CTGAN augmentation** (Figure 7), not the CTGAN-vs-TabDDPM comparison. MLP-on-Criteo at 7-of-10-seeds-fail-to-converge is a more dramatic and intuitive result than a +7.55 pt AUC margin. The author should consider whether to lead the abstract with this rather than with the regression — it would be a more attention-grabbing opening. (This is a stylistic preference, not a critical concern.)

**Remaining acknowledged limitations** (none are critical):
- The 1%–10% positive-rate region is unfilled (text-only acknowledgment in §3.1 + §5.4).
- TabSyn not evaluated (future work).
- Privacy/uplift/operational evaluation not done (future work).

These are appropriate scope statements for the paper as written.

---

## SYNTHESIS — Final Pre-Rewrite Status

### What is resolved across all five reviews

All 16 substantive issues from v1–v4 + all 4 expert-review attacks from the latest external review are now resolved. The paper has empirical evidence to anchor every major claim it makes.

### What remains

1. **Prose rewrite (author's responsibility).** This is now the single gating item.
2. **Optional 30-second extension:** add paired CTGAN-vs-Balanced t-test to §4.8 (R1's note above).
3. **Optional stylistic decision:** consider promoting the MLP rescue to the abstract opening (DA's note above).
4. **Future work** (not blocking): TabSyn, uplift/CLV, intermediate-imbalance dataset, privacy evaluation.

---

## OVERALL VERDICT

**Score:** 8–8.5/10.
**Aggregate recommendation:** **Accept after prose rewrite. Strong recommendation for KDD Applied Data Science Track.**

The paper is now methodologically defensible against the expert review attacks. The practitioner contribution is anchored against three different alternative explanations (TabDDPM 2k/10k, cost-sensitive learning, classifier-family variation). The statistical apparatus is rigorous within the constraints of the sample size. The MLP rescue is a visually compelling and intuitive result.

**Comparative trajectory:**

| Issue family | v1 verdict | v5 verdict |
|---|---|---|
| Empirical contribution | "Tautology dressed as discovery" | "Multi-anchored confirmation with TabDDPM + cost-sensitive baselines" |
| Statistical rigor | "No paired tests, no FDR" | "Full paired tests + FDR + LOO regression + Spearman" |
| Practitioner relevance | "Recommend CTGAN without comparing to free alternatives" | "Empirically benchmarked vs class_weight; CTGAN wins by 7.55 pts" |
| TabDDPM comparison | "Library defaults — unfair" | "Extended training widens the gap" |
| Robustness | "Single classifier, single seed in places" | "Multi-classifier, 10-seed extension, LOO regression" |

**The empirical work is done. The remaining work is the author's voice.**

---

## What the panel agrees on

All five reviewers independently arrive at **accept** in this round:

1. The paper has a publishable finding well-supported by multiple anchored comparisons.
2. The statistical apparatus is appropriate for the available data.
3. The practitioner-facing recommendation is empirically defensible.
4. No further experiments are required (within the paper's stated scope).
5. The remaining work — prose rewrite — is outside peer-review scope.

The reviewer panel signs off.
