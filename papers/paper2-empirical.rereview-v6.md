# Peer Review Panel — Sixth Re-Review of `paper2-empirical.md`

**Manuscript:** *When Class Imbalance Dominates: A Controlled Empirical Study of Synthetic Data Augmentation for Marketing Classification*
**Re-review date:** 2026-06-14
**Previous reviews:** v1, v2, v3, v4, v5
**Mode:** Re-review v6 — post Mistral-7B addition

---

## EDITORIAL ASSESSMENT — Editor-in-Chief

**Verdict: Accept after prose rewrite. Score down slightly from v5 due to integration risk.**

**Score progression:**

| Pass | Score | Verdict |
|---|---|---|
| v1 | 4/10 | Major revision |
| v2 | 5/10 | Major revision |
| v3 | 6.5–7/10 | Conditional accept |
| v4 | 7.5–8/10 | Accept after rewrite |
| v5 | 8–8.5/10 | Strong accept after rewrite |
| **v6 (Mistral-7B added)** | **7.5–8/10** | **Accept with caveats** |

**Why the score dropped slightly from v5:** the paper was a clean, focused empirical study at v5. Adding the Mistral-7B subsection introduces two new concerns:

1. The §4.7 GReaT section is now ~50% larger and starts to dominate the paper's narrative
2. The scale claim ("model scale does not fix GReaT's failure modes") is stronger than the evidence supports

The Mistral-7B addition is genuinely valuable but creates an integration issue: the paper now contains material that probably should be its own paper (Paper 3 was drafted separately for this reason). A KDD Applied reviewer will read the Mistral-7B section and wonder why this empirical paper includes an AI-research-level investigation.

**Recommendation:** Keep the Mistral-7B addition but compress it. Two paragraphs and one table — not three tables, a figure, and 600 words. The detailed comparison belongs in Paper 3.

---

## REVIEWER 1 — Statistical Methods (v6)

**Status:** No regression on statistical apparatus, but the new §4.7 Mistral-7B content has no statistical tests. This is the same R1.1 issue from Paper 3's review v1.

**R1.8 (New) — §4.7 Mistral-7B subsection has no paired tests.**

Three tables of GPT-2 vs Mistral-7B comparisons (German, Hillstrom, Telco) report means and CIs but no paired t-tests. The §4.8 Statistical Summary covers GReaT-vs-Baseline but not GPT-2-vs-Mistral-7B. For consistency with the rest of the paper's statistical rigor, add paired tests for at least the headline Mistral-7B claims:
- GPT-2 vs Mistral-7B on Telco at n=100 (most consistent positive finding)
- GPT-2 vs Mistral-7B on Hillstrom at n=100 (most relevant for imbalance hypothesis)

Without these, the "Mistral-7B is marginally better" claim is descriptive, not inferential.

**R1.9 (New) — n=50 Mistral-7B Hillstrom row is statistically uninterpretable.**

"0.473 ± —" with footnote "only 1 valid seed (4/5 sampling failures)". A single value with no error bar should not appear in a comparison table. Either:
- Report the 1-seed value as a separate observation with the failure rate
- Move it to the discussion as "Mistral-7B has higher generation failure rate at extreme small n"
- Exclude from table entirely

Currently it creates an apples-to-oranges comparison with the 5-seed GPT-2 result in the same row.

**Other R1 items:** all v5 items remain resolved.

**Recommendation:** Minor revision for statistical consistency.

---

## REVIEWER 2 — Synthetic Data / Tabular ML (v6)

**Status:** The Mistral-7B addition is well-motivated but creates new concerns specific to LLM evaluation methodology.

**R2.9 (New, important) — Scale-architecture confound.**

GPT-2 (117M, 2019) and Mistral-7B (7B, 2024) differ in much more than scale: tokenizer, attention mechanism (grouped query attention, sliding window), training data, possibly instruction tuning. The §4.7 claim "Mistral-7B (60× larger than GPT-2)" frames the comparison as a scale comparison when it is actually a model-family comparison. Reframe to: "Mistral-7B (a substantially more capable modern LLM)" without attributing the differences specifically to scale.

This is not a critical issue for an applied empirical paper — readers understand that "bigger model from a different era" is a meaningful comparison — but the language should be precise.

**R2.10 (New) — Conditional-vs-unconditional architectural claim now appears in two places.**

§4.7 conclusion: "Crucially, Mistral-7B's best Hillstrom gain (+3.55 pts at n=100) remains far below CTGAN (+5.75 pts)."

§5.2 mechanism: TabDDPM unconditional sampling explanation.

These are the same architectural argument applied to two different generators. The paper should make this explicit: the same mechanism (conditional vs unconditional sampling) explains both why CTGAN beats TabDDPM and why CTGAN beats Mistral-7B GReaT. Currently this connection is implicit; making it explicit elevates the architectural insight from a TabDDPM-specific explanation to a general principle.

**R2.11 (New, observation) — Mistral-7B Telco result is informative but underweighted.**

Mistral-7B consistently improves over GPT-2 on Telco across all four n values. This is the cleanest positive finding in the §4.7 subsection. The paper currently buries this in the table and emphasizes the negative findings on German Credit and Hillstrom. A balanced framing: "Mistral-7B helps when LLM priors are operative (Telco: semantic + balanced) but not in the imbalanced regime that matters for the paper's central claim." That's a more nuanced and interesting reading.

**Recommendation:** Minor revision — R2.10 is a high-value content improvement.

---

## REVIEWER 3 — Applied Marketing ML (v6)

**Status:** The Mistral-7B addition does not change the practitioner recommendation but adds compelling evidence.

**R3.7 (New) — The Mistral-7B result strengthens the practitioner recommendation.**

For a marketing practitioner reading the paper, the §4.7 update answers a real question: "Should I try a modern LLM instead of CTGAN?" The answer is now empirically grounded: Mistral-7B was tested, it does not beat CTGAN on imbalanced marketing data, and it costs ~30 min GPU per fit. The recommendation (default to CTGAN) is more defensible with this comparison than without.

**R3.8 (Observation) — Compute cost should be stated.**

§4.7 mentions H100 GPU and bf16 + 8-bit Adam but does not state wall-clock time. For practitioners considering Mistral-7B fine-tuning, "~30 min per (n, seed) fit on H100" is the relevant fact. Add a sentence.

**Recommendation:** Accept after prose rewrite. The Mistral-7B addition is a net positive for the applied-track verdict.

---

## DEVIL'S ADVOCATE (v6)

**Status:** New objections raised by the Mistral-7B addition, but none are critical for an applied venue.

**DA.10 (New) — The §4.7 Mistral-7B section makes the paper feel overpacked.**

The paper at v5 was a clean empirical study with a clear thesis: class imbalance is the driver, CTGAN wins. The §4.7 Mistral-7B addition is interesting but introduces a new sub-thesis (LLM scale doesn't help). The two theses are related — both are evidence for the architectural conditional-vs-unconditional argument — but the paper does not explicitly unify them. A reviewer reading §4.7 might wonder: "Is this paper about class imbalance or about LLM scale?"

A clean unification: the architectural argument (conditional sampling matters more than model class or scale) is what ties TabDDPM, GReaT/GPT-2, and GReaT/Mistral-7B together. All three are unconditional samplers; all three lose to CTGAN under extreme imbalance. State this explicitly in §5.2.

**DA.11 (New) — The Telco positive finding undercuts the headline.**

Mistral-7B helps consistently on Telco (4/4 n values positive). The paper says "scale does not fix failure modes" but on Telco — where there are no failure modes — scale does help. A more accurate framing: "Scale helps when GReaT's preconditions (semantic features, sufficient class balance) are met; it does not fix the failure modes (anonymized features, extreme imbalance) when preconditions are violated." That's more nuanced but more honest.

**Recommendation:** Conditional accept — the integration concerns are real but not blocking for an applied venue.

---

## SYNTHESIS

### Issues introduced by the Mistral-7B addition (v6)

1. **R1.8 — No paired tests on GPT-2 vs Mistral-7B** (~5 minute fix)
2. **R1.9 — n=50 Hillstrom Mistral cell is uninterpretable** (text edit)
3. **R2.9 — Scale-architecture confound language** (text edit, soften "scale" to "modern LLM")
4. **R2.10 — Unify the architectural argument across TabDDPM and Mistral-7B** (one paragraph in §5.2)
5. **R2.11 / DA.11 — Reframe Telco positive finding** (one sentence in §4.7 conclusion)
6. **DA.10 — Integration with paper's central thesis** (explicit connection in §5.2)
7. **R3.8 — Compute cost statement** (one sentence)

All issues are text-only. None require new experiments. Total effort: one hour of focused editing.

### Trade-off

The Mistral-7B addition is net positive (~+1 point on applied-track score for completeness, ~−1 point for integration risk; rounds to zero or slight positive). The cleaner alternative — keeping Paper 2 focused and putting Mistral-7B in Paper 3 — was the v5 plan and remains valid. The current arrangement (both in Paper 2) is defensible but requires the integration work above.

### What would NOT change the verdict

- More datasets for the Mistral-7B comparison (not blocking)
- Same-family scale comparison (not blocking for empirical paper, blocking for Paper 3's AI framing)
- Class-conditional prompting ablation (would strengthen but not blocking)

---

## OVERALL VERDICT

**Score: 7.5–8/10**
**Aggregate recommendation: Accept after prose rewrite + 1 hour of §4.7 integration edits.**

The Mistral-7B addition is genuinely valuable as a defense against the "you didn't test modern LLMs" reviewer objection. It does cost some narrative focus, but the paper survives that with one paragraph of explicit unification (R2.10 / DA.10) and softer scale language (R2.9).

**The prose rewrite remains the gating item.** The Mistral-7B addition does not change that calculus — only the author can replace the AI-generated structural scaffolding with their own voice.

**Time to submission:** ~1 hour of §4.7 edits + 2–3 days of prose rewrite. Still on track for KDD Applied Data Science track submission this month.
