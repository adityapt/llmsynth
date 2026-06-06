# Co-Author Task Split — LLMSynth Paper Revision

**Date:** 2026-06-05
**Companion docs:** [`revision-memo-2026-06.md`](revision-memo-2026-06.md) (drop-in text + audited citations) · [`coauthor-meeting-primer.md`](coauthor-meeting-primer.md) (venue framing + experiment status) · [`improvement_plan.md`](improvement_plan.md) (Phase 1–5 plan)

Two roughly equal-effort packages (~3 hours each). All work is on the paper text and citations — **zero new compute required**. Work in parallel; coordination notes at the bottom.

---

## Paper status snapshot (read this first)

The paper is in solid shape after the citation audit and the §6.6 rewrite incorporating Phase 5 (α-sweep on three datasets). What's left before submission:

1. **Apply the drop-in prose** already drafted in `revision-memo-2026-06.md` — citation corrections (memo §A), related-work modernization (memo §B), variance-finding reframe (memo §C). The prose is finished; it just needs to be pasted into the right places in `synthetic-data-marketing-eval.md`.
2. **Full citation audit** beyond the 4 references the memo already flagged. The "Tanha et al." → "Agrawal et al." error in the memo proves at least one citation was fabricated; we need to verify the rest before submission.
3. **Compute work in progress** (background, not on your critical path):
   - Aditya running TabDDPM on Databricks GPU (item #1 from the memo)
   - Aditya running 10-seed × multi-classifier robustness on local CPU (item #4a, `experiments/run_ci_multi_classifier.py`)
   - Both produce results that will be incorporated into §6.8 by Aditya once they finish.

You don't need to wait for compute results to do the work below.

---

## Co-author A — Citations + Related-Work Modernization (~3 hrs)

### A1. Apply the 4 citation fixes from `revision-memo-2026-06.md` §A  (~45 min)

These are concrete errors the audit found. The fixes are in the memo verbatim; you just need to find each occurrence and replace.

- [ ] **Ref 10 — author name wrong.** Replace `Tanha et al.` → `Agrawal et al.` in **5 places**:
   - §2.2 hybrid SMOTE+GAN paragraph
   - §4.1 Evidence paragraph
   - Fig-related mention(s)
   - §10 Recommendation #4
   - Reference 10 itself
   - Use Find&Replace but **verify each one in context** — don't blind-replace; the surrounding sentence may need slight rephrasing.
   - Correct cite: Agrawal, R., Hamdare, S., Ghosh, D., et al. (2026). *Improving Predictive Performance in Telecom Churn Modeling with Hybrid SMOTE and GAN-Based Synthetic Data Generation.* *International Journal of Computational Intelligence Systems*, DOI 10.1007/s44196-026-01204-3.

- [ ] **Ref 13 — placeholder title.** Currently `[Optimal Synthetic-to-Real Data Ratio: A Learning-Theoretic Framework.]`. Replace with real title and drop the `[title inferred]` caveat in both the reference and the provenance doc.
   - Correct cite: Shidani, A., Farghly, T., Sun, Y., Ganjgahi, H., Deligiannidis, G. (2025). *Beyond Real Data: Synthetic Data through the Lens of Regularization.* arXiv:2510.08095.

- [ ] **Ref 14 — placeholder title.** Currently `[Optimal Synthetic Oversampling Ratio for Imbalanced Credit Scoring Data.]`. Replace and drop the `[title inferred]` caveat.
   - Correct cite: Chia Ramírez, L. (2025). *Finding the Sweet Spot: Optimal Data Augmentation Ratio for Imbalanced Credit Scoring Using ADASYN.* arXiv:2510.18252.
   - **One precision note:** the headline method in that paper is **ADASYN specifically**, not generic "synthetic oversampling." §3.4 item 2 already correctly says "ADASYN with 1× multiplication" — no body-text change needed, just the reference title.

- [ ] **Ref 9 — date + author check.** Currently cited as Won, D.-H. et al. (2026). The memo couldn't verify the "Won" authorship programmatically.
   - **Action:** open the MDPI page (Electronics 15(4), 883) and **confirm author list directly**. If it's not Won, fix it. Also confirm the publication date is 2026 (Feb 2026 per the audit) and make the year consistent across all mentions in the paper (some places imply 2025 — find and fix).

### A2. Full citation audit (~1 hour)

The memo only verified the 4 references already flagged as uncertain. The Tanha→Agrawal error means at least one "verified" citation was wrong, so we can't trust the rest by default.

For **every reference in the bibliography**, verify by clicking the DOI/arXiv link:

- [ ] Author names match the paper exactly (especially first author)
- [ ] Year matches the official publication year
- [ ] Journal/venue name is correct (no typos, correct abbreviation)
- [ ] DOI/arXiv ID resolves to the paper actually being cited
- [ ] Title matches (look for typos and substitutions)

For each reference where you find an issue, fix it in the bibliography **and** in every body-text mention. Keep a short log of what you changed so we have audit trail.

The bibliography is at the end of `synthetic-data-marketing-eval.md`. Roughly 15–20 references total (most already audited; this is the remaining ~10 plus a re-pass).

### A3. Apply related-work modernization (memo §B.2)  (~1 hour)

The memo's §B.2 has finished drop-in prose for two locations:

- [ ] **§2.2 LLM-based generators paragraph.** Replace the existing single-paragraph GReaT-centric framing with the memo's §B.2 first block (starts: *"LLM-based serialization methods have diversified rapidly since GReaT..."*). This frames GReaT as the 2023 anchor and names REaLTabFormer, TabuLa, TabMT, LLM-TabFlow as 2023–2025 successors.

- [ ] **§2.2 diffusion-paragraph update.** Apply the memo's §B.2 second block (starts: *"Diffusion models are now the strongest single-table generators..."*). This updates the SOTA pointer from TabSyn (2024) to TabDiff (2025).

- [ ] **§9 Limitations section** — apply the memo's §B.2 third block (starts: *"No diffusion model is evaluated in §6."*). This is the most reviewer-attack-defusing change in the whole revision: it pre-emptively names the diffusion gap and explains why the imbalance-driven conclusion survives it.

### A4. Add new references to bibliography  (~15 min)

The memo §B.1 lists ~5–8 new references to add. Add them as new numbered entries at the end of the bibliography:

- [ ] TabDiff — Shi, J., Xu, M., Hua, W., Zhang, H., Ermon, S., Leskovec, J. (2025). *TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation.* ICLR 2025. arXiv:2410.20626.
- [ ] REaLTabFormer — Solatorio & Dupriez (2023). arXiv:2302.02041.
- [ ] TabuLa — Zhao et al. (2023). arXiv:2310.12746.
- [ ] TabMT — Gulati & Roysdon (2023). NeurIPS 2023. arXiv:2312.06089.
- [ ] Stoian, Giunchiglia, Lukasiewicz (2025). *A Survey on Deep Learning Approaches for Tabular Data Generation: Utility, Alignment, Fidelity, Privacy, Diversity, and Beyond.* arXiv:2503.05954.

Optional (only if you have room in the related-work section):
- [ ] LLM-TabFlow (arXiv:2503.02161) — frontier LLM+diffusion hybrid; cite only if keeping a forward-looking paragraph.

---

## Co-author B — Variance Reframe + Narrative Pass (~3 hrs)

### B1. Apply variance-finding reframe (memo §C)  (~1.5 hrs)

This is the most strategically important edit in the revision. The GReaT-fit variance result is currently buried in §6.6 Methodological Limitations as a hedge on our own claims. The memo argues it should be a **named contribution** because it documents a previously-uncatalogued evaluation-protocol weakness in synthetic-data benchmarking.

- [ ] **C.1 contribution bullet** — add the bullet from memo §C.1 to the Introduction (Contributions list) section of the paper. Prose is finished; paste verbatim with light fitting to surrounding format. The bullet ends with the explicit positioning vs. Bouthillier et al. (2021) and van Breugel et al. (2023, ICML).

- [ ] **C.2 abstract sentence** — find the current abstract clause about GReaT-fit variance (it's the long hedged sentence starting *"GReaT-fit variance"* or similar; if it doesn't exist verbatim, find the closest equivalent). Replace with the memo's §C.2 sentence (sharper, more positioning-conscious, and shorter).

- [ ] **Add two new references to the bibliography** to support the reframe:
   - Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021. arXiv:2103.03098.
   - van Breugel, B., Qian, Z., van der Schaar, M. (2023). *Synthetic Data, Real Errors: How (Not) to Publish and Use Synthetic Data.* ICML 2023. arXiv:2305.09235.

- [ ] **Decision: §6.3 structural move (memo §C.3) — yes or no?**
   - **Yes (more aggressive, Path C signal):** pull the Experiment-5 "GReaT-fit variance" natural-experiment table and Methodological-Limitations item 6 out of §6.6 and promote them into a short standalone subsection at the §6.9 position: **"Generator-Fit Variance and the Limits of the k-Seed Protocol."** Cross-reference from the abstract and contributions. This signals "we have a methodological contribution," appropriate for NeurIPS Datasets & Benchmarks or workshop.
   - **No (less invasive, Path B-friendly):** keep the variance content in §6.6 where it currently lives, but the contribution bullet + abstract sentence still go through. Appropriate for KDD applied / CIKM applied / PeerJ CS / similar.
   - Pick one based on the venue we agree on at the meeting. If undecided, default to **No** for now — easier to add §6.9 later than to remove it.

### B2. Author-read §6.6 end-to-end for narrative continuity  (~45 min)

The §6.6 section has been heavily patched over multiple revisions. Read it cold, beginning to end, as if you've never seen it before. Specifically check for:

- [ ] Does the 5-experiment structure flow naturally, or do transitions feel jarring after Experiment 3 → GC/CTGAN German subsection → Experiment 4 → Methodological Limitations → Experiment 5?
- [ ] Does the Telco n=50 framing read coherently? It went from "paired-significant headline +3.9 pts" in Phase 1 to "not paired-significant in Phase 5 re-run; direction-stable but magnitude noisy" after the cascade. Make sure both runs' results are present and the recalibration reads as honest science rather than damage control.
- [ ] Are there leftover references to the old "Telco n=50 +3.9 pts paired-significant" framing that should now say "+3.9 pts in Phase 1, +2.57 pts in Phase 5 replication"?
- [ ] Does §6.6 Methodological Limitations item 6 read consistently with the §6.6 Experiment 5 GReaT-fit variance table? Both should tell the same story.

Note any rough spots in a short list at the bottom of the section (or in a separate doc), don't try to fix them yet — Aditya will integrate fixes with the TabDDPM and multi-classifier results.

### B3. Sanity-check claim/number consistency across the paper  (~45 min)

Several numerical claims in the paper depend on the empirical results we have. Walk through:

- [ ] **Abstract finding (4)** — the cumulative GReaT story. Does it match what §6.6 actually says after the cascade? Specific claims to check: "α=1.0 suboptimal in 8 of 9 (dataset, n) cells", "best α=0.1 (3 cells), 0.2 (3), 0.3 (2), 1.0 (1)", "up to 4.6 pp drift", "up to ±12 pp per-seed AUC differences."
- [ ] **§6.5 GReaT GPU summary table** (around the synthesis discussion) — Hillstrom row should mention the sign-flip at n=50 in Phase 5; Telco row should reflect the Phase 5 magnitude weakening. Both should match the corresponding §6.6 detail tables.
- [ ] **§7 Decision Framework / §10 Recommendations / §11 Conclusion** — the GReaT recommendation should be calibrated to the recalibrated empirical story (small-n niche tool with non-trivial fit variance, not a paired-significant Telco-n=50 headline).
- [ ] **§5 Method-Level Evidence GReaT row** — same calibrated framing.

If any of these places still say the old strong-headline version, flag in the same short list as B2.

### B4. Read the co-author meeting primer + improvement plan, weigh in on venue  (~30 min)

- [ ] Read [`coauthor-meeting-primer.md`](coauthor-meeting-primer.md) §2 ("the one big decision: venue → contribution framing"). The three paths are:
   - **A — Industry/practitioner:** keep decision framework, trim lit review. Highest accept odds (~95%). PeerJ CS / Decision Support Systems / Journal of Marketing Analytics.
   - **B — Applied ML conference:** add TabDDPM, modernize lit review. Medium accept odds (~75%). KDD Applied / CIKM Applied / RecSys Industry.
   - **C — Methods venue:** lead with the generator-fit variance contribution. Lower accept odds but higher prestige (~40%). NeurIPS Datasets & Benchmarks / SyntheticData4ML workshop.
- [ ] Form a preference. The variance reframe decision (B1 above, especially the §6.9 move) depends on this.

---

## Coordination notes

Both packages edit the same file (`papers/synthetic-data-marketing-eval.md`). To avoid merge conflicts:

- **Work on separate branches.** `coauthor-A/citations-and-related-work` and `coauthor-B/variance-and-narrative`. Open PRs to `main`; Aditya merges them sequentially.
- **Don't touch each other's sections.** A's edits are concentrated in: References list, §2.2 LLM/diffusion paragraphs, §9 Limitations new paragraph. B's edits are: Abstract, Introduction/Contributions, optionally §6.9 promotion from §6.6.
- **§6.6 Experiment 5 GReaT-fit variance content** — A doesn't touch it, B might if doing the §6.3 structural move.
- **Bibliography** — both add references but to different ranges (A adds Tier-3 modernization refs, B adds Bouthillier + van Breugel). Coordinate via the branch PRs.

If you hit a question about what the paper "should say" on a substantive empirical point — leave it as a `<!-- TODO: question for Aditya -->` HTML comment and continue. Don't block.

---

## Timeline expectation

If both packages are completed within ~1 week:
- Aditya integrates the in-progress TabDDPM + multi-classifier compute results into §6.8
- Aditya does a final consistency pass across the whole paper
- We hold the co-author meeting to lock venue + sign off on submission
- Submit shortly after

If anyone needs to push past ~1 week, ping the group so we re-plan timing.
