# Coauthor Meeting Primer — LLMSynth Paper

**Prepared:** 2026-06-03 · **For:** revision planning with coauthors
**Companion docs:** `revision-memo-2026-06.md` (drop-in text + citation fixes), `improvement_plan.md` (original Phase 1–5 plan)

---

## 1. Where the paper stands (30-second version)

The paper is rigorous, internally consistent, and unusually honest (it argues against its own earlier headline on Telco n=50). The empirical story is solid: **class imbalance is the dominant driver of statistical-synthesizer value; GReaT is a narrow small-n niche tool; and the multi-seed protocol understates uncertainty for LLM-based synthesis.** It is close to submittable. The open questions are about **positioning and a few targeted additions**, not about fixing broken work.

**Two things to settle as a group:** (1) what venue we're targeting, and (2) which of the additions below are worth the GPU hours before submission.

---

## 2. The one big decision: venue → contribution framing

How we frame the paper changes which additions matter. Three viable paths:

| Path | What it is | What it needs | Accept odds* |
|---|---|---|---|
| **A. Industry / practitioner track** | Keep the decision framework + practitioner lens; trim the literature review | Citation fixes only; minimal new experiments | High (~95%) |
| **B. Applied ML (KDD/CIKM/RecSys)** | Empirical "when does augmentation help" study | Add TabDDPM (diffusion); modernize related work | Medium (~75% w/ additions) |
| **C. Methods / benchmarks venue (NeurIPS D&B, workshop)** | Lead with the **generator-fit variance** contribution | Variance-decomposition experiment + TabDDPM | Lower but higher-prestige (~40%) |

*Rough estimates carried from the improvement plan; for discussion, not gospel.

**Recommendation to put on the table:** Path B as the base, with the generator-fit variance result promoted to a named contribution (cheap, and it's our most novel point). Decide whether Path C's extra rigor is worth it.

---

## 3. MUST-FIX before any submission (citation errors found)

These are concrete errors a reviewer or desk-check would catch. Details + verified facts in `revision-memo-2026-06.md` §A.

- [ ] **Ref 10 — wrong author.** "Tanha et al." → **Agrawal, Hamdare, Ghosh et al.** (IJCIS 2026). Appears in ~5 places (§2.2, §4.1, §10, refs). Content is fine; only the name is wrong.
- [ ] **Ref 13 — placeholder title.** → *Beyond Real Data: Synthetic Data through the Lens of Regularization* (Shidani et al.; author was correct).
- [ ] **Ref 14 — placeholder title.** → *Finding the Sweet Spot: … Using ADASYN* (Chia Ramírez; author correct). Note method is **ADASYN** specifically.
- [ ] **Ref 9 — date + author check.** Won et al. published **Feb 2026** (not 2025); confirm "Won, D.-H." authorship directly on the MDPI page (couldn't verify it programmatically).

---

## 4. Low-hanging fruit — checklist with effort & resources

Status legend: ✅ done · ⏳ ready to run · 🔒 blocked · 💬 needs group decision

| # | Item | Effort | Compute / data needed | Status |
|---|---|---|---|---|
| 5 | **Verify & fix citations** (§3 above) | ~1 hr | none | ✅ audited; fixes listed |
| 2 | **Modernize related work** — add TabDiff (ICLR'25) as SOTA pointer, frame GReaT as the 2023 anchor, name diffusion gap in Limitations | 2–3 hrs | none | ✅ drop-in prose written |
| 3a | **Reframe generator-fit variance as a named contribution** (cite Bouthillier 2021; distinguish from van Breugel DGE) | 2–3 hrs | none | ✅ drop-in text written |
| 1 | **Add TabDDPM** on Hillstrom + Criteo (the diffusion augmentation champion) | script ✅; ~3–8 GPU-hrs to run | **GPU** + our prepared Hillstrom/Criteo CSVs | ⏳ script ready, harness-validated |
| 4a | **Multi-classifier robustness** (LR/RF/MLP) + **10-seed** for statistical generators | 3–4 hrs + ~2–4 CPU-hrs | CPU only + our dataset CSVs | 🔒 needs dataset files |
| 3b | **Variance-decomposition experiment** (vary GReaT seed separately from split seed) — turns the variance claim from "natural experiment" into "measured" | ~2–3 hrs script | **GPU**, ~30–60 GPU-hrs | 💬 worth it only for Path C |
| 4b | **GReaT 10-seed expansion** at small-n (pre-registered in Phase 3) | ~1–2 hrs script | **GPU**, ~25–40 GPU-hrs | 💬 decide if needed |

---

## 5. Already done (so coauthors are caught up)

- Repo imported with full history; everything reviewed end-to-end.
- Literature scan completed (LLM-tabular SOTA, diffusion SOTA, variance/reproducibility methodology).
- **`revision-memo-2026-06.md`** — citation fixes + drop-in prose for items #2, #3a, #5.
- **`experiments/run_tabddpm_databricks.py`** — TabDDPM runner, mirrors the §6.8 CI harness exactly; logic-validated on CPU and kwargs checked against the synthcity source. Ready to run on a GPU cluster.

---

## 6. Discussion questions for the group

1. **Venue** — A, B, or C? (drives everything below)
2. **Diffusion** — do we run TabDDPM before submission (strongly recommended for B/C), or cite-and-acknowledge only (acceptable for A)?
3. **Variance contribution** — promote it to the abstract/contributions, or keep it as a limitation? (cheap to promote; it's our most novel point)
4. **GPU budget** — how many GPU-hours can we spend? Determines whether 3b/4b happen.
5. **Generator scope** — is GReaT-as-sole-LLM defensible if we frame it as a 2023 representative, or does a reviewer expect REaLTabFormer/TabuLa too?
6. **Author/data** — who owns the Databricks runs and can export the prepared Telco/Hillstrom/Criteo CSVs so the CPU work (#4a) can start?

---

## 7. Proposed sequencing (if the group agrees)

1. **This week, no compute:** apply citation fixes (#5) + related-work modernization (#2) + variance reframe (#3a) — all drop-in from the memo.
2. **Unblock data:** export prepared Telco/Hillstrom/Criteo CSVs → unlocks #4a (CPU) and the TabDDPM run (#1).
3. **GPU pass:** run TabDDPM (#1); if Path C, run the variance-decomposition (#3b) and GReaT 10-seed (#4b).
4. **Integrate:** fold new results into §6.8 / §6.6, update abstract + synthesis, final consistency pass.

---

## 8. The one-line ask for the meeting

> "We have a solid, honest paper. I want us to (a) sign off on the citation fixes, (b) pick a venue, and (c) decide whether to spend ~10–60 GPU-hours adding the diffusion baseline and the variance experiment — both of which directly answer the two things a reviewer is most likely to flag."
