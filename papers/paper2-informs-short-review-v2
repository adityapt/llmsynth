
Phase 0 — Field Analysis

┌────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│     Dimension      │                                               Result                                                │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Title              │ Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Primary discipline │ Applied machine learning / data science                                                             │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Secondary          │ Marketing analytics, applied statistics, business OR (INFORMS venue)                                │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Research paradigm  │ Quantitative — controlled empirical benchmark                                                       │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Methodology type   │ Computational experiment / benchmarking (statistical modeling + ML)                                 │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Target tier        │ Workshop short paper (INFORMS WDS 2026); empirically solid, moderate ambition → Q2-equivalent       │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Maturity           │ Pre-submission — polished prose, figures, statistical reporting, explicit limitations               │
└────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘

Panel configuration:

┌─────┬──────────────────┬───────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│  #  │       Role       │                      Persona                      │                                      Primary lens                                      │
├─────┼──────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ 1   │ EIC              │ Associate editor, INFORMS Journal on Data Science │ Fit, originality, significance for OR/DS readership                                    │
├─────┼──────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ 2   │ R1 — Methodology │ ML/stats benchmarking expert                      │ Experimental design, holdout asymmetry, paired tests/FDR/CIs, n=6 inference            │
├─────┼──────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ 3   │ R2 — Domain      │ Tabular synthetic-data generation researcher      │ Generator mechanisms, literature coverage, contribution vs. existing benchmarks        │
├─────┼──────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ 4   │ R3 — Perspective │ Marketing-analytics practitioner                  │ Deployment value, privacy, actionability of the recommendation table                   │
├─────┼──────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ 5   │ Devil's Advocate │ Core-argument challenger                          │ "Scarcity drives value" claim, the ~100 threshold from 6 datasets, mechanism causality │
└─────┴──────────────────┴───────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

Dispatching the panel now.

⏺ 5 agents finished
EIC review · 5 tool uses · 10.3k tokens
⎿  Done
Methodology review · 6 tool uses · 44.1k tokens
⎿  Done
Domain review · 9 tool uses · 42.6k tokens
⎿  Done
Perspective review · 4 tool uses · 10.3k tokens
⎿  Done
Devils advocate review · 2 tool uses · 10.5k tokens
⎿  Done

All five reviewers have reported. The Devil's Advocate flagged CRITICAL issues, which u removes "Accept" from the table. Here is Phase 2 — Editorial Synthesis.

---
📋 Editorial Decision Package

Synthetic Data Augmentation in the Extreme-Imbalance Regime: Evidence from Marketing Classification

Venue: INFORMS Workshop on Data Science 2026 (short paper) · Round: 1 · Date: 2026-06-19

Decision: MAJOR REVISION

Four of five reviewers independently converged on Major Revision; the EIC alone proposed Minor Revision on workshop-leniency grounds. The Devil's Advocate identified three CRITICAL issues, so per panel
rules the decision cannot be Accept/Minor regardless of venue tier. The empirical core sually honest — but the paper's headline causal claim currently outruns its evidence, and that is fixable.

Panel recommendation spread

┌──────────────────┬─────────────────────┬───────────────────────┬────────────────┐
│     Reviewer     │        Lens         │    Recommendation     │ Weighted score │
├──────────────────┼─────────────────────┼───────────────────────┼────────────────┤
│ EIC              │ Fit / significance  │ Minor Revision        │ 70.6           │
├──────────────────┼─────────────────────┼───────────────────────┼────────────────┤
│ R1               │ Methodology         │ Major Revision        │ 66.3           │
├──────────────────┼─────────────────────┼───────────────────────┼────────────────┤
│ R2               │ Domain (generators) │ Major Revision        │ 68.1           │
├──────────────────┼─────────────────────┼───────────────────────┼────────────────┤
│ R3               │ Practitioner        │ Major Revision        │ 71.4           │
├──────────────────┼─────────────────────┼───────────────────────┼────────────────┤
│ Devil's Advocate │ Core argument       │ CRITICAL issues found │ —              │
└──────────────────┴─────────────────────┴───────────────────────┴────────────────┘

---
🔴 Consensus issues (raised independently by ≥3 reviewers)

1. The "~100 minority examples" threshold is under-supported and internally inconsistent — flagged by ALL FIVE.
The entire "augmentation helps" result rests on two datasets (Criteo, 16 minority; Hillstrom, 72). There is no data point between 72 and 240 minority examples, nor between 0.9% and 11.7% positive rate. The threshold is stated as "~100" (abstract/intro, lines 12/26) but "~200" in Figure 1 (line 44) and "~240" in §3 — a headline number that disagrees with itself by 2×, located in an empirical void. The DA escalates this to CRITICAL: minority count is perfectly confounded with positive rate across the two dataset clusters, so the data cannot separate "scarcity" from "imbalance" — making the word "specifically" (line 26) unsupported by design.

2. Dataset count mismatch: "seven datasets" vs. six rows in Table 1 — flagged by EIC, R1, R2, R3.
Abstract and §1 (lines 12, 24, 26) say seven; Table 1 lists six. R1 independently verified the results/ directory contains additional datasets (credit_default, nomao_sparse, online_retail_clv) not in Table 1. This is the foundational evidence roster of a benchmark paper and must be reconciled.

3. Best-α-over-the-sweep reporting inflates every positive number — flagged by R1 and DA.
All gains are the maximum over α ∈ {0.1…1.0} (line 102). Disclosed honestly, but disclosure doesn't remove the upward bias; the borderline CTGAN>TabDDPM result (d_z=1.25, p=0.049, line 110) is one selection away from non-significance, and BH-FDR corrects across method pairs, not across the α grid.

*4. AUC-only evaluation may be the wrong objective — flagged by R3 (CRITICAL) and DA (frame-lock).*
The motivation is revenue (missed conversions/churn, line 22) but the evidence is pure AUC-ROC. Under extreme imbalance, marketing decisions run on precision@k / lift / expected profit and PR-AUC; augmentation also shifts the base rate and distorts calibration. The §6 table could give confidently wrong "skip"/"augment" advice under a profit objective never measured.

---
⚖️ Reviewer disagreements & arbitration

Decision severity (EIC Minor vs. three Major). The EIC scored on workshop-short-paper leniency and viewed the blockers as non-methodological (count fix, tone, threshold wording). R1/R2/R3 each independently found that a load-bearing claim — not just polish — is unsupported. Arbitration: the methodology, domain, and practitioner objections all attach to the central thesis, and the DA's confound argument (C1) is decisive. Major Revision stands.

R2's distinctive technical challenge (not raised by others, flagged for author attention). R2 asserts the paper's mechanism framing is mechanistically wrong: TabDDPM is class-conditional by architecture (it samples at the empirical class prior — it doesn't rebalance), and GReaT natively supports label-conditioned sampling via feature-order permutation but was run unconditionally. If correct, the "failure is architectural / structural" language (§4, §5) should become "we used default/unconditional sampling that does not rebalance the minority class," and §5 currently contradicts §6 (which proposes conditioned LLM synthesis as the fix). This is a single-expert claim — the authors should verify against Kotelnikov et al. (2023) and Borisov et al. (2023) — but it is well-reasoned and, if upheld, materially strengthens the paper by converting a confound into a runnable experiment.

---
⚠️ Devil's Advocate CRITICAL findings (cannot be ignored)

┌─────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┐
│ ID  │                                                        Issue                                                         │                         Why it gates the decision                         │
├─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C1  │ Minority count ⟂ positive rate are perfectly confounded across the two dataset clusters                              │ "Scarcity specifically drives value" is observationally indistinguishable │
│     │                                                                                                                      │  from "imbalance drives value" under this design                          │
├─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C2  │ The ~100 threshold is interpolated across a region with zero observations                                            │ Headline finding asserted where no data exist                             │
├─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C3  │ Untested rival hypothesis: gains track broken baselines (Criteo 0.846±0.228, 7/10 non-converging seeds; Hillstrom    │ A more parsimonious explanation fits the same data and is not ruled out   │
│     │ 0.548 ≈ chance), i.e., augmentation may rescue degenerate optimization, not enrich scarcity                          │                                                                           │
└─────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────┘

The DA's supporting logic for C3 is sharp: if enrichment magnitude drove gains, SMOTE's 100% enrichment should beat CTGAN's 6–27% — yet they tie (line 100), which the paper itself concedes by switching to a "mechanism" axis.

---
🗺️ Revision Roadmap (prioritized — ready for academic-paper revision mode)

P0 — Blocking (must fix before acceptance):
1. Break or bound the confound (C1/C2/W1). Add 2–4 datasets in the 80–240 minority-example / 1–10% rate band — cheaply done by subsampling positives from existing datasets (Bank/Telco) to engineer 3/5/8% rates at fixed schema. This is the single highest-value action: it directly tests the scarcity-vs-imbalance prediction and fills the empty Figure 1 region. If infeasible for this submission, downgrade all threshold language to an interval ("between ~70 and ~240; not localized") and show Figure 1 with a shaded uncertainty band over the uninterpolated region.
2. Test the broken-baseline rival (C3). Add a stabilization-only control (minority duplication / noise injection) alongside the existing class_weight='balanced' baseline to isolate "enrichment" from "optimization rescue." Report Criteo convergence-rate separately from AUC gain.
3. Reconcile the dataset count (W2-EIC/R1). Fix seven-vs-six; ensure Table 1 matches the roster behind every claim.

P1 — Major (re-review will check):
4. De-bias α reporting (R1-W3/DA-M3). Report gain at a pre-specified fixed α (e.g., 0.3) as the primary inferential result; keep best-α as a labeled oracle upper bound. Re-examine the p=0.049 cell under selection-aware correction.
5. Fix the holdout-comparison apples-to-oranges (R1-W2). Recompute the CTGAN reference in Table 3 on the same fixed random_state=42 holdout used for GReaT, or remove the column and label the comparison qualitative.
6. Correct or reframe the mechanism claims (R2-W1/W2). Replace "unconditional / lacks conditioning / structural" with "default empirical-rate (non-rebalanced) sampling"; reconcile §5 with §6. Ideally add the label-conditioned TabDDPM and GReaT runs.
7. Add a decision-aligned metric panel (R3-W1). PR-AUC + precision@k/lift at a realistic targeting depth for the two marketing datasets, plus a calibration note (base-rate shift → recalibrate before scoring).
8. Account for dropped/failed seeds (R1-W4). Report attempted/succeeded/failed counts per cell; treat any cell with <5 valid seeds as descriptive-only (no CI/p).

P2 — Minor (polish):
9. Reproducibility appendix: library versions, seed lists, preprocessing/subsampling recipe, holdout construction (R1-W5); reconcile Table 0 vs Table 1 baseline differences.
10. Privacy: convert the §6 bullet into a measured column (DCR or a simple membership-inference attack) and reconcile "start with SMOTE" with its highest-risk label (R3-W3).
11. Add the missing conditional-generation-for-imbalance literature: CTAB-GAN(+), cWGAN oversampling line, class-conditional diffusion (R2-W3).
12. Lift register ("Spoiler: it can't"), renumber Table 0→1, sort references, remove the blinding marker (EIC).

---
✅ What the panel agreed is strong (preserve in revision)

- Measured mechanism (Table 2) — synthetic positive rates observed, not asserted; praised by all five.
- Scarcity-not-ratio reframing — decision-relevant, changes practitioner behavior (EIC, R2, R3).
- Scope honesty — the 1–10% gap is flagged everywhere; best-α reporting disclosed; unreliable cells marked. Rare and credibility-building.
- The anti-hype LLM-scale null (GPT-2 vs Mistral-7B) and the TabDDPM "more budget made it worse" result — timely, counter-narrative, well-suited to a workshop.

---
The single most leverage-positive move is P0-1: a subsampling experiment that places datasets in the empty 72–240 / 1–10% region. It simultaneously dissolves the C1 confound, fills the C2 void, populates the blank §6 row R3 flagged, and tests the scarcity prediction directly — converting the panel's central objection into the paper's strongest result.

Want me to (a) write these into a revision memo file, (b) hand the roadmap to the academic-paper revision skill, or (c) save the five full reviewer reports to disk?
