# Paper Improvement Plan — §6.6 Statistical Rigor

**Status:** Drafted 2026-05-16. Locked before any new experimental runs (see §2 — Pre-registration).

This plan locks in the analysis decisions before more data is collected, so that any subsequent claim of statistical significance is confirmatory rather than exploratory.

---

## 1. Current state of the paper

**Solid:**
- Empirical numbers verified against raw CSVs end-to-end
- §6.6 narrative is internally consistent after the 2026-05-16 cascade
- Methodological Limitations subsection acknowledges the eight statistical caveats
- Provenance, README, summary_table, scripts all aligned
- Empirical story: statistical generators dominate under imbalance; LLM is a small-n niche tool

**Known weak spots:**
1. Independent CIs reported, not paired — wrong test for matched-seed design
2. No multiple-comparison correction across 16 paired GReaT tests
3. Telco n=50 +3.9 pt finding is the linchpin and sits at borderline paired p=0.023 (would not survive Bonferroni; barely survives BH-FDR)
4. Single α (=1.0) for GReaT while statistical generators were swept (§6.3)
5. Single downstream classifier (gradient boosting)
6. Single fixed holdout per dataset
7. No effect size reporting (only raw AUC pts)
8. 5-seed evidence is directionally clear but magnitudinally loose

---

## 2. Pre-registration

This plan is pre-registered as of 2026-05-16. Any new experimental work below will be executed against the analysis plan specified here. Findings are split into:

- **Confirmatory:** the claims this plan is designed to test (locked in §3 below)
- **Exploratory:** anything else that emerges from the data analysis

The paper's §6.6 will mark confirmatory vs exploratory findings explicitly after this plan executes.

**Specifically pre-registered:**
- The decision rule for whether to run additional seeds: if Telco n=50 paired-t test does *not* survive BH-FDR at q=0.10 across the 16-test family on the existing 5-seed data, run Phase 3 (5 more seeds at small-n × 3 datasets). Otherwise Phase 3 is optional.
- Test choice: paired t-test on per-seed differences for matched comparisons.
- Multiple-comparison correction: Benjamini-Hochberg FDR at q=0.10 across the 16-test family.
- Effect size: Cohen's d_z (mean of paired differences / standard deviation of paired differences).
- Equivalence bound for "tied" claims: |Δ| < 0.5 AUC pts (TOST procedure).

---

## 3. Phase 1 — Zero-compute re-analysis (today, ~3 hours)

**Goal:** find out what's actually robust on existing 5-seed data before spending any new GPU hours.

| # | Task | Effort | Tool |
|---|---|---|---|
| 1.1 | Add paired t-CIs alongside independent CIs in all §6.6 tables | 30 min | Python (scipy.stats) |
| 1.2 | BH-FDR correction on family of 16 paired tests | 15 min | statsmodels.stats.multitest |
| 1.3 | Bootstrap holdout indices — **DEFERRED** (requires prediction scores or trained classifiers, neither of which is persisted; will run as part of Phase 3) | — | — |
| 1.4 | Cohen's d_z paired effect sizes per cell | 15 min | numpy |
| 1.5 | Jackknife sensitivity (drop-one-seed) per headline finding | 30 min | numpy |
| 1.6 | Sanity-check §6.2 "4–27% TSTR gap" claim against `metrics_*.csv` | 30 min | pandas |
| 1.7 | TOST equivalence test for "tied" claims at |Δ| < 0.5 pts bound | 30 min | scipy |

**Output:**
- `results/rigorous_analysis.csv` — tidy per-cell stats
- `results/rigorous_analysis.md` — human-readable summary

**Decision point at end of Phase 1:** Does Telco n=50 survive BH-FDR?
- If yes (q < 0.10): Phase 3 is optional gold-plating
- If no: Phase 3 becomes necessary to lock the headline

---

## 4. Phase 2 — Engineering (parallel to Phase 1, ~2 hours)

| # | Task | Effort | Reason |
|---|---|---|---|
| 2.1 | Refactor `run_great_*_databricks.py` to persist trained GReaT models, keyed by `(dataset, n, seed)` | 1 hr | Unlocks future α-sweep / multi-classifier without refitting |
| 2.2 | Add resumable seed-expansion logic; safe re-run on (n, seed) already done | 30 min | Lets Phase 3 run only the additional 5 seeds |
| 2.3 | Pre-register this analysis plan in a git commit | 5 min | Defuses cherry-picking critique |

---

## 5. Phase 3 — Targeted seed expansion (only if Phase 1 decision rule fires)

**Cost:** ~25–40 GPU hours
**Scope:** 5 more seeds at small-n cells (n ∈ {50, 100, 200}) symmetric across all three datasets.

| Dataset | n's | Seeds added | ~GPU cost |
|---|---|---|---|
| Telco | 50, 100, 200 | 5 new (10, 20, 30, 40, 50) | ~10–15 hrs |
| Hillstrom | 50, 100, 200 | 5 new (same seeds) | ~10–15 hrs |
| German | 50, 100, 200 | 5 new (same seeds) | ~5–10 hrs |

**Why not n ≥ 500:** large-n findings are already paired-sig with large effects; doubling seeds is waste.
**Why symmetric across datasets:** defuses "you only added seeds where positive findings live" critique.
**New seed list:** {10, 20, 30, 40, 50} added to existing {42, 123, 7, 2024, 999} → 10 seeds total at small-n.

**Output:** updated `great_*_results.csv` files with 10 seeds × 3 small-n cells each.

---

## 6. Phase 4 — Paper updates (after Phases 1–3)

| # | Update | When |
|---|---|---|
| 4.1 | All §6.6 tables: add `paired_CI`, `paired_p`, `p_fdr_q0.10`, `d_z` columns | After Phase 1 |
| 4.2 | §6.6 Methodological Limitations: update items (1)–(3) to note what's now corrected | After Phase 1 |
| 4.3 | Add §6.6 "Robustness Analysis" subsection with jackknife + TOST equivalence results | After Phase 1 |
| 4.4 | Abstract finding (4): update p-values to FDR-adjusted, cite 10-seed power if Phase 3 ran | After Phase 3 (or Phase 1 if Phase 3 skipped) |
| 4.5 | §6 synthesis paragraph: update Telco n=50 effect size + CI to 10-seed | After Phase 3 |
| 4.6 | §11 Conclusion: re-state GReaT envelope with corrected statistics | After Phase 4.5 |
| 4.7 | Update provenance doc with Phase 1 analysis + Phase 3 run metadata | After Phase 3 |
| 4.8 | Mark all §6.6 claims as confirmatory (this plan) vs exploratory (Phase 1 byproducts) | Final pass |

---

## 7. Phase 5 — α-sweep across all three datasets [COMPLETED 2026-05-18/19]

Extended beyond the original "Telco only" Methods-venue scoping after the German α-sweep result (Δ=+1.69 pts at n=100, α=0.1; raw p=0.005, d_z=2.45) showed α=1.0 is suboptimal in 8 of 9 (dataset, n) cells and contradicted the §6.6 main-table interpretation.

**What was run:** α-sweep across all three datasets at n ∈ {50, 100, 200} × α ∈ {0.1, 0.2, 0.3, 0.5, 1.0} × 5 seeds (matched-design, same GReaT fit per (n, seed), sub-sample synthetic for each α). Three scripts: `experiments/run_great_alpha_sweep_{dataset}_databricks.py`.

**Key findings (incorporated into paper §6.6 Experiment 5):**
- α=1.0 almost never optimal: best α was 0.1 (3 cells), 0.2 (3), 0.3 (2), 1.0 (1: Hillstrom n=100).
- Strongest single positive cell: German Credit n=100, α=0.1 (Δ=+1.69 pts, d_z=2.45, raw p=0.005). BH-FDR over the 45-test family: p=0.104 (just over q=0.10).
- No α-sweep cell survives BH-FDR at q=0.10 over the 45-test family.
- α-sweep at α=1.0 vs Phase 1 results = natural experiment for GReaT-fit variance: drift up to 4.6 pp, Telco n=50 headline lost paired significance (p=0.023→0.134), Hillstrom n=50 flipped sign. Documents that user-level `seed` controls NumPy/sklearn only; PyTorch/CUDA/Transformers RNGs are independent.
- Patch applied to all six GReaT scripts: `seed_everything()` covering Python random, NumPy, PyTorch CPU+CUDA, Transformers, cuDNN-deterministic + benchmark-off + CUBLAS_WORKSPACE_CONFIG. fp16 retained; full bit-exact reproducibility would also require fp32.

**Still-optional Phase 5 sub-tasks (not run):**
- Multiple downstream classifiers (LR, RF, MLP). Would test classifier-robustness; not run.
- TabDDPM/TabSyn baseline comparison (50+ GPU hrs, weeks of engineering). Not run.

---

## 8. Critical path

```
Phase 1 (analysis) ─┐
                    ├── Decision rule fires? ─── Phase 3 (GPU expansion) ──┐
Phase 2 (eng)   ───┘                                                       ├── Phase 4 (paper updates)
                                                                           │
                                                                Phase 5 ──┘ (optional)
```

---

## 9. Timeline estimates

| Path | Total time | GPU cost |
|---|---|---|
| Minimum (Phases 1 + 2 + 4 only) | 1–2 days | 0 |
| Recommended (Phases 1 + 2 + 3 + 4) | ~1 week (including GPU queue) | ~25–40 hrs |
| Methods-venue (everything) | ~3–4 weeks | ~100–150 hrs |

---

## 10. Acceptance probability lift (estimated)

| Venue | Ship as-is | After this plan |
|---|---|---|
| Industry/practitioner | ~95% | ~97% |
| Applied ML (KDD/CIKM/RecSys) | ~50% | ~75% |
| Top-tier methods (NeurIPS/ICML/ICLR) | ~20% | ~40% (only with Phase 5) |
| ML workshop | ~70% | ~90% |
