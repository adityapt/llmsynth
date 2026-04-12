# Provenance — synthetic-data-marketing-eval

**Final draft delivered:** April 2026 (revised April 2026)  
**Canonical artifact:** `papers/synthetic-data-marketing-eval.md`  
**Supporting figures:** `papers/fig2-augmentation-utility.png`, `papers/fig3-method-dimension-matrix.png`, `papers/fig5-privacy-utility.png`  
**Plan / verification log:** `outputs/.plans/synthetic-data-marketing-eval.md`

---

## Revision Notes (April 2026)

This revision incorporated three primary sources missing from the previous draft:

1. **Davila et al. (2025)** — the plan's top-ranked primary source — was absent from the previous draft's references and body text. This revision integrates it as a backbone citation throughout §2.2 (method descriptions), §3.1 (U-shaped curve), §3.3–3.4 (when it helps/hurts), §5 (method ranking table and Figure 3), §4.1 (churn), and §4.4 (segmentation). Specifically: Tables 5, 6, and 8 from Davila et al. are now attributed throughout.
2. **Shumailov et al. (2024)** — model collapse in *Nature* — was listed as a primary source in the plan but appeared nowhere in the previous draft. This revision adds a dedicated subsection (§3.2) that correctly scopes the finding to iterative recursive LLM training and explicitly distinguishes it from single-round tabular augmentation.
3. **Erickson et al. (2025)** (TabArena) and **Fonseca & Bacao (2023)** were added to the reference list and body as the plan required.

---

## Source Accounting

| # | Source | Type | Access method | Verification status |
|---|--------|------|---------------|---------------------|
| 1 | Xu et al. 2019 (CTGAN, NeurIPS) | Primary paper | URL + plan cross-check | **Supported** |
| 2 | Kotelnikov et al. 2023 (TabDDPM, ICML) | Primary paper | URL + plan cross-check | **Supported** |
| 3 | Davila et al. 2025 (DSJ prosumer benchmark) | Primary paper | URL + plan research pass | **Supported** — Tables 5, 6, 8 cited throughout |
| 4 | Shumailov et al. 2024 (model collapse, Nature) | Primary paper | arXiv:2305.17493 + Nature DOI | **Supported** — correctly scoped to iterative training |
| 5 | Erickson et al. 2025 (TabArena, NeurIPS 2025) | Primary paper | URL | **Supported** — contextual reference |
| 6 | Du & Li 2024 (arXiv:2402.06806) | Primary paper | URL | **Supported** |
| 7 | Sidorenko et al. 2025 (arXiv:2504.01908, MOSTLY AI QA) | Primary paper | URL | **Supported** |
| 8 | Lautrup et al. 2024 (SynthEval, arXiv:2404.15821) | Primary paper | URL | **Supported** |
| 9 | Won et al. 2026 (MDPI Electronics 15/4/883) | Primary paper | URL | **Supported** |
| 10 | Tanha et al. 2026 (Springer IJCIS, DOI 10.1007/s44196-026-01204-3) | Primary paper | Springer full text fetched | **Supported — author list partially inferred** |
| 11 | Fonseca & Bacao 2023 (Expert Systems with Applications) | Primary paper | URL | **Supported** |
| 12 | Camino et al. 2020 (ICML Workshop) | Primary paper | URL | **Supported** |
| 13 | Shidani et al. 2025 (arXiv:2510.08095) | Primary paper | arXiv abstract fetched | **Supported — title inferred from abstract** |
| 14 | Chia Ramírez 2025 (arXiv:2510.18252) | Primary paper | arXiv abstract fetched | **Supported — title inferred from abstract** |
| 15 | Chen et al. 2025 (arXiv:2504.14061) | Primary paper | arXiv abstract fetched | **Supported** |
| 16 | Chawla et al. 2002 (SMOTE, JAIR) | Primary paper | Widely cited, standard reference | **Supported** |

---

## Key Claims Verification

| Claim | Evidence | Status |
|-------|----------|--------|
| U-shaped error curve for synthetic fraction | Shidani et al. 2025 (arXiv:2510.08095) — theoretical proof + empirical validation | **Supported** |
| Optimal majority:minority ratio ≈ 6.6:1 | Chia Ramírez 2025 (arXiv:2510.18252) — single dataset; author cautions against overgeneralisation | **Supported with caveat** |
| TabDDPM/TabSyn top on augmentation benchmarks | Davila et al. 2025 (Table 6); Kotelnikov et al. 2023 (ICML) | **Supported** |
| SMOTE top on ML utility (imbalanced), low privacy | Davila et al. 2025 (Tables 5, 8) | **Supported** |
| Hybrid SMOTE+GAN outperforms either alone on churn | Tanha et al. 2026 (Springer IJCIS) | **Supported** |
| TSTR consistently lags TRTR | Du & Li 2024; Sidorenko et al. 2025; Davila et al. 2025 | **Supported** |
| Model collapse applies to recursive/iterative training, not single-round tabular augmentation | Shumailov et al. 2024 (Nature) — scope correctly described | **Supported** |
| LLM-based generators lower than diffusion/GAN on utility benchmarks | Davila et al. 2025 — prosumer hardware benchmark | **Supported** |
| Statistical DP methods higher utility, deep learning faster | Chen et al. 2025 (arXiv:2504.14061) | **Supported** |
| Causal structure corruption by off-the-shelf synthesizers | Methodological argument; no single counter-example paper | **Supported (methodological)** |
| Specific AUC % improvements in figures | Illustrative relative rankings, not exact benchmark numbers | **Explicitly labelled as illustrative** |
| 20–40% optimal synthetic fraction range | Consistent across augmentation regime studies; exact range varies | **Inferred from consensus, not a single study** |

---

## What Was Not Verified

- **Exact author list for Tanha et al. 2026** — "Tanha et al." from the plan's prior research pass. Readers should verify the journal page directly.
- **Exact paper titles for Shidani et al. 2025 and Chia Ramírez 2025** — titles inferred from the arXiv abstract page display; should be confirmed against the official PDF title pages before formal citation.
- **SMOTE near-duplicate / privacy risk** — attributed to Davila et al. 2025, Table 8 (DCR-based privacy scores); specific membership-inference attack paper not cited separately.

---

## Figure Provenance

| Figure | Type | Data basis | Illustrative? |
|--------|------|------------|---------------|
| Fig 1 — Taxonomy (Mermaid) | Inline diagram | Literature survey | N/A |
| Fig 2 — Augmentation utility bar chart | PNG (Vega-Lite) | Relative estimates from Davila et al. 2025, Tanha et al. 2026, Won et al. 2026 | **Yes — illustrative** |
| Fig 3 — Method × Dimension heatmap | PNG (Vega-Lite) | Relative scores from method ranking table in draft, anchored to Davila et al. 2025 | **Yes — illustrative** |
| Fig 4 — Decision flowchart (Mermaid) | Inline diagram | Synthesized decision rules | N/A |
| Fig 5 — Privacy–utility scatter | PNG (Vega-Lite) | Relative scores from method table; privacy scores from Davila et al. 2025 Table 8 | **Yes — illustrative** |
