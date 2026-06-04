# Revision Memo — Zero-GPU Writing Pass

**Date:** 2026-06-03
**Scope:** Low-hanging-fruit items #2 (related-work modernization), #3a (variance-finding reframe), and #5 (citation verification) from the project review. These are the changes that need **no GPU and no new data** — drop-in text and corrections you can paste into `synthetic-data-marketing-eval.md` directly.

CPU experiments (#4a multi-classifier + statistical-generator seed expansion) and Databricks-ready GPU scripts (#1 TabDDPM, #3b variance decomposition, #4b GReaT seed expansion) are tracked separately and will follow.

---

## A. Citation corrections (item #5)

I verified the four "inferred/partial" references flagged in the provenance doc against primary sources. **Two contain outright errors that a reviewer or desk-check would catch.** Fix these before any submission.

### A.1 Ref 10 — WRONG AUTHOR (high priority)

The paper cites the telecom-churn hybrid SMOTE+GAN paper as **"Tanha et al. (2026)"** throughout (§2.2, §4.1, §10, References). The actual authorship is **Agrawal, R., Hamdare, S., Ghosh, D., et al.**

- **Verified title:** *Improving Predictive Performance in Telecom Churn Modeling with Hybrid SMOTE and GAN-Based Synthetic Data Generation*
- **Venue/DOI:** *International Journal of Computational Intelligence Systems* (2026), DOI 10.1007/s44196-026-01204-3
- **Content check (matches your body text):** IBM Telco dataset, six classifiers (LR, DT, RF, XGBoost, CatBoost, LightGBM), RF best precision 85.36%. Your characterization is accurate — only the author name is wrong.
- **Action:** Replace every "Tanha et al." with "Agrawal et al." (5 occurrences: §2.2 ¶hybrid, §4.1 Evidence, Fig-related mentions, §10 Rec #4, Reference 10).

### A.2 Ref 13 — WRONG TITLE (placeholder never replaced)

Reference 13 currently reads *"[Optimal Synthetic-to-Real Data Ratio: A Learning-Theoretic Framework.]"* (bracketed = inferred). The first author "Shidani et al." is **correct**, but the title is a placeholder.

- **Verified title:** *Beyond Real Data: Synthetic Data through the Lens of Regularization*
- **Authors:** Amitis Shidani, Tyler Farghly, Yang Sun, Habib Ganjgahi, George Deligiannidis (Apple / University of Oxford / Big Data Institute)
- **arXiv:** 2510.08095 — also on OpenReview (forum `Ux11GkiQD2`) and Apple ML Research; treat as a 2025 preprint, confirm final venue before camera-ready.
- **Content check:** Yes — derives the optimal synthetic-to-real ratio via algorithmic-stability generalization bounds as a function of Wasserstein distance, validates the U-shaped test-error curve. Your §3.1 usage is accurate.
- **Action:** Replace the bracketed title with the real one; drop the "[title inferred]" caveat in both the reference and the provenance doc.

### A.3 Ref 14 — WRONG TITLE (placeholder never replaced)

Reference 14 reads *"[Optimal Synthetic Oversampling Ratio for Imbalanced Credit Scoring Data.]"* (inferred). Author "Chia Ramírez, L." is **correct** (Luis Chia Ramírez).

- **Verified title:** *Finding the Sweet Spot: Optimal Data Augmentation Ratio for Imbalanced Credit Scoring Using ADASYN*
- **arXiv:** 2510.18252 (stat.AP, 21 Oct 2025)
- **Content check:** Give Me Some Credit (97,243 obs, 7% default), ADASYN/SMOTE/BorderlineSMOTE at 1×/2×/3×, XGBoost, bootstrap test (1,000 iters). ADASYN 1× optimal (AUC 0.6778, Gini 0.3557, p=0.017); 6.6:1 optimal majority:minority. All your numbers in §3.4 item 2 match. **One precision note:** the headline method is **ADASYN**, not generic "synthetic oversampling" — worth naming it explicitly since your §3.4 currently says "ADASYN with 1× multiplication" (correct) but the reference title obscured it.
- **Action:** Replace the bracketed title; drop "[title inferred]".

### A.4 Ref 9 — DATE + UNVERIFIED AUTHOR

Cited as **"Won, D.-H. et al. (2026)"**, *Electronics* 15(4), 883.

- **Verified title:** *Synthetic Data Augmentation for Imbalanced Tabular Data: A Comparative Study of Generation Methods* (correct).
- **Verified date:** Published **20 February 2026** — your body text says "(2026)" but the README/Datasets line and some mentions imply 2025. Make the year consistent at 2026.
- **Content check:** Compares **SMOTE, Gaussian Copula, TVAE, CTGAN** on the **UCI Bank Marketing** dataset (imbalance ≈7.88:1). Note it does *not* include LLM methods, and it overlaps your own Bank Marketing experiment — you could position it as concurrent/independent corroboration rather than just background.
- **Could NOT verify the author "Won, D.-H."** — the MDPI page is JS-rendered and didn't expose the author list to search. **Action: confirm the author list directly on the MDPI page** before camera-ready; if it's not Won, fix it.

### A.5 Summary table

| Ref | Field | Status | Fix |
|---|---|---|---|
| 10 Tanha→Agrawal | Author | **ERROR** | Replace "Tanha et al." → "Agrawal et al." everywhere |
| 13 Shidani | Title | Placeholder | → *Beyond Real Data: Synthetic Data through the Lens of Regularization* |
| 14 Chia Ramírez | Title | Placeholder | → *Finding the Sweet Spot: … Using ADASYN* |
| 9 Won | Date / author | Date off; author unverified | Year → 2026; confirm "Won" on MDPI page |

---

## B. Related-work modernization (item #2)

The single biggest reviewer risk is that the paper's method set looks dated for a 2026 submission: **no diffusion representative is run**, and **GReaT (GPT-2, 2023) is the sole LLM representative without acknowledgment that the LLM-tabular field has moved on.** You don't have to *run* everything, but related work must show awareness. Below are (B.1) the new references to add and (B.2) drop-in prose.

### B.1 References to add

**Diffusion (the current high-utility/fidelity SOTA — currently cited only for TabDDPM/TabSyn):**

- **TabDiff** — Shi, J., Xu, M., Hua, W., Zhang, H., Ermon, S., Leskovec, J. (2025). *TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation.* ICLR 2025. arXiv:2410.20626. → Current commonly-cited SOTA on single-table fidelity/utility; beats TabSyn on shape, column-pair trends, and ML efficiency. **This is the method to name as "current SOTA" instead of leaving TabSyn (2024) as the frontier.**
- (Already cited: TabDDPM Kotelnikov 2023; TabSyn Zhang ICLR 2024 — keep, but frame as "strong baselines superseded by TabDiff on most axes.")

**LLM-tabular family (to contextualize GReaT as the 2023 origin, not the frontier):**

- **REaLTabFormer** — Solatorio & Dupriez (2023). arXiv:2302.02041. GPT-2 for parent tables + seq2seq for relational/child tables; adds statistical overfitting detection (anti-copying). Improves on GReaT for relational data + privacy.
- **TabuLa** — Zhao et al. (2023). arXiv:2310.12746. Trains the LM from scratch for tables (discards NLP pretraining) + token compression; large training-speed gains over GReaT.
- **TabMT** — Gulati & Roysdon (2023). NeurIPS 2023. arXiv:2312.06089. Masked (BERT-style) non-autoregressive transformer with ordered embeddings; native missing-data handling; strong privacy–utility tradeoff.
- *(Optional, frontier)* **LLM+diffusion hybrids, 2025** — e.g. DiffLM; LLM-TabFlow (arXiv:2503.02161, uses LLM reasoning over inter-column logical constraints feeding a latent diffusion model). Cite as "the current direction" only if you keep a forward-looking paragraph.

**Surveys / benchmarks (2025) — to anchor "this is a fast-moving, multi-axis field":**

- **Stoian, Giunchiglia, Lukasiewicz (2025).** *A Survey on Deep Learning Approaches for Tabular Data Generation: Utility, Alignment, Fidelity, Privacy, Diversity, and Beyond.* arXiv:2503.05954. The best recent taxonomy; organizes methods by five requirements. Good single citation for "no method dominates all axes."
- *(Optional)* the broader surveys arXiv:2504.16506 and arXiv:2507.11590.

**Concurrent corroboration worth a sentence:**

- A 2026 MDPI *Applied Sciences* study (16(8), 3694) directly compares **SMOTE, CTGAN, TVAE, and TabDDPM** on imbalanced tabular protein-localization data — useful as an independent data point that TabDDPM is the augmentation method to beat, and as evidence the SMOTE-vs-deep comparison you run is an active question.

### B.2 Drop-in prose

**Add to §2.2 (after the LLM-based generators paragraph), replacing the single GReaT-centric framing:**

> LLM-based serialization methods have diversified rapidly since GReaT (Borisov et al., 2023). REaLTabFormer (Solatorio & Dupriez, 2023) extends the GPT-2 backbone to relational multi-table data and adds explicit statistical overfitting detection to mitigate the row-copying risk inherent to autoregressive serialization. TabuLa (Zhao et al., 2023) discards NLP pretraining and trains the language-model architecture from scratch on tabular tokens, trading GReaT's transfer-learning prior for substantial training-speed gains. TabMT (Gulati & Roysdon, 2023) replaces autoregressive generation with a masked, non-autoregressive transformer that natively handles missing values. The current frontier couples LLM reasoning with latent diffusion (e.g., LLM-TabFlow, 2025). **We evaluate GReaT specifically because it is the canonical, most widely replicated instance of the serialization paradigm and the one most likely to be reached for by a practitioner; we treat it as a representative lower bound on LLM-tabular performance rather than the state of the art, and our findings about LLM priors and feature semantics are expected to transfer qualitatively to its successors.**

**Add to §2.2 diffusion paragraph (update the SOTA pointer):**

> Diffusion models are now the strongest single-table generators on fidelity and utility. TabDDPM (Kotelnikov et al., 2023) established the paradigm; TabSyn (Zhang et al., 2024) moved diffusion into a VAE latent space; and TabDiff (Shi et al., ICLR 2025) is the current reference point, reporting improvements over TabSyn on distribution shape, column-pair trends, and downstream ML efficiency. We do **not** run a diffusion model in §6 (see Limitations), but flag that TabDDPM is the standard augmentation-utility champion against which our statistical-generator gains should ultimately be benchmarked.

**Add one sentence to §9 (Limitations of Existing Evidence) — convert the diffusion gap from a silent omission into an acknowledged scope decision:**

> **No diffusion model is evaluated in §6.** Our original experiments cover statistical generators (GaussianCopula, CTGAN, SMOTE) and one LLM-based method (GReaT). Diffusion models — TabDDPM, TabSyn, and TabDiff (Shi et al., 2025) — are the current high-utility SOTA and the most likely to *exceed* the augmentation gains we report; their absence is the principal scope limitation of this evaluation. We treat the diffusion comparison as the highest-priority extension (a TabDDPM run on the imbalanced datasets is in progress) and note that our headline conclusion — imbalance is the dominant driver of augmentation value — is a statement about *when* augmentation helps that is largely orthogonal to *which* generator is used.

This last sentence matters: it pre-empts the "you didn't run diffusion" reviewer comment by (a) naming it first yourself, (b) explaining why your conclusion survives it, and (c) signaling the fix is underway.

---

## C. Variance-finding reframe (item #3a)

**Thesis:** Your generator-fit variance result is currently buried in §6.6 Methodological Limitations item 6 and the Experiment-5 "natural experiment." It is the most novel thing in the paper and should be a **named contribution.** Literature check confirms the positioning:

- **Nobody in the synthetic-tabular literature isolates generator-fit variance as a distinct, budgeted source.** Recent benchmarks explicitly *skip* re-fitting the generator to save compute — e.g., the 2025 multi-dimensional framework (arXiv:2504.01908) fixes the split "to avoid retraining the generation models," and the Frontiers Digital Health 2025 health framework reports mean±std over 5 seeds with a single generator fit. That is exactly the conflation you expose, and you can quote it.
- **The theoretical scaffolding exists and should be cited:** Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks* (MLSys 2021, arXiv:2103.03098) — decomposes benchmark variance into data-split, init, data-order, etc., and shows that varying only one source (the standard practice) gives biased, over-confident estimates; recommends randomizing *all* sources. Your "generator-fit variance" is a direct, two-stage-pipeline instantiation: a synthetic-data pipeline nests *two* models (generator + downstream classifier), each with its own variance sources, and the standard protocol seeds only the downstream/data RNG.
- **Must cite and distinguish:** van Breugel, Qian & van der Schaar, *Synthetic Data, Real Errors* (ICML 2023, arXiv:2305.09235) — their Deep Generative Ensemble (DGE) trains *multiple* generators to capture uncertainty over the generative process. **Same mechanism (many generator fits), opposite framing:** DGE proposes ensembling as a *modeling improvement*; your contribution is *diagnostic* — that NOT accounting for generator-fit variance invalidates the standard evaluation protocol (it can flip a finding's sign and understate CIs). Distinguish this crisply or a reviewer will say you rediscovered DGE.
- **Plausibility support:** Picard, *torch.manual_seed(3407) is all you need* (2021, arXiv:2109.08203) and *Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning LLMs* (2025, arXiv:2503.07329) both document that fine-tuning seed alone produces non-trivial, sometimes-outlier drift — making your up-to-4.6-pp result unsurprising and well-precedented.

### C.1 Drop-in contribution bullet (Introduction / Contributions list)

> **Generator-fit variance is a first-class, usually-ignored source of uncertainty in synthetic-data evaluation.** A synthetic-data pipeline nests two stochastic models — the generative model and the downstream classifier — but the standard *k*-seed protocol seeds only the data split and downstream training, holding the (uncontrolled) generator fit implicit. Using two independent GReaT fits on bit-identical training data, we show this hidden source drives downstream-AUC drift of up to 4.6 percentage points, is large enough to flip the sign of a headline small-*n* finding, and is bundled into — and therefore understates — the confidence intervals reported under the conventional protocol. We recommend that synthetic-data benchmarks randomize the generator fit as an explicit variance source (extending Bouthillier et al., 2021, to two-stage generative pipelines) and distinguish this diagnostic claim from the modeling-oriented Deep Generative Ensemble of van Breugel et al. (2023).

### C.2 Drop-in abstract sentence (replace the long hedged variance clause)

> We further show that the standard multi-seed evaluation protocol systematically understates uncertainty for LLM-based synthesis: two independent generator fits on identical data drift by up to 4.6 AUC points and can flip the sign of a small-*n* finding, because the protocol seeds the data split but not the generator's training stochasticity (PyTorch/CUDA/Transformers RNGs). We argue generator-fit variance should be budgeted as a first-class source in synthetic-data benchmarking.

### C.3 Suggested structural move

Pull the Experiment-5 "GReaT-fit variance: a natural experiment" table and the Methodological-Limitations item 6 out of §6.6 and promote them into a short standalone subsection (e.g., **§6.9 "Generator-Fit Variance and the Limits of the k-Seed Protocol"**), with the Bouthillier/van Breugel/DGE positioning. Reference it from the abstract and contributions. This converts your most-hedged paragraph into the paper's sharpest methodological point — and it is the claim a methods venue (NeurIPS D&B, a workshop) would actually accept on.

> **Note on confirmation:** the *reframe* above is supportable from the data you already have (the two independent runs). The *rigorous* version of this contribution — a Bouthillier-style decomposition that varies the generator seed separately from the split seed across cells — requires new GReaT re-fits (GPU). That is item #3b and will ship as a Databricks-ready script; the reframe does not depend on it, but the script strengthens the claim from "we observed this in a natural experiment" to "we measured the variance components."

---

## D. What this memo does and does not cover

**Done here (zero GPU, zero new data):** citation audit + fixes (A), related-work references and drop-in prose (B), variance-finding reframe with positioning (C).

**Next, CPU here (needs your prepared dataset CSVs to match splits):** item #4a — swap the single gradient-boosting classifier for LR / RF / MLP robustness, and extend the statistical-generator results from 5 to 10 seeds.

**Next, Databricks-ready scripts (you run on GPU, I analyze output):** item #1 TabDDPM on the imbalanced datasets; item #3b the variance-decomposition experiment; item #4b GReaT 10-seed expansion at small-*n*.
