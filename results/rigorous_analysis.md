# Phase 1 Rigorous Re-Analysis — §6.6 GReaT Findings

**Pre-registered analysis plan:** `papers/improvement_plan.md` (2026-05-16)

**Family size for FDR:** 16 paired tests across 3 datasets × variable n.
**FDR target:** Benjamini-Hochberg q=0.1.
**Equivalence bound (TOST):** |Δ| < 0.5 AUC pts.

## Headline finding decision rule

- **Telco n=50:** Δ=+3.93 pts, paired t=+3.58, raw p=0.0232, BH-FDR p=0.0741, Bonferroni p=0.3706, d_z=1.60, wins=5/5, jack-stable=True

**Decision:** ✓ Headline finding survives BH-FDR. Phase 3 (seed expansion) is OPTIONAL.

## Full per-cell results

| dataset | n | Δ (mean) | Δ ±CI (paired) | d_z | wins/5 | p (paired) | p (FDR) | p (TOST equiv) | jack-stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| German Credit | 50 | -0.0071 | 0.1037 | -0.0845 | 2/5 | 0.8593 | 0.9921 | 0.5207 | False |
| German Credit | 100 | -0.0702 | 0.0387 | -2.2538 | 0/5 | 0.0073 | 0.0583 | 0.9953 | True |
| German Credit | 200 | -0.0592 | 0.063 | -1.1668 | 1/5 | 0.0595 | 0.1586 | 0.9624 | True |
| German Credit | 500 | -0.03 | 0.0231 | -1.6139 | 0/5 | 0.0226 | 0.0741 | 0.9802 | True |
| Telco Churn | 50 | 0.0393 | 0.0305 | 1.6012 | 5/5 | 0.0232 | 0.0741 | 0.9823 | True |
| Telco Churn | 100 | -0.0138 | 0.0275 | -0.6248 | 1/5 | 0.2349 | 0.4176 | 0.7885 | True |
| Telco Churn | 200 | -0.0007 | 0.0239 | -0.0344 | 3/5 | 0.9423 | 0.9921 | 0.3203 | False |
| Telco Churn | 500 | 0.0015 | 0.0088 | 0.2072 | 3/5 | 0.6673 | 0.8897 | 0.1639 | False |
| Telco Churn | 1000 | -0.0021 | 0.0084 | -0.3126 | 1/5 | 0.5231 | 0.8369 | 0.1976 | True |
| Telco Churn | 2000 | 0.0048 | 0.0036 | 1.6499 | 5/5 | 0.021 | 0.0741 | 0.4524 | True |
| Hillstrom | 50 | 0.0225 | 0.043 | 0.6505 | 4/5 | 0.2195 | 0.4176 | 0.8395 | True |
| Hillstrom | 100 | 0.0115 | 0.0681 | 0.2095 | 3/5 | 0.6638 | 0.8897 | 0.5978 | False |
| Hillstrom | 200 | 0.004 | 0.0753 | 0.0657 | 3/5 | 0.8903 | 0.9921 | 0.4859 | False |
| Hillstrom | 500 | -0.0002 | 0.0599 | -0.0047 | 3/5 | 0.9921 | 0.9921 | 0.4179 | False |
| Hillstrom | 1000 | -0.0397 | 0.0531 | -0.9274 | 1/5 | 0.1068 | 0.2441 | 0.9279 | True |
| Hillstrom | 2000 | -0.0687 | 0.0194 | -4.4018 | 0/5 | 0.0006 | 0.0096 | 0.9996 | True |

## Interpretation guide

- **Δ (mean) > 0** = GReaT helps (mean AUC gain in proportion).
- **Δ ±CI (paired)** = 95% CI on the paired difference. If CI excludes zero, the paired test rejects H0:Δ=0 at α=0.05 (unadjusted).
- **d_z** = paired effect size (Cohen's d_z). |d_z| > 0.8 = large; 0.5–0.8 = medium; 0.2–0.5 = small.
- **p (paired)** = raw paired t-test p-value (matched-seed comparison).
- **p (FDR)** = BH-FDR-adjusted p across the 16-test family. Reject if p (FDR) < 0.1.
- **p (TOST equiv)** = two-one-sided-test p for equivalence within ±0.5 AUC pts. If p (TOST) < 0.05, statistically confirmed as 'tied'.
- **jack-stable** = does dropping any single seed leave the sign of Δ unchanged?
