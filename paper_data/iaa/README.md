# Dual-annotator IAA (Paper 1, Appendix B)

Released artifacts behind the inter-annotator agreement table (`tab:iaa-main`):
two expert linguists (Annotator A / B), *not* the taxonomy designers,
independently re-labelled 147 stratified QUANTA instances on 7 verification
criteria. Reproduces Gwet's AC1 exactly.

## Files
- `phase1_T1_paired_AvsB.csv` - 55 T1 (real-MT) instances, 16 distinct classes
  (frozen from an earlier Stage-A snapshot than the 12-class released T1).
- `phase1_T2_paired_AvsB.csv` - 92 T2 (synthetic) instances, 23 classes (4/class).
- `compute_iaa_ac1.py` - recomputes the AC1 table from the two CSVs.

Columns per file: `Eval_ID`, `Sub_Subtype_{A,B}`, and `Q1..Q7_{A,B}` =
Q1 Error, Q2 Anchor, Q3 Severity (0-5 normalised), Q4 Minimal-pair, Q5 Label,
Q6 Naturalness, Q7 UN-register. No annotator identities are stored (A/B only).

## Reproduced values (`python compute_iaa_ac1.py`)
| Criterion | T1 | T2 |
|---|---|---|
| Error | 0.900 | 0.989 |
| Anchor | 0.900 | 0.978 |
| Label | 0.680 | 0.930 |
| Severity | 0.434 | 0.987 |
| UN-register | - | 0.608 |
| **Core-three** (error/anchor/label) | **0.827** | **0.966** |
| All-reported | 0.729 | 0.898 |

## Notes
- **5 of 7 criteria reported.** Minimal-pair (Q4) and naturalness (Q6) are
  recorded in the protocol but omitted from the table: both drew systematic
  annotator interpretation divergence (e.g. minimal-pair positive rates 0.90 vs
  0.12), so their low agreement reflects protocol ambiguity, not unreliable
  labels. All 7 are retained in these CSVs for transparency.
- **Gwet's AC1, not Cohen's kappa:** the binary criteria are high-prevalence by
  design, the regime where kappa collapses artifactually (kappa paradox;
  Feinstein & Cicchetti 1990; Gwet 2008). The AC1 choice is **not** a
  consequence of the class stratification.
