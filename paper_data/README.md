# paper_data/ — Files directly used in the Raqeeb paper

All files here feed a specific table or figure in paper.tex.
Naming convention: `{TABLE|FIG|MASTER}_{description}.{csv|json}`

## ⚠️ CRITICAL USAGE NOTES (READ BEFORE GENERATING ANY FIGURE)

### Figure 3 (data efficiency) — use the JSON, not the fold CSVs

**The correct macro-F1 numbers live in `FIG3_data_efficiency_12class_ablation.json`.**
Read these fields directly:
- `t1_12.mean_gold_f1`       → **0.656**  (T1 real macro-F1)
- `t2_12.mean_gold_f1`       → **0.312**  (T2-12 synthetic macro-F1)
- `t1t2_12.mean_gold_f1`     → **0.740**  (T1+T2-12 combined macro-F1)
- `{cond}.des_12class`       → DES values (0.889, 0.073, 0.148)
- `{cond}.bootstrap_ci_95`   → 95% bootstrap CIs
- `{cond}.fold_results[].gold_f1` → per-fold macro-F1 (already computed correctly)

**DO NOT compute macro-F1 from the per-fold CSVs by averaging the `correct` column.**
That produces ACCURACY, not macro-F1, and the two differ significantly when class
support is imbalanced:

| Condition | Accuracy (wrong) | Macro-F1 (correct) |
|---|---|---|
| T1       | 0.661 | **0.656** |
| T2-12    | 0.474 | **0.312** |
| T1+T2-12 | 0.801 | **0.740** |

The per-fold CSVs (`FIG3_ablation_*_fold*.csv`) are provided for confusion matrices
and per-class error analysis. If you need macro-F1 from them, use
`sklearn.metrics.f1_score(..., average="macro")` — do NOT use `.mean()` on the
`correct` column.

### Figure 9 (SFR violin) — use the FULL cross-domain set

The paper caption specifies "full cross-domain set (clean + noisy)" (n = 6,402).
Concatenate all three files:
- `FIG5_6_7_8_9_cross_domain_cleanA.csv`  (653 rows)
- `FIG5_6_cross_domain_cleanAB.csv`        (1,513 rows; already A+B)
- `FIG5_6_cross_domain_noisy.csv`          (4,889 rows)

**Do NOT use Clean_A alone** — it omits Meaning Shift and Definiteness Shift
entirely (the two most narratively important classes), because Meaning Shift
rows are distributed across Clean_B and noisy, and Definiteness Shift is 0/12
accepted.

## Core datasets
| File | Description |
|------|-------------|
| `gold_test_438.csv` | Gold test set (438 instances, all 23 classes) |
| `train_T1T2_8594.csv` | **Released** post-audit T1+T2 training data — **8,594 rows (738 T1 + 7,856 T2)** |
| `train_T2_only.csv` | **Released** post-audit T2-only synthetic training data — **7,856 rows** |
| `train_T1T2_raw_8782.csv` | Pre-audit superset (8,782 rows = 738 T1 + 8,044 T2), before the T2 re-classification audit |
| `train_T2_only_raw_8044.csv` | Pre-audit T2 superset (8,044 rows) |
| `mizan_taxonomy_v7_weights.json` | Full Mizan V7 taxonomy with severity weights |

**T2 re-classification audit (raw → released).** The released T2 (7,856) is the pre-audit
generation set (8,044) after the Partial-Translation/Total-Omission audit recorded in
`audit_manifests/`: **−188** excluded PT/TO confounds (`excluded_188.csv`), **15** relabelled
Partial Translation → Total Omission (`relabelled_15.csv`, `label_id` → 1), and **20** corrected
`Best_Match` spans (`corrected_20.csv`). Applying these to the `*_raw_*` files reproduces the
released files exactly (`8,044 − 188 = 7,856`; T1 = 738 throughout, so `738 + 7,856 = 8,594`).
The matching `T1 (real)` column is `tier = TIER1` (738 rows, 12 classes); Stage-A Clean_A spans
15 of 23 classes, of which Gold-overlap removal drops 3 (Literal Translation, Perfective to
Progressive, Tense Shift Under Negation) to give T1's 12.

## Tables
| Prefix | Paper section | What it feeds |
|--------|--------------|---------------|
| `TABLE_main_benchmark_23class` | Table 3 (Main benchmark) | 5 encoders mean/std/CI F1 |
| `TABLE_main_preds_{Encoder}_fold{N}` | Table 3 + per-class | Per-sample predictions |
| `TABLE_perclass_F1_{Encoder}` | Appendix G per-class F1 | Full classification report |
| `TABLE_perclass_reliable10_{Encoder}` | Appendix G reliable | 10-class (n>=10) report |
| `TABLE_bootstrap_CI_{Encoder}` | Table 3 CIs | Bootstrap confidence intervals |
| `TABLE_baselines` | Table 4 (Baselines) | Majority/TF-IDF/SVM/ALLaM |
| `TABLE_paired_bootstrap_significance` | Section 7 significance | All pairwise encoder tests |
| `TABLE_multilevel_L1_L4` | Section 7.4 multi-level | L1-L4 macro-F1 summary |
| `TABLE_crosslevel_bootstrap` | Section 7.4 significance | Adjacent-level comparisons |
| `TABLE_L{1,2,3}_*_benchmark` | Section 7.4 | Per-level aggregated results |
| `TABLE_scoring_validation_W2` | Appendix C scoring | MQS weight validation |
| `TABLE_MQS_raqeeb_all_models` | Section 7 scoring | MQS scores across 5 encoders |

## Figures
| Prefix | Paper figure | Chart type |
|--------|-------------|------------|
| `FIG3_data_efficiency_12class_ablation.json` | Figure 3 | 2-panel bar chart |
| `FIG3_ablation_{cond}_fold{N}.csv` | Figure 3 raw data | Per-fold predictions |
| `FIG4_confusion_matrix_arabert_fold3.csv` | Figure 4 | 23x23 confusion matrix heatmap |
| `FIG5_6_7_8_9_cross_domain_cleanA.csv` | Figures 5-9 | All cross-domain Clean_A data |
| `FIG5_8_cross_domain_stats.json` | Figures 5, 8 | Class counts + exclusion reasons |
| `FIG5_9_validation_stats.json` | Figures 5, 9 | Validation pass rates |

## Master references
| File | Description |
|------|-------------|
| `MASTER_all_paper_numbers_final.json` | Final assembled numbers, all phases (canonical; matches the paper) |
