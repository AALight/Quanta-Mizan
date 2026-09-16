# Raqeeb — Anchor-Constrained Detection of Fine-Grained Arabic MT Errors

> *Code and data for QUANTA (ArabicNLP 2026): a fine-grained dataset and benchmark for Arabic machine-translation error analysis.*

This repository releases the five artifacts contributed by the Raqeeb resource paper:

1. **Mizan** — a 23-class taxonomy of Arabic MT linguistic-shift errors, with MQM-compatible severity weights
2. **QUANTA** — the 9,032-instance dataset (738 T1 + 7,856 T2 + 438 Gold)
3. **ETCA** — the cross-vendor LLM-as-judge audit protocol (GPT-4o generator, Claude Sonnet 4 judge)
4. **Raqbench** + the released **Raqeeb classifier** (AraBERT v2, 5-fold mean Gold macro-F1 = 0.614 ± 0.017). All 5 fold checkpoints are released; the **default deployable is fold-0** (best fold, single-fold Gold F1 = 0.635), which reproduces the paper's per-class and Frontier-tier numbers. Fold-3 (single-fold F1 = 0.589) is retained as the **cross-domain / companion-lineage** checkpoint used for the WMT24++ cross-domain probe.
5. **Dual-annotator validation protocol** + reusable workbook templates

QUANTA was built with **TRIVET**, the anchor-constrained validation pipeline (anchor extraction → ESV structural gates; ETCA seed/quality audit runs in parallel — see *QUANTA T2 selection* below). TRIVET's methodology is the contribution of a companion methods paper; its validator code is included here (`raqeeb/trivet/`) solely so QUANTA can be reproduced, and is not claimed as a contribution of this paper.

## QUANTA T2 selection (provenance)

The released **7,856 T2** instances are selected by TRIVET's **structural validation** (the ESV anchor gates — OCS / CPS / SFR) followed by the **Partial-Translation / Total-Omission re-classification audit**: of 8,044 generated candidates, 188 are excluded as PT/TO confounds, 15 are relabelled (Partial Translation → Total Omission), and 20 receive corrected `Best_Match` spans, yielding 7,856. Manifests: `paper_data/audit_manifests/excluded_188.csv`, `relabelled_15.csv`, `corrected_20.csv`.

**ETCA is a parallel cross-vendor quality audit, not the T2 selection gate.** It (a) audits the Clean_A / Tier0 generation **seeds** (794 audited → 664 recommended; results in `paper_data/etca_audits/`; prompt in `raqeeb/data/prompts/etca_audit_prompt.txt`, whose legacy title line "Pre-Tier2 Audit Gate" refers to *seed* auditing — the prompt body states it "verifies anchors" and is "NOT reclassifying labels"), and (b) is run as an **ETCA audit** (LLM scores only — not a human evaluation) on a stratified ~5% T2 subset (414 instances, all 23 classes; `paper_data/etca_audits/etca_t2_audit_414.csv`; ETCA per-dimension means ≈ 4.70 / 4.74 / 4.62 of 5). ETCA's external validation against the dual-annotator human consensus (Gwet's AC1 = 0.62) is a *separate* 43-instance Clean_A check, not the 414 subset. The `DQI ≥ 0.75` threshold marks a high-confidence audited instance; it is **not** applied as a filter to the released T2 (the released 7,856 exceed the 6,851 that would pass DQI ≥ 0.75, so DQI cannot have gated the split). The optional `--etca` flag in `raqeeb/run_pipeline.py` applies a `DQI ≥ 0.75 AND seed_recommendation ≥ 4` acceptance rule to *candidates* and is off by default; it did not produce the released dataset.

## Headline numbers

| Encoder | Mean F1 ± std | 95% CI |
|---|---|---|
| **AraBERT v2** | **0.614 ± 0.017** | [0.563, 0.675] |
| CAMeLBERT-Mix | 0.597 ± 0.021 | [0.553, 0.660] |
| CAMeLBERT-MSA | 0.587 ± 0.029 | [0.564, 0.669] |
| ConfliBERT | 0.583 ± 0.021 | [0.566, 0.646] |
| XLM-R | 0.565 ± 0.023 | [0.544, 0.648] |
| TF-IDF + SVM (classical) | 0.237 | — |
| ALLaM-7B (zero-shot) | 0.025 | — |

Pairwise paired-bootstrap with BH-FDR: AraBERT v2 significantly outperforms ConfliBERT (p_adj < 0.001) and XLM-R (p_adj = 0.007); CAMeLBERT-Mix significantly outperforms XLM-R (p_adj = 0.015). Other 7 pairs non-significant.

> **XLM-R fold-1 reseed.** The original XLM-R fold-1 run collapsed (val macro-F1 = 0.0041, Gold macro-F1 = 0.0092), which dragged the raw 5-fold mean to 0.4566 ± 0.251. We retrained fold 1 at seed 43; it converged normally (val 0.8397, Gold 0.5499), giving the reported XLM-R mean of **0.565 ± 0.023**. The retrained number is carried in `paper_data/TABLE_main_benchmark_23class.json` and `paper_data/MASTER_all_paper_numbers_final.json`; the raw collapsed fold-1 predictions are retained for transparency in `paper_data/TABLE_main_preds_XLM-R_fold1.csv`. All other encoders and folds are unchanged.

Controlled 12-class data ablation: T1 (738 real) reaches macro-F1 = 0.656 vs T2-12 (4,274 synthetic) = 0.312 → DES (per-1k samples) of 0.889 vs 0.073 → ~12× per-sample advantage for validated real data.

## Repository layout

```
Quanta-Mizan/
├── README.md                                 ← this file
├── LICENSE                                   ← CC-BY-4.0
├── CITATION.cff                              ← citation metadata
├── ARR_REPRODUCIBILITY_CHECKLIST.md          ← ACL/EMNLP reproducibility checklist
├── requirements.txt                          ← top-level pinned dependencies
│
├── raqeeb/                                   ← the runtime toolkit (end-user package)
│   ├── README.md                                end-user usage guide
│   ├── requirements.txt
│   ├── run_pipeline.py                          end-to-end CLI
│   ├── lib/
│   │   ├── trivet_validator.py                  TRIVET v3.3 validator (clitic-aware, OCS/CPS/SFR/MPQS)
│   │   ├── patterns.py                          anchor extraction
│   │   ├── esv_gates.py                         lightweight ESV gate (OCS + token-overlap proxies)
│   │   ├── etca_auditor.py                      Claude Sonnet 4 audit wrapper
│   │   ├── classifier.py                        AraBERT v2 classifier wrapper (default fold-0; fold-3 for cross-domain)
│   │   └── raqeeb_encoder*.py                   training-identical tokenisation/head
│   ├── trivet/wrapper.py                        thin TRIVET pipeline wrapper
│   ├── data/
│   │   ├── label_map.json                       23-class Mizan label → integer id
│   │   ├── canonical_mizan_v7.json              canonical Mizan taxonomy (L1/L2/L4 + severity weights)
│   │   ├── patterns_v1.json                     177 frozen anchor patterns (full set, all 23 classes)
│   │   ├── rubric_v33.csv                       TRIVET v3.3 per-class ESV thresholds
│   │   ├── thresholds_v33.json                  high-level threshold reference
│   │   ├── tier0_gold_79.csv                    79 T0 canonical seeds (Safety Net layer 0)
│   │   ├── gold_pool_446.csv                    446-row expert-annotated gold pool
│   │   └── prompts/
│   │       ├── etca_audit_prompt.txt            verbatim ETCA judge prompt (Claude Sonnet 4)
│   │       └── t2_generation_prompt.txt         verbatim T2 generator prompt (GPT-4o)
│   └── examples/
│       ├── un_demo_input.csv                    10-row UN demo input
│       └── un_demo_output.csv                   produced by the CLI
│
├── paper_data/                              ← every table/figure number in the paper (see paper_data/README.md)
│   ├── README.md                                per-file guide + critical usage notes
│   ├── MASTER_all_paper_numbers_final.json      single roll-up of all reported numbers
│   ├── TABLE_main_benchmark_23class.json        Table 4: 5-encoder mean ± std + 95% CIs
│   ├── TABLE_paired_bootstrap_significance.json pairwise paired-bootstrap + BH-FDR
│   ├── TABLE_perclass_F1_*.json                 per-class F1 per encoder (+ reliable10 variants)
│   ├── TABLE_{L1_4tqa,L2_5parent,L3_14subtype,multilevel_L1_L4}.json   multi-level benchmarks
│   ├── TABLE_baselines.json                     TF-IDF+SVM and ALLaM zero-shot baselines
│   ├── FIG3_data_efficiency_12class_ablation.json   12x DES real-vs-synthetic ablation
│   ├── FIG5_*_cross_domain_*.{csv,json}         cross-domain WMT24++ probe
│   ├── TABLE_main_preds_<encoder>_fold*.csv     frozen per-fold Gold predictions (438 instances)
│   ├── {gold_test_438,train_T1T2_8594,train_T2_only}.csv   QUANTA splits
│   ├── audit_manifests/                         T2 PT/TO re-classification manifests (188 / 15 / 20)
│   ├── etca_audits/                             ETCA seed + 414-subset audits + external-validation bundle
│   ├── iaa/                                     dual-annotator IAA bundle (Gwet AC1)
│   └── baselines/ · allam_baseline/ · format_ablation/   baseline + ablation artifacts
│
├── src/                                     ← training / data-prep pipeline (research code)
│   ├── data/        prepare_splits.py, validate_data.py, validate_taxonomy.py
│   ├── models/      anchor_extractor.py, encoder.py
│   └── training/    trainer.py, evaluator.py, train_anchor_extractor.py
│
└── tools/                                   ← regen_figures.py (rebuild all figures), restore_cross_domain.py
```

> The paper PDF/source and `references.bib` are **not** included in this repository.

## Quick start

```bash
cd raqeeb
pip install -r requirements.txt

# Run the demo end-to-end (10 UN sentences → 23-class predictions)
python run_pipeline.py \
    --input examples/un_demo_input.csv \
    --output examples/un_demo_output.csv

# With ETCA audit (requires ANTHROPIC_API_KEY)
python run_pipeline.py \
    --input my_input.csv --output errors.csv --etca

# Apply Raqeeb to a new domain — replace the patterns file
python run_pipeline.py \
    --input medical_corpus.csv --output errors.csv \
    --patterns my_medical_patterns.json
```

## Reproducing the paper's numbers

All reported numbers are shipped as frozen JSON/CSV under `paper_data/` (read `paper_data/README.md` first — it documents per-file usage and a critical macro-F1-vs-accuracy caveat for the ablation CSVs). No GPU or retraining is required to re-derive the paper's tables.

| Number | Read from |
|---|---|
| Table 4 mean ± std (5 encoders) + 95% CI | `paper_data/TABLE_main_benchmark_23class.json` (rolled up in `MASTER_all_paper_numbers_final.json`); per-fold preds in `paper_data/TABLE_main_preds_<encoder>_fold*.csv` |
| Pairwise paired-bootstrap + BH-FDR | `paper_data/TABLE_paired_bootstrap_significance.json` |
| 12× DES data ablation | `paper_data/FIG3_data_efficiency_12class_ablation.json` (use the JSON, not the fold CSVs) |
| Cross-domain WMT24++ | `paper_data/FIG5_8_cross_domain_stats.json` + `paper_data/FIG5_*_cross_domain_*.csv` |
| Per-class F1 (best fold) | `paper_data/TABLE_perclass_F1_<encoder>.json` (+ `_reliable10_` variants) |
| Multi-level L1–L4 benchmarks | `paper_data/TABLE_{L1_4tqa,L2_5parent,L3_14subtype,multilevel_L1_L4}.json` |
| All paper figures (rebuild) | `python tools/regen_figures.py` (reads `paper_data/`) |

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

CC BY 4.0 — see [`LICENSE`](LICENSE).

## Ethics

- All translations and audit outputs are released under permissive license alongside the dataset.
- LLM-generated synthetic instances (T2) are clearly labeled and distinct from real annotations (T1, Gold).
- Cross-vendor audit (GPT-4o generator, Claude Sonnet 4 judge) mitigates self-preference bias.
- Annotator compensation, IRB status, and language-data ownership documented in the paper's Ethics Statement.
