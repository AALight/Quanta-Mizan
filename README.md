# Raqeeb — Anchor-Constrained Detection of Fine-Grained Arabic MT Errors

> *Anonymized repository for the Raqeeb paper, currently under review at the ACL Rolling Review (ARR).*

This repository releases the five artifacts contributed by the Raqeeb resource paper:

1. **Mizan** — a 23-class taxonomy of Arabic MT linguistic-shift errors, with MQM-compatible severity weights
2. **QUANTA** — the 9,032-instance dataset (738 T1 + 7,856 T2 + 438 Gold)
3. **ETCA** — the cross-vendor LLM-as-judge audit protocol (GPT-4o generator, Claude Sonnet 4 judge)
4. **Raqbench** + the released **Raqeeb classifier** (AraBERT v2, 5-fold mean Gold macro-F1 = 0.614 ± 0.017). All 5 fold checkpoints are released; the **default deployable is fold-0** (best fold, single-fold Gold F1 = 0.635), which reproduces the paper's per-class and Frontier-tier numbers. Fold-3 (single-fold F1 = 0.589) is retained as the **cross-domain / companion-lineage** checkpoint used for the WMT24++ probe and the Paper-2 cascade.
5. **Dual-annotator validation protocol** + reusable workbook templates

QUANTA was built with **TRIVET**, the anchor-constrained validation pipeline (anchor extraction → ESV structural gates; ETCA seed/quality audit runs in parallel — see *QUANTA T2 selection* below). TRIVET's methodology is the contribution of the **companion methods paper** (anonymized for review); its validator code is included here (`raqeeb/trivet/`) solely so QUANTA can be reproduced, and is not claimed as a contribution of this paper.

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

> **XLM-R fold-1 reseed.** The original XLM-R fold-1 run collapsed (val macro-F1 = 0.0041, Gold macro-F1 = 0.0092), which dragged the raw 5-fold mean to 0.4566 ± 0.251. We retrained fold 1 at seed 43; it converged normally (val 0.8397, Gold 0.5499), giving the reported XLM-R mean of **0.565 ± 0.023**. The retrained fold is shipped in `experiments/E04_xlmr_fold1_retrain/results/retrain_fold1_seed_43/`, the raw collapsed run is retained for transparency, and `paper_data/TABLE_main_benchmark_23class.json` / `MASTER_all_paper_numbers_final.json` carry the retrained number. All other encoders and folds are unchanged.

Controlled 12-class data ablation: T1 (738 real) reaches macro-F1 = 0.656 vs T2-12 (4,274 synthetic) = 0.312 → DES (per-1k samples) of 0.889 vs 0.073 → ~12× per-sample advantage for validated real data.

## Repository layout

```
raqeeb-paper1/
├── README.md                                 ← this file
├── LICENSE                                   ← MIT
├── CITATION.cff                              ← citation metadata
├── ARR_REPRODUCIBILITY_CHECKLIST.md          ← ACL/EMNLP reproducibility checklist
├── paper_v2.tex                              ← the paper source (compiles to 34 pages)
├── paper_v2.pdf                              ← compiled PDF
├── references.bib                            ← 46 cited references
│
├── raqeeb/                                   ← the runtime toolkit
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
│   ├── data/
│   │   ├── label_map.json                       23-class Mizan label → integer id
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
├── delta_upload/                             ← training/eval pipeline (Delta GPU runs)
│   ├── data/processed/                          QUANTA splits: train_t1t2.csv, test_gold.csv
│   ├── results/M01/                             5-encoder Raqbench predictions + summaries
│   └── ...
│
├── experiments/                              ← research experiments
│   ├── E01_cross_domain_classifier/             AraBERT v2 fold-3 checkpoint + WMT24++ probe
│   ├── E03_L1_layer1_validation/                Layer-1 sentence-level validation
│   └── E04_xlmr_fold1_retrain/                  XLM-R seed-43 retrain (replaces collapsed fold)
│
├── analysis/                                 ← post-hoc analysis (ETCA validation, etc.)
├── figures/                                  ← rendered figures
├── TRIVET_SCRIPTS/                           ← original research notebooks (TRIVET pipeline)
└── historical_handoff/                       ← validated historical artifacts (gold pool, patterns)
```

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

| Number | Where | Script |
|---|---|---|
| Table 4 mean ± std (5 encoders) | `delta_upload/results/M01/benchmark_23class.json` | `experiments/E04_xlmr_fold1_retrain/scripts/compute_5fold_stats.py` |
| Pairwise paired-bootstrap + BH-FDR | `experiments/E04_xlmr_fold1_retrain/scripts/pairwise_significance.py` | run locally, no Delta needed |
| 12× DES data ablation | `delta_upload/results/M02/data_efficiency.json` | run on Delta |
| Cross-domain WMT24++ | `experiments/E01_cross_domain_classifier/scripts/inference.py` | local |
| Per-class F1 (best fold) | `delta_upload/results/M02/<encoder>/reliable_classification_report.json` | per encoder |

## Citation

See [`CITATION.cff`](CITATION.cff). At review time the paper is anonymized; full citation will appear at camera-ready.

## License

MIT — see [`LICENSE`](LICENSE).

## Ethics

- All translations and audit outputs are released under permissive license alongside the dataset.
- LLM-generated synthetic instances (T2) are clearly labeled and distinct from real annotations (T1, Gold).
- Cross-vendor audit (GPT-4o generator, Claude Sonnet 4 judge) mitigates self-preference bias.
- Annotator compensation, IRB status, and language-data ownership documented in the paper's Ethics Statement.
