# Raqeeb — Experiments & Artefacts Manifest

Single source of truth for what exists on disk, where, and its status. Read this first when continuing work on the paper, release, or classifier publication.

Last updated: 2026-04-22

---

## 1. Paper

**Location:** `paper.tex` (in the paper source repository)

- ~35 pages, ArabicNLP 2026 target
- Status (per last working session): submission-ready pending final compile
- Recent changes: 13 captions reduced, "Safety Net" named (three-layer coverage),
  abstract IAA fixed (0.90 Tier-2 / 0.73 Tier-1), dataset = 9,032, §4 TRIVET
  expanded with 4 equations + 3 Stage-A tables, Future Work trimmed with
  appendix pointer.
- Compile: `xelatex -interaction=nonstopmode paper.tex` (run twice for refs).

---

## 2. Reviewer reproducibility bundle (the release)

**Location:** `raqeeb_paper1_release/`

Self-contained, 24 MB. Each subfolder has: `README.md`, `config.yaml`,
`requirements.txt`, `run_all.bat`/`run_all.sh`, `scripts/`, `data/`, `results/`.

| Folder | Reproduces | Runtime | Needs extra files? |
|---|---|---|---|
| `E00_un_benchmark/` | AraBERT v2 macro-F1 = 0.614 on 438-instance Gold | < 30 s | No — frozen prediction CSVs |
| `E01_cross_domain_classifier/` | Cross-domain Total-Omission F1 = 0.647 on WMT24++ (n=447) | ~2 min GPU / ~20 min CPU | **Yes** — 540 MB AraBERT v2 fold-3 checkpoint (see §5 below) |
| `E02_data_ablation/` | 12× per-sample real-vs-synthetic advantage; T1+T2-12 = 0.740 | < 20 s | No |
| `E03_baselines/` | TF-IDF+SVM = 0.237; ALLaM zero-shot = 0.025; anchor effect = +0.377 | < 5 s | No |
| `E04_iaa/` | Tier-2 core-three Gwet AC1 = 0.966; Tier-1 core-three = 0.827 | < 3 s | No — anonymised per-Eval_ID CSV |

---

## 3. Working experiments folder (dev copy, paper repo)

**Location:** `experiments/`

Same five subfolders (E00–E03; E04 lives only in the release bundle). This is
the working copy used during paper development. Keep in sync with the release
bundle if either changes.

---

## 4. Training pipeline (authoritative training record)

**Location:** `delta_upload/`

Not shipped in the release bundle. This is the original training pipeline on
the delta cluster, retained as the record of truth for the 23-class classifier
and its sub-level variants.

### Classifier benchmark results (JSON + per-fold CSVs, **no weights**)

| Level | Classes | Folder | Best fold | Best Gold F1 | Mean Gold F1 | 95% CI |
|---|---|---|---|---|---|---|
| 4-class TQA | 4 | `results/P05/` (`benchmark_4tqa.json`) | **fold 2** | 0.6893 | 0.6557 ± 0.0216 | [0.634, 0.740] |
| 14-subtype | 14 | `results/P01/` (`benchmark_14subtype.json`) | (read JSON) | — | — | — |
| 5-parent | 5 | `results/P03/` (`benchmark_5parent.json`) | (read JSON) | — | — | — |
| 23-class | 23 | `results/M01/` (`benchmark_23class.json`) | (read JSON) | — | — | — |

**Each JSON contains:** per-fold val_f1, gold_f1, gold_acc, best_fold, bootstrap CI.
**Each CSV is:** per-fold predictions on the Gold test set (438 instances).
**What is NOT on disk:** model weights (`.safetensors`, `pytorch_model.bin`). Only
the cross-domain E01 fold-3 checkpoint is saved locally (see §5).

> Note: paper text sometimes refers to "15-class" — this corresponds to the
> 14-subtype setup in `P01/` (14 sub-subtypes + background, depending on framing).
> Confirm exact framing against the current paper before publishing.

---

## 5. Model checkpoints (weights)

### 5.1 On local disk

| Classifier | Path | Size | Purpose |
|---|---|---|---|
| AraBERT v2, cross-domain fold 3 | `experiments/E01_cross_domain_classifier/checkpoints/arabert_v2_fold3/model.safetensors` | ~540 MB | E01 reproduction (WMT24++ probe) |

### 5.2 Missing from local disk (retrieve from delta cluster)

| Classifier | Best fold | Where it should come from |
|---|---|---|
| 4-class TQA | fold 2 | Delta cluster — training run P05 |
| 14-subtype  | (check JSON) | Delta cluster — training run P01 |
| 5-parent    | (check JSON) | Delta cluster — training run P03 |
| 23-class    | (check JSON) | Delta cluster — training run M01 |

**Action required before release:** retrieve best-fold checkpoints for the
three public-facing classifiers (23, 14, 4) from delta, then package per §6.

---

## 6. Release folders for classifier publication (GitHub + HuggingFace)

**Status: NOT YET CREATED.**

Planned structure (one folder per published classifier):

```
raqeeb-classifiers/
├── raqeeb-23class-arabert/        # full Mizan taxonomy
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer files
│   ├── README.md                   # model card (HF format)
│   ├── inference_example.py
│   └── benchmark_results.json      # copy from delta_upload/results/M01/
├── raqeeb-14subtype-arabert/       # sub-subtype level
│   └── (same structure)
└── raqeeb-4tqa-arabert/            # TQA top-level
    └── (same structure)
```

Each model card should include: task description, training data (with paper
citation), intended use, limitations, per-fold metrics, bootstrap CI, example
inputs/outputs, license.

---

## 7. IAA data (anonymised)

**Location:** `raqeeb_paper1_release/E04_iaa/data/iaa_responses_anonymised.csv`

- 147 rows (55 Tier-1 + 92 Tier-2)
- Annotators anonymised as A and B
- Five reported questions only (Q1, Q2, Q3, Q5, Q7); Q4 and Q6 excluded per paper Appendix D
- Raw Excel workbooks and annotator identities reserved for a separate
  methodology-focused publication (see paper §Future Work)

---

## 8. What is NOT included in any of the above

- Raw annotator workbooks (Excel) — reserved for methodology publication
- Full `delta_upload/` in the release bundle — authoritative training record only
- Main-benchmark model weights in the release bundle — E00 re-derives metrics from frozen prediction CSVs without retraining; only E01's fold-3 checkpoint ships
- MQS end-to-end calibration against human MT quality ratings — future work

---

## 9. Open tasks (at time of writing)

1. Final xelatex compile of `paper.tex` — confirm no errors, no undefined refs, page count stable.
2. Retrieve best-fold checkpoints for 23 / 14 / 4 classifiers from delta cluster.
3. Build `raqeeb-classifiers/` release folders per §6.
4. Publish to GitHub + HuggingFace under permissive license.
5. Camera-ready version after ArabicNLP review.
