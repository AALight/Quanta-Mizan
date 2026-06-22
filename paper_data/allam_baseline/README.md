# ALLaM-7B — Format D (anchor-guided) — for Paper 1

Self-contained bundle of the **Format-D (anchor-guided)** ALLaM-7B zero-shot + few-shot
baseline for Paper 1. Carry this into the Paper-1 writing session to fix the ALLaM row in
the baselines table (App G, `tab:baselines`) and to add a reproducible artifact to the repo.

**All numbers below were re-verified by recomputing macro-F1/accuracy from the prediction
CSVs in this folder.**

## Verified results (n = 438 Gold, 23 classes)

| Mode | macro-F1 | accuracy | unknown rate |
|---|---|---|---|
| **zero-shot** | **0.0495** | **0.1484** | 0.29 |
| few-shot (5-shot) | 0.0981 | 0.2283 | 0.11 |

These match `outputs/<mode>/metrics.json` exactly and the per-class reports in `per_class/`.

## Provenance
- **Model:** `ALLaM-AI/ALLaM-7B-Instruct-preview` (fp16, greedy decoding, temperature 0.01,
  max_new_tokens 30).
- **Input = Format D (anchor-guided):** the prompt fills **Keyword**, **Best_Match**,
  **Reference (ar)**, **MT Output** — i.e. the same anchor information the fine-tuned
  encoders receive (`[CLS] keyword best_match [SEP] ar mt [SEP]`). This is the
  apples-to-apples LLM comparison.
- **Few-shot:** 5 in-prompt examples drawn from `train_t2` (diverse classes, seed 42).
- **Generator:** `script/run_format_d_our_models.py` (Part 2 = the ALLaM section). It writes
  the `metrics.json` + `predictions_test_gold.csv` reproduced here.
- **Test set:** the frozen 438-instance Gold test, full 23/23 class coverage.

## Why this replaces the paper's current ALLaM row
The paper currently prints **ALLaM-7B zero-shot = macro-F1 0.025, accuracy 0.068**. That is
**not** this Format-D run:
- The **0.025** comes from a *different*, **non-anchor-guided** ALLaM run (prompt = reference
  + MT only; folder `allam_verify/`). It is not comparable to the anchor-guided encoders.
- The **0.068 accuracy** matches **no artifact** found anywhere (the non-anchor zero-shot
  accuracy is 0.119; a "Team B" ALLaM run is referenced in code as F1 0.0262 but was never
  saved). So the printed pair (0.025 / 0.068) is not jointly reproducible.

**Recommended fix:** report the Format-D (anchor-guided) zero-shot number
**macro-F1 = 0.049 (≈0.05), accuracy = 0.148**, optionally adding the 5-shot result
(0.098 / 0.228). The qualitative claim is unchanged — ALLaM zero-shot (0.05) remains far
below the fine-tuned encoders (0.56–0.61), so the 23-class task still "requires task-specific
fine-tuning." Only the specific number and its fairness framing change (now anchor-matched).

## Contents
```
outputs/zero_shot/metrics.json + predictions_test_gold.csv     (F1 0.0495 / acc 0.1484)
outputs/few_shot_5/metrics.json + predictions_test_gold.csv    (F1 0.0981 / acc 0.2283)
per_class/per_class_allam_zero_shot_format_d.txt               (per-class P/R/F1 + macro avg)
per_class/per_class_allam_few_shot_5_format_d.txt
script/run_format_d_our_models.py                              (generator; ALLaM = Part 2)
```

## Notes for the repo
- `metrics.json` records the model only as `"ALLaM-7B"`; pin the full id
  `ALLaM-AI/ALLaM-7B-Instruct-preview` when releasing.
- The generator also trains CAMeLBERT encoders in Part 1; for an ALLaM-only release, keep
  just Part 2 (the ALLaM section) or document that Part 1 is unrelated.
- This bundle was prepared from `exold/AllamFiles/` (already-extracted folders only; no zips
  opened). Source folders: `allam_format_d/`, the loose `per_class_allam_*_format_d.txt`, and
  `delta (1)/run_format_d_our_models.py`.
