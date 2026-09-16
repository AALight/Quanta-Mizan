# ALLaM-7B — Format D (anchor-guided)

Self-contained bundle of the **Format-D (anchor-guided)** ALLaM-7B zero-shot + few-shot
baseline for the ALLaM row in the baselines table (`tab:baselines`).

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
- **Generator:** `script/run_format_d_our_models.py` (the ALLaM section). It writes
  the `metrics.json` + `predictions_test_gold.csv` reproduced here.
- **Test set:** the frozen 438-instance Gold test, full 23/23 class coverage.

## Contents
```
outputs/zero_shot/metrics.json + predictions_test_gold.csv     (F1 0.0495 / acc 0.1484)
outputs/few_shot_5/metrics.json + predictions_test_gold.csv    (F1 0.0981 / acc 0.2283)
per_class/per_class_allam_zero_shot_format_d.txt               (per-class P/R/F1 + macro avg)
per_class/per_class_allam_few_shot_5_format_d.txt
script/run_format_d_our_models.py                              (generator; ALLaM section)
```
