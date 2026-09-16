# Format Ablation (Format A blind vs Format D anchor)

Self-contained bundle for the **format ablation** (`tab:format-ablation`) and the
**Format-A (blind BERT) accuracy** in `tab:baselines`.

## Verified results (AraBERT v2, 5-fold, n=438 Gold, 23-class)
| Format | Input | macro-F1 (mean ± std) | best fold | accuracy (mean) |
|---|---|---|---|---|
| **A (blind)** | `[CLS] ar mt [SEP]` | **0.1135 ± 0.0302** | 0.1609 | **0.232** |
| **D (anchor)** | `[CLS] kw bm [SEP] ar mt [SEP]` | **0.6185 ± 0.0225** | 0.6397 | 0.6028 |

- **Δ (D − A) = +0.505 (+445% relative)**
- **Paired bootstrap** (10k resamples on best-fold gold preds): one-sided **p < 0.0001**
  (`p_value_one_sided = 0.0`), 95% CI of diff **[0.3999, 0.5358]**. (The file's
  `paired_bootstrap.mean_diff = 0.4697` is the resampled best-fold diff; the headline
  Δ = +0.505 is the difference of the 5-fold means.)
- Format-A per-class (best fold, in `per_class_format_ablation.txt`): accuracy 0.240,
  macro F1 0.161. The paper's `tab:baselines` uses the **5-fold mean accuracy 0.232**
  (consistent with its F1-mean convention).

## Provenance / how it was produced
- **Generator:** `run_format_ablation.py` — AraBERT v2 (`aubmindlab/bert-base-arabertv02`),
  lr 5e-5, batch 32, 5 epochs, max_len 512, class-weighted CE.
- **Data:** fixed T1+T2 (**8,594** train) + **438** Gold, 23 classes — same data as the main
  benchmark.
- **Folds:** the script's own `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  with rare classes (<5 instances) always placed in train; per-fold seed `= fold`.
  **Format A and Format D use IDENTICAL folds with each other** (clean ablation — this is the
  "identical folds/hyperparameters" the paper claims).
- **Caveat (folds vs main benchmark):** this is **NOT** `shared.prepare_stratified_folds`
  (used by m01). So the folds are not byte-identical to the main 23-class benchmark — tell-tale:
  Format-D mean here is 0.6185 (best 0.6397) vs the benchmark AraBERT 0.6144 (best 0.6348).
  Same data + same 5-fold CV protocol, different fold construction. Paper wording should say
  "same fixed T1+T2 data; identical folds across the two formats," not "same folds as the main
  benchmark."

## Caveat — no per-instance predictions
`run_format_ablation.py` saves only the aggregate JSON + per-class report; it keeps the
best-fold predictions in memory (for the bootstrap) and never writes a predictions CSV. So:
- **No Format-A (blind) per-instance prediction file exists anywhere.**
- The numbers above are **aggregate-sourced** (this `format_ablation.json` +
  `per_class_format_ablation.txt`). To produce per-instance predictions you must **re-run on a
  GPU** (10 AraBERT trainings: 5 folds × 2 formats) — not reproducible on CPU.

## Contents
```
format_ablation.json            aggregate: A/D mean±std, best fold, mean acc, per-fold, paired bootstrap
per_class_format_ablation.txt   per-class P/R/F1 for both formats' best fold (+ accuracy lines)
run_format_ablation.py          generator (GPU)
README.md                       this file
```

## Paper mapping
- `tab:format-ablation`: A 0.1135 ± 0.030 (best 0.1609) · D 0.6185 ± 0.023 (best 0.6397) ·
  Δ +0.505 (+445%) · p < 0.0001.
- `tab:baselines` Format-A (blind BERT) accuracy = **0.232** (5-fold mean).

No retraining was performed to stage this bundle; the figures are aggregate-sourced from `format_ablation.json` and `per_class_format_ablation.txt`.
