# Raqeeb — Anchor-Constrained Detection of Fine-Grained Arabic MT Errors

A Python toolkit for fine-grained 23-class Arabic MT error classification with anchor-constrained validation, cross-vendor LLM audit, and dual-layer human evaluation. The toolkit can be used in three ways: as a Python library (recommended for research integration), as a CLI (recommended for batch jobs), or as individual components (TRIVET, the classifier, ETCA — for ablations and extensions).

## Quick start (Python library)

```python
from raqeeb import Raqeeb, Trivet, RaqeebClassifier, ETCAAuditor

# Use the validator alone
t = Trivet()
matched = t.match("حالة", "نعرض الحالة الراهنة")            # True (clitic-aware)
result  = t.validate(keyword="السلام", best_match="الأمن",
                     ar_ref="مبدأ السلام", mt_output="مبدأ الأمن",
                     sub_subtype="Terminology Substitution")
print(result.ocs, result.cps, result.mpqs, result.quality_tier)

# Use the classifier alone (loads the released AraBERT v2 fold-3 checkpoint)
clf = RaqeebClassifier(checkpoint_dir="path/to/ckpt", label_map_path="data/label_map.json")
preds = clf.predict([{"keyword": "...", "best_match": "...",
                       "ar_ref": "...", "mt_output": "..."}])

# Use the ETCA cross-vendor audit (requires ANTHROPIC_API_KEY)
auditor = ETCAAuditor()
audited = auditor.audit([{...}])
```

See `examples/trivet_demo.py` for a runnable, end-to-end TRIVET demo.

## Quick start (CLI)

```bash
pip install -r requirements.txt

# Run on the bundled 10-row UN demo
python run_pipeline.py \
    --input examples/un_demo_input.csv \
    --output examples/un_demo_output.csv

# Run on your own data (CSV must have columns: source_en, ar_ref, mt_output)
python run_pipeline.py --input my_input.csv --output my_errors.csv

# Apply Raqeeb to a new domain (medical, legal, etc.) — replace the patterns file
python run_pipeline.py \
    --input medical_corpus.csv \
    --output medical_errors.csv \
    --patterns my_medical_patterns.json
```

## Public API at a glance

```python
# Top-level convenience
from raqeeb import Trivet, RaqeebClassifier, ETCAAuditor
from raqeeb import calculate_mpqs, clitic_aware_match

# TRIVET — anchor-constrained validator
from raqeeb.trivet import (
    Trivet, ValidationResult,
    clitic_aware_match, normalize_arabic_ocs,
    calculate_ocs, calculate_csr, calculate_cps, calculate_sfr, calculate_mpqs,
    classify_quality_tier, get_regime, load_rubric_from_csv,
)

# Classifier — 23-class AraBERT v2 fold-3 wrapper
from raqeeb.classifier import RaqeebClassifier

# ETCA — cross-vendor LLM audit (Claude Sonnet 4)
from raqeeb.etca import ETCAAuditor
```

`raqeeb.lib.*` is internal implementation. Import from `raqeeb.*` (public) instead.

## What you get back

Each row of the output CSV is one detected error candidate, with:

| Column | Meaning |
|---|---|
| `source_row_id` | Index into the input CSV |
| `sub_subtype` | Pattern-matched candidate class (Mizan label, before classifier) |
| `keyword` | Arabic anchor span found in the reference |
| `best_match` | Span found in the MT output (or `[OMITTED]` for total omissions) |
| `ar_ref`, `mt_output` | Source sentences |
| `predicted_label` | Raqeeb classifier's 23-class prediction |
| `confidence` | Softmax probability of the predicted class |

## Components

| File | Role |
|---|---|
| `run_pipeline.py` | End-to-end CLI |
| `lib/trivet_validator.py` | TRIVET v3.3 validator (clitic-aware matching, OCS / CPS / SFR / MPQS) |
| `lib/patterns.py` | Anchor extraction — applies (keyword, best_match) patterns to candidate text |
| `lib/classifier.py` | AraBERT v2 fold-3 classifier wrapper (23 Mizan classes) |
| `lib/raqeeb_encoder.py`, `raqeeb_encoder_shared.py` | Tokenisation + classifier head, byte-for-byte identical to training |
| `data/patterns_v1.json` | 177 frozen anchor patterns from the 446-row expert-annotated gold pool (23/23 class coverage) |
| `data/rubric_v33.csv` | TRIVET v3.3 per-class ESV thresholds (CSR_Min, OCS_Min, SFR_Min/Max, weights, MPQS targets) |
| `data/tier0_gold_79.csv` | 79 T0 canonical seeds (Safety Net Layer 0 — one example per class) |
| `data/gold_pool_446.csv` | 446-row expert-annotated gold pool (Mizan source data) |
| `data/prompts/etca_audit_prompt.txt` | Verbatim Claude Sonnet 4 audit prompt |
| `data/prompts/t2_generation_prompt.txt` | Verbatim GPT-4o T2 generation prompt |
| `data/label_map.json` | 23-class Mizan label → integer id |
| `data/thresholds_v33.json` | TRIVET v3.3 ESV thresholds (Regime F / Regime D, MPQS bands, DQI floor) |
| `examples/un_demo_input.csv` | 10 UN-domain rows for end-to-end testing |

## Cross-domain use (medical, legal, news, …)

The pipeline is **domain-agnostic in framework, domain-flexible in patterns**. To apply Raqeeb to
a new Arabic MT domain:

1. Annotate a small gold sample in your domain (~50–200 rows) labelling the error class and the
   `(keyword, best_match)` anchor pair.
2. Save the anchor pairs as a JSON in the same schema as `data/patterns_v1.json`.
3. Pass it to the CLI via `--patterns my_domain_patterns.json`.

The 23-class Mizan taxonomy applies broadly to Arabic linguistic-shift errors and works as a
**baseline rubric** for new domains. Most classes (Tanween Omission, Definiteness Shift,
Wrong Word Order, Gender Disagreement, Total Omission, etc.) are language-level Arabic phenomena;
**only Terminology Substitution and Register Mismatch are genuinely domain-flavoured** and may
benefit from domain-specific exemplars.

## Reproducing the paper's numbers

Sanity check on the 438-instance UN Gold test set (should reproduce macro-F1 ≈ 0.589, accuracy
≈ 0.578 — fold-3 of the AraBERT v2 release, same as Table 4 in the paper):

```bash
python run_pipeline.py \
    --input ../delta_upload/data/processed/test_gold.csv \
    --output gold_predictions.csv
```

(Note: `test_gold.csv` already has `keyword`/`best_match` columns prepared, so the candidate-extraction
step is a no-op for that file. The CLI handles either input format.)

## Dependencies

See `requirements.txt`. Core: `pandas`, `torch`, `transformers`, `scikit-learn`, `tqdm`.
