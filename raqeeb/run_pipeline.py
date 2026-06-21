"""
run_pipeline.py — Raqeeb end-to-end CLI for fine-grained Arabic MT error
detection.

Pipeline:
  1. Load EN/AR-ref/AR-MT triples from --input CSV
  2. Match candidate (keyword, best_match) anchors against the loaded
     pattern set (TRIVET v3.3 anchor extraction; clitic-aware)
  3. Run TRIVET structural gates (OCS / CPS / SFR / MPQS) per candidate
  4. (Optional) ETCA audit via --etca flag (requires Anthropic API key)
  5. Run the Raqeeb classifier (AraBERT v2, default deployable = fold-0) for
     23-class prediction
  6. Write per-candidate output to --output CSV

Example:
    python run_pipeline.py \\
        --input examples/un_demo_input.csv \\
        --output errors.csv \\
        --checkpoint ../checkpoints/arabert_v2_fold0

    # With user-supplied patterns (medical, legal, etc.)
    python run_pipeline.py --input my_input.csv --output errors.csv \\
        --patterns my_domain_patterns.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Local imports (relative to this file's directory)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.classifier import RaqeebClassifier  # noqa: E402
from lib.patterns import find_candidates, load_patterns  # noqa: E402
from lib.esv_gates import add_esv_columns, filter_clean  # noqa: E402
from lib.etca_auditor import audit_candidates  # noqa: E402

DEFAULT_PATTERNS = HERE / "data" / "patterns_v1.json"
DEFAULT_LABEL_MAP = HERE / "data" / "label_map.json"
DEFAULT_THRESHOLDS = HERE / "data" / "thresholds_v33.json"
DEFAULT_RUBRIC = HERE / "data" / "rubric_v33.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True,
                    help="Input CSV with columns: source_en, ar_ref, mt_output")
    ap.add_argument("--output", required=True,
                    help="Output CSV path")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to an AraBERT v2 fold checkpoint dir "
                         "(default deployable: ../checkpoints/arabert_v2_fold0, the best fold; "
                         "use ../experiments/E01_cross_domain_classifier/checkpoints/arabert_v2_fold3 "
                         "for the cross-domain checkpoint)")
    ap.add_argument("--patterns", default=str(DEFAULT_PATTERNS),
                    help="Pattern JSON (default: data/un_t1_patterns.json). "
                         "Replace this with a domain-specific patterns file "
                         "to apply Raqeeb to medical, legal, news, etc.")
    ap.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP),
                    help="Label map JSON (default: data/label_map.json)")
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS),
                    help="TRIVET v3.3 thresholds JSON (default: data/thresholds_v33.json)")
    ap.add_argument("--device", default="cpu",
                    help="Inference device (default: cpu; use 'cuda' if GPU available)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-classifier", action="store_true",
                    help="Stop after candidate extraction; skip 23-class prediction")
    ap.add_argument("--no-clitic-aware", action="store_true",
                    help="Disable clitic-aware matching (use literal containment only)")
    ap.add_argument("--skip-esv", action="store_true",
                    help="Skip lightweight ESV gate (OCS / CPS-proxy / SFR-proxy / MPQS-lite)")
    ap.add_argument("--esv-strict", action="store_true",
                    help="Apply Clean_A threshold (mpqs_lite >= 0.65) instead of Clean_B (>= 0.55)")
    ap.add_argument("--etca", action="store_true",
                    help="Run ETCA cross-vendor audit (requires ANTHROPIC_API_KEY env var)")
    ap.add_argument("--dqi-min", type=float, default=0.75,
                    help="ETCA DQI accept threshold (default: 0.75; only used with --etca)")
    args = ap.parse_args()

    print("=" * 78)
    print("Raqeeb pipeline")
    print("=" * 78)

    # 1. Load input
    df_in = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"ar_ref", "mt_output"}
    missing = required - set(df_in.columns)
    if missing:
        print(f"ERROR: input CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)
    print(f"  Input rows:     {len(df_in)}")

    # 2. Load patterns
    patterns = load_patterns(args.patterns)
    print(f"  Patterns:       {len(patterns)} from {Path(args.patterns).name}")

    # 3. Match candidates (anchor extraction)
    rows = df_in.to_dict("records")
    cands = find_candidates(rows, patterns, use_clitic_aware=not args.no_clitic_aware)
    print(f"  Candidates:     {len(cands)} (from {len(rows)} input triples)")

    if not cands:
        print("\nNo candidates found. The pattern set may not match this input domain. "
              "Try loading a domain-specific patterns file via --patterns.", file=sys.stderr)
        df_out = pd.DataFrame(columns=[
            "source_row_id", "sub_subtype", "keyword", "best_match",
            "ar_ref", "mt_output", "predicted_label", "confidence",
        ])
        df_out.to_csv(args.output, index=False, encoding="utf-8")
        print(f"  Empty output written to: {args.output}")
        return

    # 4a. ESV structural gates (lightweight: OCS + token-overlap proxies for CPS/SFR)
    if not args.skip_esv:
        cands = add_esv_columns(cands)
        n_before = len(cands)
        cands = filter_clean(cands, strict=args.esv_strict)
        tier = "Clean_A" if args.esv_strict else "Clean_B"
        print(f"  ESV gate ({tier}):  {n_before} -> {len(cands)} accepted")
        if not cands:
            print("\nAll candidates rejected by ESV gate. Try --no-clitic-aware or --skip-esv.",
                  file=sys.stderr)
            pd.DataFrame(cands).to_csv(args.output, index=False, encoding="utf-8")
            return

    # 4b. ETCA cross-vendor audit (optional, requires Anthropic API key)
    if args.etca:
        print(f"  ETCA audit:     running on {len(cands)} candidates...")
        cands = audit_candidates(cands, dqi_min=args.dqi_min)
        accepted = [c for c in cands if c.get("etca_accepted")]
        print(f"  ETCA accepted:  {len(accepted)} / {len(cands)} (DQI >= {args.dqi_min})")
        cands = accepted

    # 5. Classifier prediction
    if args.skip_classifier:
        df_out = pd.DataFrame(cands)
        df_out["predicted_label"] = ""
        df_out["confidence"] = 0.0
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else (
            HERE.parent / "checkpoints" / "arabert_v2_fold0"
        )
        if not ckpt.exists():
            print(f"ERROR: checkpoint not found: {ckpt}\n"
                  f"  Provide --checkpoint or place the AraBERT v2 fold-0 directory there "
                  f"(fold-0 is the default deployable; fold-3 lives under "
                  f"experiments/E01_cross_domain_classifier/checkpoints for cross-domain use).",
                  file=sys.stderr)
            sys.exit(3)
        print(f"  Classifier:     {ckpt.name}")
        t0 = time.time()
        clf = RaqeebClassifier(
            checkpoint_dir=ckpt,
            label_map_path=args.label_map,
            device=args.device,
            batch_size=args.batch_size,
        )
        preds = clf.predict(cands)
        df_out = pd.DataFrame(cands)
        df_out["predicted_label"] = [p["predicted_label"] for p in preds]
        df_out["confidence"] = [p["confidence"] for p in preds]
        elapsed = time.time() - t0
        print(f"  Classification: {elapsed:.1f}s for {len(cands)} candidates")

    # 6. Write output
    df_out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"  Output:         {args.output}")

    # 7. Summary
    if "predicted_label" in df_out.columns and not args.skip_classifier:
        dist = df_out["predicted_label"].value_counts()
        print("\nPredicted-class distribution (top 10):")
        for cls, n in dist.head(10).items():
            print(f"  {cls:32s} {n:5d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
