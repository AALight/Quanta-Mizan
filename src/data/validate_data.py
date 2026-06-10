"""
validate_data.py
================
Validates all processed data files before training.
Checks label consistency, keyword presence, no-error signal logic,
and duplicate detection.

Usage:
    python src/data/validate_data.py

Produces:
    data/processed/validation_report.json
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE      = Path(__file__).parent.parent.parent
PROC_DIR  = BASE / "data" / "processed"
REPORT    = PROC_DIR / "validation_report.json"

SUBSTITUTION_ERROR    = "REF_HAS_KEYWORD_AND_MT_HAS_BESTMATCH"
SUBSTITUTION_NO_ERROR = "REF_HAS_KEYWORD_MT_MISSING_BESTMATCH"
OMISSION_CONTEXT      = "OMISSION_CONTEXT_REF_HAS_KW_MT_HAS_KW"


def keyword_in_text(keyword: str, text: str) -> bool:
    if pd.isna(keyword) or pd.isna(text):
        return False
    return str(keyword).strip() in str(text)


def validate_file(path: Path, label_map: dict) -> dict:
    name = path.name
    print(f"\n{'─'*50}")
    print(f"Validating: {name}")
    print(f"{'─'*50}")

    if not path.exists():
        print(f"  ❌ File not found: {path}")
        return {"file": name, "exists": False}

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  Rows: {len(df)}")

    results = {
        "file":   name,
        "rows":   len(df),
        "exists": True,
        "checks": {},
        "issues":   [],
        "warnings": [],
    }

    passed = True

    # ── Check 1: label consistency ────────────────────────────
    if "Sub-Subtype" in df.columns:
        unknown = set(df["Sub-Subtype"].dropna().unique()) - set(label_map.keys())
        ok = len(unknown) == 0
        results["checks"]["label_consistency"] = ok
        if ok:
            print("  ✅ label_consistency")
        else:
            print("  ❌ label_consistency")
            results["issues"].append(f"Unknown labels found: {unknown}")
            passed = False
    else:
        results["checks"]["label_consistency"] = False
        results["issues"].append("Column 'Sub-Subtype' missing")
        passed = False

    # ── Check 2: duplicate sample_ids ────────────────────────
    if "sample_id" in df.columns:
        dupes = df["sample_id"].duplicated().sum()
        ok = dupes == 0
        results["checks"]["duplicate_ids"] = ok
        if ok:
            print("  ✅ duplicate_ids")
        else:
            print(f"  ❌ duplicate_ids — {dupes} duplicates")
            results["issues"].append(f"{dupes} duplicate sample_ids")
            passed = False

    # ── Check 3: Arabic text quality ─────────────────────────
    if "ar" in df.columns:
        arabic_re = re.compile(r'[\u0600-\u06FF]')
        empty_ar = df["ar"].isna().sum() + (df["ar"].astype(str).str.strip() == "").sum()
        no_arabic = df["ar"].astype(str).apply(lambda x: not bool(arabic_re.search(x))).sum()
        ok = empty_ar == 0 and no_arabic == 0
        results["checks"]["arabic_quality"] = ok
        if ok:
            print("  ✅ arabic_quality")
        else:
            print(f"  ❌ arabic_quality — {empty_ar} empty, {no_arabic} without Arabic chars")
            results["issues"].append(f"Arabic quality: {empty_ar} empty, {no_arabic} non-Arabic")
            passed = False

    # ── Check 4: keyword in reference ────────────────────────
    if "Keyword" in df.columns and "ar" in df.columns:
        kw_in_ref = df.apply(lambda r: keyword_in_text(r["Keyword"], r["ar"]), axis=1)
        kw_missing = (~kw_in_ref).sum()
        ok = kw_missing == 0
        results["checks"]["keyword_in_reference"] = ok
        if ok:
            print("  ✅ keyword_in_reference")
        else:
            print(f"  ⚠️  keyword_in_reference — {kw_missing} not found (likely clitic variants)")
            results["warnings"].append(
                f"{kw_missing} samples where Keyword not found in reference (ar) "
                "— likely Arabic clitic attachment"
            )
            # Save flagged rows for AI review
            flagged = df[~kw_in_ref]
            flagged_path = PROC_DIR / f"ai_verify_flagged_{path.stem}.csv"
            flagged.to_csv(flagged_path, index=False, encoding="utf-8-sig")
            results["warnings"].append(
                f"Flagged rows saved to: {flagged_path.name} — upload to Claude for AI verification"
            )

    # ── Check 5: signal-specific checks ──────────────────────
    if "retrieval_signal" in df.columns:
        # 5a: No-Error omission — keyword must be in mt_output
        omit_ne = df[df["retrieval_signal"] == OMISSION_CONTEXT]
        if len(omit_ne) > 0:
            kw_in_mt = omit_ne.apply(
                lambda r: keyword_in_text(r["Keyword"], r["mt_output"]), axis=1
            )
            fails = (~kw_in_mt).sum()
            ok    = fails == 0
            results["checks"]["omission_no_error_kw_in_mt"] = ok
            if ok:
                print("  ✅ omission_no_error_kw_in_mt")
            else:
                print(f"  ❌ omission_no_error_kw_in_mt — {fails} failures")
                results["issues"].append(
                    f"{fails} OMISSION_CONTEXT samples where Keyword missing from mt_output!"
                )
                passed = False

        # 5b: No-Error substitution — keyword in MT, best_match NOT in MT
        sub_ne = df[df["retrieval_signal"] == SUBSTITUTION_NO_ERROR]
        if len(sub_ne) > 0:
            kw_in_mt = sub_ne.apply(
                lambda r: keyword_in_text(r["Keyword"], r["mt_output"]), axis=1
            )
            bm_in_mt = sub_ne.apply(
                lambda r: keyword_in_text(r["Best_Match"], r["mt_output"]), axis=1
            )
            kw_fails = (~kw_in_mt).sum()
            bm_fails = bm_in_mt.sum()
            ok = kw_fails == 0 and bm_fails == 0
            results["checks"]["substitution_no_error"] = ok
            if ok:
                print("  ✅ substitution_no_error")
            else:
                if kw_fails:
                    results["issues"].append(
                        f"{kw_fails} SUBSTITUTION No-Error where Keyword missing from mt"
                    )
                if bm_fails:
                    results["issues"].append(
                        f"{bm_fails} SUBSTITUTION No-Error where Best_Match IS in mt — these are ERRORS not No-Error!"
                    )
                passed = False

        # 5c: Omission error — keyword must NOT be in mt_output
        omit_err = df[df["retrieval_signal"] == "OMISSION_CONTEXT_REF_HAS_KW_MT_MISSING_KW"]
        if len(omit_err) > 0:
            kw_in_mt = omit_err.apply(
                lambda r: keyword_in_text(r["Keyword"], r["mt_output"]), axis=1
            )
            fails = kw_in_mt.sum()
            ok    = fails == 0
            results["checks"]["omission_error_kw_absent"] = ok
            if ok:
                print("  ✅ omission_error_kw_absent")
            else:
                results["issues"].append(
                    f"{fails} OMISSION errors where Keyword IS in mt_output — not a real omission!"
                )
                passed = False

        # 5d: Substitution error — best_match must be in mt_output
        sub_err = df[df["retrieval_signal"] == SUBSTITUTION_ERROR]
        if len(sub_err) > 0:
            bm_in_mt = sub_err.apply(
                lambda r: keyword_in_text(r["Best_Match"], r["mt_output"]), axis=1
            )
            fails = (~bm_in_mt).sum()
            ok    = fails == 0
            results["checks"]["substitution_error_bm_in_mt"] = ok
            if ok:
                print("  ✅ substitution_error_bm_in_mt")
            else:
                results["issues"].append(
                    f"{fails} SUBSTITUTION errors where Best_Match missing from mt_output!"
                )
                passed = False

    # ── AI spot-check sample ──────────────────────────────────
    sample_n = max(30, int(len(df) * 0.05))
    sample   = df.sample(n=min(sample_n, len(df)), random_state=42)
    cols     = ["sample_id", "Sub-Subtype", "en", "ar", "mt_output", "Keyword", "Best_Match"]
    sample   = sample[[c for c in cols if c in sample.columns]]
    sample_path = PROC_DIR / f"ai_check_{path.stem}.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"\n  📋 AI spot-check sample: {sample_path}")
    print(f"     {len(sample)} rows ({len(sample)/len(df)*100:.1f}% of {len(df)})")

    if results["issues"]:
        print(f"\n  🚨 ISSUES:")
        for issue in results["issues"]:
            print(f"     - {issue}")
    if results["warnings"]:
        print(f"\n  ⚠️  WARNINGS:")
        for warn in results["warnings"]:
            print(f"     - {warn}")

    results["passed"] = passed
    return results


def main():
    print("=" * 70)
    print("RAQEEP CLASSIFIER — DATA VALIDATION")
    print("=" * 70)
    print(f"Processed dir:  {PROC_DIR}")

    label_map_path = PROC_DIR / "label_map.json"
    if not label_map_path.exists():
        print(f"❌ label_map.json not found at {label_map_path}")
        sys.exit(1)
    with open(label_map_path) as f:
        label_map = json.load(f)

    files_to_validate = [
        PROC_DIR / "test_gold.csv",
        PROC_DIR / "test_transfer.csv",
        PROC_DIR / "train_t2.csv",
        PROC_DIR / "no_error_train.csv",
    ]

    all_results = {}
    all_passed  = True

    for f in files_to_validate:
        if f.exists():
            result = validate_file(f, label_map)
            all_results[f.name] = result
            if not result.get("passed", True):
                all_passed = False
        else:
            print(f"\n⚠️  Skipping {f.name} — not found")

    # Save report
    with open(REPORT, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Full validation report: {REPORT}")
    print(f"📋 AI check prompt saved: {PROC_DIR}/AI_CHECK_PROMPT.md")

    print("\n" + ("✅ All checks passed." if all_passed else "🚨 SOME CHECKS FAILED — review issues before training."))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
