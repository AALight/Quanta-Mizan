"""
validate_taxonomy.py
====================
Enforces taxonomy consistency across EVERY file that encodes V7 Mizan facts:
label_map.json, parent_map.json, mizan_severity_weights.json, train_t2.csv,
train_t1t2.csv, test_gold.csv, test_transfer.csv, prediction CSVs,
docs/MIZAN_TAXONOMY.md, paper writing contract, and the Appendix A .tex.

Treats `data/processed/canonical_mizan_v7.json` as the SINGLE SOURCE OF TRUTH.
Every other file must agree with it or the validator fails.

Exit code 0 = all checks passed. Exit code 1 = at least one FAIL.
Warnings do not fail the run (printed as WARN).

Usage:
    python src/data/validate_taxonomy.py
    python src/data/validate_taxonomy.py --json     # machine-readable output
    python src/data/validate_taxonomy.py --strict   # warnings count as failures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE       = Path(__file__).parent.parent.parent                 # raqeeb-paper1/
CANONICAL  = BASE / "data" / "processed" / "canonical_mizan_v7.json"
LABEL_MAP  = BASE / "data" / "processed" / "label_map.json"
PARENT_MAP = BASE / "data" / "processed" / "parent_map.json"
MIZAN_WTS  = BASE / "data" / "processed" / "mizan_severity_weights.json"

# Training / test CSVs live in the classifier project, not raqeeb-paper1.
CLASSIFIER = BASE.parent / "Align" / "Clu" / "RaqeepExperiments" / "classifier" / "data" / "processed"
TRAIN_T2   = CLASSIFIER / "train_t2.csv"
TRAIN_T1T2 = CLASSIFIER / "train_t1t2.csv"
TEST_GOLD  = CLASSIFIER / "test_gold.csv"
TEST_XFER  = CLASSIFIER / "test_transfer.csv"

# Diagnosis / prediction CSVs (only files we have)
PRED_ARABERT = BASE / "results" / "diagnosis_arabert" / "predictions_arabert_v2.csv"
PRED_CAMEL   = BASE / "results" / "diagnosis" / "predictions_camelbert_msa.csv"

# Human-facing docs / writing contract
TAXONOMY_MD = BASE / "docs" / "MIZAN_TAXONOMY.md"
WRITING_CTR = BASE.parent / "raqeeb-paper1" / "Paper1_TRIVET_Session_LaTeX.md"
APPENDIX_TX = BASE.parent / "Align" / "Clu" / "Gpt_P" / \
    "TRIVET__A_Taxonomy_Aware_Two_Layer_Validation_Frameworkfor_Arabic_Machine_Translation_Error_Detection__Copy_" / \
    "appendix_mizan.tex"

# ------------------------------------------------------------------
# Tolerances
# ------------------------------------------------------------------
WEIGHT_EPS = 1e-9   # canonical sums (exact)
RATIO_EPS  = 1e-6   # derived ratios

# ------------------------------------------------------------------
# Check registry
# ------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.fails: list[dict[str, Any]] = []
        self.warns: list[dict[str, Any]] = []

    def ok(self, check_id: str, msg: str) -> None:
        self.passes.append(f"[PASS] {check_id}: {msg}")

    def fail(self, check_id: str, msg: str, **ctx: Any) -> None:
        self.fails.append({"id": check_id, "msg": msg, "context": ctx})

    def warn(self, check_id: str, msg: str, **ctx: Any) -> None:
        self.warns.append({"id": check_id, "msg": msg, "context": ctx})

    def summary(self) -> dict[str, Any]:
        return {
            "passes": len(self.passes),
            "fails": len(self.fails),
            "warns": len(self.warns),
            "pass_lines": self.passes,
            "fail_items": self.fails,
            "warn_items": self.warns,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def csv_column(path: Path, col: str) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            raise KeyError(f"column {col!r} not in {path} (found: {reader.fieldnames})")
        return [row[col] for row in reader]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_close(a: float, b: float, eps: float = WEIGHT_EPS) -> bool:
    return math.isclose(a, b, abs_tol=eps)


def file_exists(path: Path, report: Report, check_id: str) -> bool:
    if not path.exists():
        report.fail(check_id, f"required file missing", path=str(path))
        return False
    return True


# ------------------------------------------------------------------
# Checks — CANONICAL integrity (C01-C05)
# ------------------------------------------------------------------
def check_canonical_integrity(canon: dict, report: Report) -> None:
    classes = canon["classes"]
    parent_weights = canon["parent_group_weights"]
    tqa_weights = canon["tqa_category_weights"]

    # C01: 23 classes
    if len(classes) == 23:
        report.ok("C01", f"canonical has 23 classes")
    else:
        report.fail("C01", f"expected 23 classes, found {len(classes)}")

    # C02: Every class has required fields
    required = {"group", "tqa_category", "severity_level", "severity_score", "class_weight", "mqm_dimension"}
    missing = {c: required - set(v.keys()) for c, v in classes.items() if required - set(v.keys())}
    if not missing:
        report.ok("C02", "all 23 canonical classes have required fields")
    else:
        report.fail("C02", f"{len(missing)} classes missing fields", missing=missing)

    # C03: Per-parent class-weight sum == parent_group_weights
    computed_parents: dict[str, float] = {}
    for cls, meta in classes.items():
        g = meta["group"]
        computed_parents[g] = computed_parents.get(g, 0.0) + float(meta["class_weight"])
    mismatches_p = {g: (computed_parents[g], parent_weights.get(g)) for g in parent_weights
                    if not is_close(computed_parents[g], parent_weights[g])}
    if not mismatches_p and set(computed_parents) == set(parent_weights):
        report.ok("C03", f"parent weights reconstruct exactly from class weights ({len(parent_weights)} groups)")
    else:
        report.fail("C03", "parent group weights do not match sum of class weights",
                    computed=computed_parents, declared=parent_weights, mismatches=mismatches_p)

    # C04: Per-TQA class-weight sum == tqa_category_weights
    computed_tqa: dict[str, float] = {}
    for cls, meta in classes.items():
        t = meta["tqa_category"]
        computed_tqa[t] = computed_tqa.get(t, 0.0) + float(meta["class_weight"])
    mismatches_t = {t: (computed_tqa[t], tqa_weights.get(t)) for t in tqa_weights
                    if not is_close(computed_tqa[t], tqa_weights[t])}
    if not mismatches_t and set(computed_tqa) == set(tqa_weights):
        report.ok("C04", f"TQA weights reconstruct exactly from class weights ({len(tqa_weights)} TQAs)")
    else:
        report.fail("C04", "TQA category weights do not match sum of class weights",
                    computed=computed_tqa, declared=tqa_weights, mismatches=mismatches_t)

    # C05: Parent group weights sum to 1.00 AND TQA weights sum to 1.00
    p_total = sum(parent_weights.values())
    t_total = sum(tqa_weights.values())
    if is_close(p_total, 1.0) and is_close(t_total, 1.0):
        report.ok("C05", f"parent group sum = {p_total:.6f}, TQA sum = {t_total:.6f} (both 1.0)")
    else:
        report.fail("C05", "sums do not equal 1.0", parent_sum=p_total, tqa_sum=t_total)

    # C05b: label_ids contiguous 0..22
    ids = sorted(canon["label_ids"].values())
    if ids == list(range(23)):
        report.ok("C05b", "canonical label_ids contiguous 0..22")
    else:
        report.fail("C05b", "label_ids are not 0..22", label_ids=ids)


# ------------------------------------------------------------------
# Checks — label_map.json (L01-L04)
# ------------------------------------------------------------------
def check_label_map(canon: dict, report: Report) -> None:
    if not file_exists(LABEL_MAP, report, "L01"):
        return
    lm = load_json(LABEL_MAP)
    canon_labels = canon["label_ids"]

    # L01: 23 keys
    if len(lm) == 23:
        report.ok("L01", "label_map.json has 23 entries")
    else:
        report.fail("L01", f"label_map.json has {len(lm)} entries, expected 23")

    # L02: identical key set to canonical
    extra = set(lm) - set(canon_labels)
    missing = set(canon_labels) - set(lm)
    if not extra and not missing:
        report.ok("L02", "label_map.json class set matches canonical")
    else:
        report.fail("L02", "class set mismatch", extra=sorted(extra), missing=sorted(missing))

    # L03: identical label_ids
    diffs = {k: (lm.get(k), canon_labels.get(k)) for k in canon_labels if lm.get(k) != canon_labels[k]}
    if not diffs:
        report.ok("L03", "label_map.json ids match canonical exactly")
    else:
        report.fail("L03", "label_id mismatches", diffs=diffs)

    # L04: ids are unique and 0..22
    ids = sorted(lm.values())
    if ids == list(range(23)):
        report.ok("L04", "label_map.json ids are contiguous 0..22")
    else:
        report.fail("L04", "label_map.json ids not 0..22", ids=ids)


# ------------------------------------------------------------------
# Checks — parent_map.json (P01-P04)
# ------------------------------------------------------------------
def check_parent_map(canon: dict, report: Report) -> None:
    if not file_exists(PARENT_MAP, report, "P01"):
        return
    pm = load_json(PARENT_MAP)

    # Build canonical parent_map from canonical JSON
    expected: dict[str, list[str]] = {}
    for cls, meta in canon["classes"].items():
        expected.setdefault(meta["group"], []).append(cls)
    for g in expected:
        expected[g].sort()

    # P01: group count
    if len(pm) == 5:
        report.ok("P01", "parent_map.json has 5 groups")
    else:
        report.fail("P01", f"parent_map.json has {len(pm)} groups, expected 5")

    # P02: group names match canonical exactly
    extra = set(pm) - set(expected)
    missing = set(expected) - set(pm)
    if not extra and not missing:
        report.ok("P02", "parent_map.json group names match V7 canonical")
    else:
        report.fail("P02", "parent group name mismatch", extra=sorted(extra), missing=sorted(missing))

    # P03: membership identical (sorted comparison)
    if set(pm) == set(expected):
        mismatches = {}
        for g in expected:
            pm_sorted = sorted(pm.get(g, []))
            if pm_sorted != expected[g]:
                mismatches[g] = {"pm": pm_sorted, "expected": expected[g]}
        if not mismatches:
            report.ok("P03", "parent_map.json memberships match canonical exactly")
        else:
            report.fail("P03", "parent group memberships differ", mismatches=mismatches)

    # P04: every class appears exactly once
    flat: list[str] = [c for members in pm.values() for c in members]
    if len(flat) == 23 and len(set(flat)) == 23:
        report.ok("P04", "parent_map.json covers exactly 23 classes with no duplicates")
    else:
        report.fail("P04", f"parent_map.json flat count={len(flat)}, unique={len(set(flat))}")


# ------------------------------------------------------------------
# Checks — mizan_severity_weights.json (M01-M06)
# ------------------------------------------------------------------
def check_severity_weights(canon: dict, report: Report) -> None:
    if not file_exists(MIZAN_WTS, report, "M01"):
        return
    mw = load_json(MIZAN_WTS)
    classes = mw.get("classes", {})

    # M01: 23 classes
    if len(classes) == 23:
        report.ok("M01", "mizan_severity_weights.json has 23 classes")
    else:
        report.fail("M01", f"mizan_severity_weights.json has {len(classes)} classes, expected 23")

    # M02: key set match
    extra = set(classes) - set(canon["classes"])
    missing = set(canon["classes"]) - set(classes)
    if not extra and not missing:
        report.ok("M02", "severity_weights class set matches canonical")
    else:
        report.fail("M02", "severity_weights class set mismatch", extra=sorted(extra), missing=sorted(missing))

    # M03: severity_score matches canonical
    score_diffs = {}
    for cls, meta in classes.items():
        if cls in canon["classes"] and meta.get("severity_score") != canon["classes"][cls]["severity_score"]:
            score_diffs[cls] = {"in_file": meta.get("severity_score"),
                                "canonical": canon["classes"][cls]["severity_score"]}
    if not score_diffs:
        report.ok("M03", "severity_score values match canonical")
    else:
        report.fail("M03", "severity_score mismatches", diffs=score_diffs)

    # M04: severity_level matches canonical
    level_diffs = {}
    for cls, meta in classes.items():
        if cls in canon["classes"] and meta.get("severity_level") != canon["classes"][cls]["severity_level"]:
            level_diffs[cls] = {"in_file": meta.get("severity_level"),
                                "canonical": canon["classes"][cls]["severity_level"]}
    if not level_diffs:
        report.ok("M04", "severity_level values match canonical")
    else:
        report.fail("M04", "severity_level mismatches", diffs=level_diffs)

    # M05: tqa_category matches canonical (normalizing "Style"/"Style & Register" variants)
    def normalize_tqa(s: str) -> str:
        return "Style & Register" if s and s.strip().lower().startswith("style") else s

    tqa_diffs = {}
    for cls, meta in classes.items():
        if cls in canon["classes"]:
            got = normalize_tqa(meta.get("tqa_category", ""))
            want = canon["classes"][cls]["tqa_category"]
            if got != want:
                tqa_diffs[cls] = {"in_file": meta.get("tqa_category"), "canonical": want}
    if not tqa_diffs:
        report.ok("M05", "tqa_category assignments match canonical")
    else:
        report.fail("M05", "tqa_category mismatches", diffs=tqa_diffs)

    # M06: derived parent group (shift_group_weight) and tqa weight (tqa_category_weight)
    # These are optional aggregate fields. If present, they must match canonical.
    if any("shift_group_weight" in v for v in classes.values()):
        group_problems = {}
        for cls, meta in classes.items():
            declared = meta.get("shift_group_weight")
            if declared is None:
                continue
            if cls in canon["classes"]:
                canon_group = canon["classes"][cls]["group"]
                canon_weight = canon["parent_group_weights"].get(canon_group)
                if not is_close(float(declared), float(canon_weight), eps=RATIO_EPS):
                    group_problems[cls] = {"in_file": declared, "canonical": canon_weight, "group": canon_group}
        if not group_problems:
            report.ok("M06", "shift_group_weight values match canonical parent weights")
        else:
            report.fail("M06", "shift_group_weight mismatches", diffs=group_problems)


# ------------------------------------------------------------------
# Checks — Training & test CSVs (D01-D08)
# ------------------------------------------------------------------
def check_csv_labels(canon: dict, path: Path, report: Report, check_prefix: str,
                     require_label_id: bool = True) -> None:
    if not file_exists(path, report, f"{check_prefix}01"):
        return

    try:
        rows = csv_rows(path)
    except Exception as e:
        report.fail(f"{check_prefix}01", f"cannot parse CSV: {e}")
        return

    if not rows:
        report.fail(f"{check_prefix}01", "CSV has no data rows")
        return

    # Find the class column
    class_col = None
    for candidate in ("Sub-Subtype", "subtype", "sub_subtype", "shift_type_subtype", "Class", "class"):
        if candidate in rows[0]:
            class_col = candidate
            break
    if class_col is None:
        report.fail(f"{check_prefix}02", "no class column found",
                    columns=list(rows[0].keys()))
        return

    # D01 / D02 / D03: label values are canonical names
    labels = [r[class_col].strip() for r in rows if r.get(class_col)]
    unknown = sorted(set(labels) - set(canon["classes"]))
    known = sorted(set(labels) & set(canon["classes"]))
    if not unknown:
        report.ok(f"{check_prefix}02", f"{path.name}: all {len(labels)} rows use canonical class names ({len(known)} distinct)")
    else:
        report.fail(f"{check_prefix}02", f"{path.name}: found non-canonical class names",
                    unknown_values=unknown, file=str(path))

    # D03 / D04: label_id consistency with label_map
    if require_label_id and "label_id" in rows[0]:
        lm = load_json(LABEL_MAP)
        mismatches = []
        for i, r in enumerate(rows):
            cls = r.get(class_col, "").strip()
            lid = r.get("label_id", "")
            if not cls or lid == "":
                continue
            if cls not in lm:
                continue  # already failed in D02
            try:
                lid_int = int(lid)
            except ValueError:
                mismatches.append({"row": i, "cls": cls, "label_id": lid, "reason": "non-integer"})
                continue
            if lid_int != lm[cls]:
                mismatches.append({"row": i, "cls": cls, "label_id": lid_int, "expected": lm[cls]})
                if len(mismatches) >= 25:
                    break
        if not mismatches:
            report.ok(f"{check_prefix}03", f"{path.name}: label_id column matches Sub-Subtype for all rows")
        else:
            report.fail(f"{check_prefix}03", f"{path.name}: label_id mismatches (showing first 25)",
                        count=len(mismatches), examples=mismatches[:25])


# ------------------------------------------------------------------
# Check — 15-row label_id patch on train_t2.csv (D-PATCH)
# ------------------------------------------------------------------
def check_patched_total_omission(report: Report) -> None:
    """After the 188/20/15 Partial Translation audit, 15 rows were relabelled
    from Partial Translation to Total Omission. They must all have label_id=1."""
    if not TRAIN_T2.exists():
        return
    rows = csv_rows(TRAIN_T2)
    if not rows or "Sub-Subtype" not in rows[0]:
        return
    stale = []
    for i, r in enumerate(rows):
        if r.get("Sub-Subtype", "").strip() == "Total Omission":
            try:
                lid = int(r.get("label_id", -1))
            except ValueError:
                lid = -1
            if lid != 1:
                stale.append({"row": i, "sample_id": r.get("sample_id"), "label_id": lid})
                if len(stale) >= 20:
                    break
    if not stale:
        report.ok("D-PATCH", f"{TRAIN_T2.name}: all Total Omission rows have label_id=1")
    else:
        report.fail("D-PATCH", f"{TRAIN_T2.name}: Total Omission rows with stale label_id",
                    count=len(stale), examples=stale[:20])


# ------------------------------------------------------------------
# Checks — predictions CSV sanity (R01-R03)
# ------------------------------------------------------------------
def check_predictions(canon: dict, path: Path, report: Report, tag: str) -> None:
    if not path.exists():
        report.warn(f"R-{tag}-01", f"predictions CSV missing (ok for models not re-run yet)", path=str(path))
        return
    rows = csv_rows(path)
    if not rows:
        report.fail(f"R-{tag}-01", f"{path.name} is empty")
        return

    # R01: true_label is canonical
    col = "true_label" if "true_label" in rows[0] else "Sub-Subtype"
    true_vals = {r[col].strip() for r in rows if r.get(col)}
    unk = sorted(true_vals - set(canon["classes"]))
    if not unk:
        report.ok(f"R-{tag}-01", f"{path.name}: true_label values are canonical ({len(true_vals)} distinct)")
    else:
        report.fail(f"R-{tag}-01", f"{path.name}: non-canonical true_label values", unknown=unk)

    # R02: predicted_label is canonical
    if "predicted_label" in rows[0]:
        pred_vals = {r["predicted_label"].strip() for r in rows if r.get("predicted_label")}
        unk_p = sorted(pred_vals - set(canon["classes"]))
        if not unk_p:
            report.ok(f"R-{tag}-02", f"{path.name}: predicted_label values are canonical ({len(pred_vals)} distinct)")
        else:
            report.fail(f"R-{tag}-02", f"{path.name}: non-canonical predicted_label values", unknown=unk_p)

    # R03: tqa_category on rows matches canonical
    if "tqa_category" in rows[0] and "Sub-Subtype" in rows[0]:
        bad = []
        for i, r in enumerate(rows):
            cls = r.get("Sub-Subtype", "").strip()
            got = r.get("tqa_category", "").strip()
            if cls in canon["classes"]:
                want = canon["classes"][cls]["tqa_category"]
                # accept "Style" as alias for "Style & Register"
                got_norm = "Style & Register" if got.lower().startswith("style") else got
                if got_norm != want:
                    bad.append({"row": i, "cls": cls, "in_file": got, "expected": want})
                    if len(bad) >= 10:
                        break
        if not bad:
            report.ok(f"R-{tag}-03", f"{path.name}: tqa_category column matches canonical for sampled rows")
        else:
            report.warn(f"R-{tag}-03", f"{path.name}: stale tqa_category values (stored metadata, not used by model)",
                        examples=bad)


# ------------------------------------------------------------------
# Checks — Gold-set overlap with training (G01)
# ------------------------------------------------------------------
def check_gold_overlap(report: Report) -> None:
    """Train and gold must not share sample_ids (strict leak check)."""
    if not (TRAIN_T2.exists() and TEST_GOLD.exists()):
        report.warn("G01", "skipping leak check — CSVs not in expected classifier/ path",
                    train=str(TRAIN_T2), gold=str(TEST_GOLD))
        return

    train_rows = csv_rows(TRAIN_T2)
    gold_rows = csv_rows(TEST_GOLD)

    def extract_ids(rows: list[dict[str, str]]) -> set[str]:
        for col in ("sample_id", "Sentence_ID", "id"):
            if rows and col in rows[0]:
                return {r[col] for r in rows if r.get(col)}
        return set()

    train_ids = extract_ids(train_rows)
    gold_ids  = extract_ids(gold_rows)
    overlap   = train_ids & gold_ids
    if not overlap:
        report.ok("G01", f"no sample_id overlap between train ({len(train_ids)}) and gold ({len(gold_ids)})")
    else:
        report.fail("G01", f"{len(overlap)} sample_ids leak between train and gold",
                    examples=sorted(list(overlap))[:20])


# ------------------------------------------------------------------
# Checks — Docs & writing contract (W01-W04)
# ------------------------------------------------------------------
def check_docs(canon: dict, report: Report) -> None:
    # W01: writing contract + docs must mention "5 parent families" (or 5 groups) and V7 TQA weights
    if file_exists(WRITING_CTR, report, "W01"):
        text = WRITING_CTR.read_text(encoding="utf-8", errors="ignore")
        # Must reference the 4 TQA weights explicitly
        required_numbers = ["0.42", "0.225", "0.28", "0.075"]
        missing_nums = [n for n in required_numbers if n not in text]
        if not missing_nums:
            report.ok("W01", "writing contract mentions all 4 V7 TQA weights (0.42, 0.225, 0.28, 0.075)")
        else:
            report.warn("W01", "writing contract missing V7 TQA weight numbers", missing=missing_nums)

        # Must NOT still reference the stale "System A" TQA weights.
        # Use distinctive exact forms to avoid false positives:
        #   - 0.4500 is unambiguously the System A Accuracy weight (4 decimal places)
        #   - 0.4375 is the System A Fluency weight (appears nowhere else)
        #   - 0.0525 is the System A Style & Register weight
        stale_nums = ["0.4500", "0.4375", "0.0525"]
        found_stale = [n for n in stale_nums if n in text]
        if not found_stale:
            report.ok("W01b", "writing contract has no System A stale TQA weights")
        else:
            report.warn("W01b", "writing contract still contains System A stale weights — needs revision",
                        found=found_stale)

        # Parent-group count drift
        seven_fam = "7 parent" in text or "seven parent" in text
        five_fam = "5 parent" in text or "five parent" in text
        if five_fam and not seven_fam:
            report.ok("W02", "writing contract says 5 parent families (matches V7)")
        elif seven_fam and not five_fam:
            report.warn("W02", "writing contract says 7 parent families — needs update to 5")
        else:
            report.warn("W02", "writing contract parent-family count ambiguous or missing",
                        mentions_five=five_fam, mentions_seven=seven_fam)

    if file_exists(TAXONOMY_MD, report, "W03"):
        text = TAXONOMY_MD.read_text(encoding="utf-8", errors="ignore")
        if "7 parent" in text or "seven parent" in text:
            report.warn("W03", "docs/MIZAN_TAXONOMY.md still describes 7-family hierarchy — needs update to 5-group V7")
        else:
            report.ok("W03", "docs/MIZAN_TAXONOMY.md has no 7-family drift")

    # W04: Appendix A .tex parity check (flag PT/NEE typo)
    if APPENDIX_TX.exists():
        text = APPENDIX_TX.read_text(encoding="utf-8", errors="ignore")
        # Look for PT=0.08 / NEE=0.08 — known typo
        pt_typo = re.search(r"Partial Translation.*?0\.08", text, flags=re.DOTALL)
        ne_typo = re.search(r"Named? Entity Error.*?0\.08", text, flags=re.DOTALL)
        if pt_typo or ne_typo:
            report.warn("W04", "appendix_mizan.tex still has PT=0.08 and/or NEE=0.08 typo "
                        "(canonical is 0.04 each to make TQA weights sum to 1.00)",
                        pt_typo=bool(pt_typo), nee_typo=bool(ne_typo))
        else:
            report.ok("W04", "appendix_mizan.tex PT/NEE weights match canonical (0.04)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    parser.add_argument("--strict", action="store_true", help="treat WARN as FAIL for exit code")
    args = parser.parse_args()

    report = Report()

    # Canonical must exist first
    if not CANONICAL.exists():
        report.fail("C00", f"canonical_mizan_v7.json not found at {CANONICAL}")
        print_report(report, args.json)
        return 1

    canon = load_json(CANONICAL)

    # Run check suites
    check_canonical_integrity(canon, report)
    check_label_map(canon, report)
    check_parent_map(canon, report)
    check_severity_weights(canon, report)

    # CSV checks
    check_csv_labels(canon, TRAIN_T2, report, "DT2")
    check_csv_labels(canon, TRAIN_T1T2, report, "DT1T2")
    check_csv_labels(canon, TEST_GOLD, report, "DG")
    check_csv_labels(canon, TEST_XFER, report, "DX")
    check_patched_total_omission(report)
    check_gold_overlap(report)

    # Predictions
    check_predictions(canon, PRED_ARABERT, report, "AR")
    check_predictions(canon, PRED_CAMEL, report, "CAM")

    # Docs & writing contract
    check_docs(canon, report)

    print_report(report, args.json)

    failed = len(report.fails) > 0
    if args.strict and len(report.warns) > 0:
        failed = True
    return 1 if failed else 0


def print_report(report: Report, as_json: bool) -> None:
    summary = report.summary()
    if as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    for line in report.passes:
        print(line)
    for w in report.warns:
        print(f"[WARN] {w['id']}: {w['msg']}")
        if w.get("context"):
            for k, v in w["context"].items():
                print(f"         {k}: {v}")
    for f in report.fails:
        print(f"[FAIL] {f['id']}: {f['msg']}")
        if f.get("context"):
            for k, v in f["context"].items():
                print(f"         {k}: {v}")

    print()
    print(f"=== validate_taxonomy.py summary: "
          f"{summary['passes']} PASS, {summary['warns']} WARN, {summary['fails']} FAIL ===")


if __name__ == "__main__":
    sys.exit(main())
