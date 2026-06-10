"""Smoke tests for pattern matching and ESV gating."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.patterns import find_candidates, load_patterns
from lib.esv_gates import lightweight_esv, add_esv_columns, filter_clean


@pytest.fixture(scope="module")
def patterns():
    return load_patterns(ROOT / "data" / "patterns_v1.json")


def test_load_patterns_count(patterns):
    """Loader flattens 23 classes × ~7.7 avg = 177 patterns."""
    assert len(patterns) == 177


def test_load_patterns_schema(patterns):
    """Each pattern has the three required keys."""
    for p in patterns:
        assert {"sub_subtype", "keyword", "best_match"} <= set(p.keys())


def test_find_candidates_omitted_keyword(patterns):
    """Total Omission patterns hit when keyword absent from MT output."""
    rows = [{
        "ar_ref": "هذا اختبار للكلمة المفتاحية المهمة",
        "mt_output": "هذا اختبار",
    }]
    omission_patterns = [p for p in patterns
                         if p["sub_subtype"] == "Total Omission"
                         and p["best_match"] == "[OMITTED]"
                         and p["keyword"] in rows[0]["ar_ref"]]
    if not omission_patterns:
        pytest.skip("No omission pattern matches the test sentence")
    cands = find_candidates(rows, omission_patterns, use_clitic_aware=True)
    assert len(cands) > 0
    assert all(c["best_match"] == "[OMITTED]" for c in cands)


def test_lightweight_esv_perfect_match():
    """A candidate with full overlap should produce high ESV-lite score."""
    cand = {
        "keyword": "السلام",
        "best_match": "الأمن",
        "ar_ref": "نص يحتوي على كلمة السلام مهمة جدا",
        "mt_output": "نص يحتوي على كلمة الأمن مهمة جدا",
    }
    esv = lightweight_esv(cand)
    assert esv["ocs"] == 1
    assert esv["cps_proxy"] >= 0.5
    assert esv["mpqs_lite"] > 0.5


def test_lightweight_esv_no_anchor():
    """A candidate where keyword is absent from ar_ref should fail OCS."""
    cand = {
        "keyword": "غير موجودة",
        "best_match": "بديل",
        "ar_ref": "هذا نص آخر تماما",
        "mt_output": "هذا نص آخر تماما",
    }
    esv = lightweight_esv(cand)
    assert esv["ocs"] == 0


def test_filter_clean_strict_vs_loose():
    """Clean_A (strict) accepts strictly fewer than Clean_B (loose)."""
    cands = [
        {"ocs": 1, "mpqs_lite": 0.40},   # rejected by both
        {"ocs": 1, "mpqs_lite": 0.60},   # accepted by Clean_B only
        {"ocs": 1, "mpqs_lite": 0.70},   # accepted by both
        {"ocs": 0, "mpqs_lite": 0.99},   # rejected by both (no anchor)
    ]
    cands_loose = filter_clean([c.copy() for c in cands], strict=False)
    cands_strict = filter_clean([c.copy() for c in cands], strict=True)
    assert len(cands_loose) == 2
    assert len(cands_strict) == 1


def test_demo_input_runs_end_to_end(patterns):
    """The 10-row demo produces non-empty candidates after ESV gating."""
    import pandas as pd
    df = pd.read_csv(ROOT / "examples" / "un_demo_input.csv", encoding="utf-8-sig")
    rows = df.to_dict("records")
    cands = find_candidates(rows, patterns, use_clitic_aware=True)
    assert len(cands) > 100, f"Expected >100 raw candidates, got {len(cands)}"
    cands = add_esv_columns(cands)
    cands = filter_clean(cands, strict=False)
    assert len(cands) >= 1, f"ESV gate produced no accepted candidates"
