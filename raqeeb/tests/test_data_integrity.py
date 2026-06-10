"""Test data-file integrity: shapes, schemas, class-name consistency."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def label_map() -> dict:
    with open(DATA / "label_map.json", encoding="utf-8") as f:
        return json.load(f)


def test_label_map_23_classes(label_map):
    """The locked Mizan taxonomy has exactly 23 classes."""
    assert len(label_map) == 23


def test_label_map_ids_contiguous(label_map):
    """Class IDs are 0..22 with no gaps."""
    ids = sorted(label_map.values())
    assert ids == list(range(23))


def test_patterns_use_locked_class_names(label_map):
    """Every class name in patterns_v1.json must appear in the locked label map."""
    with open(DATA / "patterns_v1.json", encoding="utf-8") as f:
        patterns = json.load(f)
    locked = set(label_map.keys())
    for cls in patterns:
        assert cls in locked, f"Pattern uses class name not in label_map: {cls}"


def test_patterns_total_count():
    """Paper claims 177 unique patterns from the gold pool."""
    with open(DATA / "patterns_v1.json", encoding="utf-8") as f:
        patterns = json.load(f)
    total = sum(len(v) for v in patterns.values())
    assert total == 177, f"Expected 177 patterns, got {total}"


def test_patterns_all_23_classes_covered():
    """All 23 classes have at least one pattern (paper Figure 3 / Safety Net)."""
    with open(DATA / "patterns_v1.json", encoding="utf-8") as f:
        patterns = json.load(f)
    assert len(patterns) == 23


def test_t0_canonical_count_and_classes(label_map):
    """T0 canonical seeds: 79 rows (paper §4.5), one or more per class."""
    df = pd.read_csv(DATA / "tier0_gold_79.csv", encoding="utf-8-sig")
    assert len(df) == 79
    assert set(df["Sub-Subtype"].unique()).issubset(set(label_map.keys()))


def test_gold_pool_size():
    """Gold pool: 446 rows (paper §4.1, Figure 3)."""
    df = pd.read_csv(DATA / "gold_pool_446.csv", encoding="utf-8-sig")
    assert len(df) == 446


def test_gold_pool_classes_consistent(label_map):
    """Gold pool class names are remapped to the locked taxonomy."""
    df = pd.read_csv(DATA / "gold_pool_446.csv", encoding="utf-8-sig")
    pool_classes = set(df["Sub-Subtype"].dropna().unique())
    locked = set(label_map.keys())
    bad = pool_classes - locked
    assert not bad, f"Gold pool has class names not in locked taxonomy: {bad}"


def test_rubric_v33_has_23_classes():
    """Rubric CSV documents thresholds for all 23 classes."""
    df = pd.read_csv(DATA / "rubric_v33.csv", encoding="utf-8-sig")
    assert len(df) == 23


def test_etca_prompts_present():
    """Verbatim ETCA prompts are released."""
    audit = DATA / "prompts" / "etca_audit_prompt.txt"
    gen = DATA / "prompts" / "t2_generation_prompt.txt"
    assert audit.exists() and audit.stat().st_size > 500
    assert gen.exists() and gen.stat().st_size > 500
    assert "anchor" in audit.read_text(encoding="utf-8").lower()
    assert "arabic" in gen.read_text(encoding="utf-8").lower()
