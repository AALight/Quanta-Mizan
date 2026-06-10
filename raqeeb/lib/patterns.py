"""
patterns.py — Load anchor patterns and apply them to candidate text.

The Raqeeb pipeline expects a JSON of (keyword, best_match) patterns grouped
by Mizan class. Patterns can be:
  1. The frozen UN-T1 patterns shipped in data/un_t1_patterns.json (default)
  2. A user-supplied JSON in the same schema (medical, legal, news, etc.)

This module:
  - loads the JSON
  - flattens to a list of pattern objects
  - matches candidate (ar_ref, mt_output) sentence pairs against patterns
    using literal containment + an optional clitic-aware fallback
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Use the validator's clitic-aware utilities (already shipped in lib/)
from .trivet_validator import clitic_aware_match, normalize_arabic_ocs


def load_patterns(path: str | Path) -> List[dict]:
    """Load a patterns JSON. Returns flat list of {sub_subtype, keyword, best_match}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    flat = []
    for sub, items in data.items():
        for item in items:
            flat.append({
                "sub_subtype": sub,
                "keyword": item["keyword"],
                "best_match": item["best_match"],
            })
    return flat


def find_candidates(
    rows: List[dict], patterns: List[dict],
    use_clitic_aware: bool = True,
) -> List[dict]:
    """
    Scan candidate rows against patterns. A row is a {ar_ref, mt_output, ...} dict.

    For each row, every (sub, keyword, best_match) pattern is tested:
      - keyword must appear in ar_ref (clitic-aware if enabled)
      - best_match must appear in mt_output (clitic-aware if enabled),
        OR best_match == "[OMITTED]" and keyword is absent from mt_output

    Returns a list of candidate dicts (one per pattern x row hit), each with:
      sub_subtype, keyword, best_match, ar_ref, mt_output, source_row_id
    """
    out: List[dict] = []
    for row_id, row in enumerate(rows):
        ar = str(row.get("ar_ref", ""))
        mt = str(row.get("mt_output", ""))
        ar_norm = normalize_arabic_ocs(ar) if use_clitic_aware else ar
        mt_norm = normalize_arabic_ocs(mt) if use_clitic_aware else mt
        for p in patterns:
            kw = p["keyword"]
            bm = p["best_match"]
            kw_norm = normalize_arabic_ocs(kw) if use_clitic_aware else kw
            kw_in_ar = (kw in ar) or (kw_norm in ar_norm) or (
                use_clitic_aware and clitic_aware_match(kw, ar_norm)
            )
            if not kw_in_ar:
                continue
            if bm == "[OMITTED]":
                bm_in_mt = (kw not in mt) and (kw_norm not in mt_norm) and not (
                    use_clitic_aware and clitic_aware_match(kw, mt_norm)
                )
            else:
                bm_norm = normalize_arabic_ocs(bm) if use_clitic_aware else bm
                bm_in_mt = (bm in mt) or (bm_norm in mt_norm) or (
                    use_clitic_aware and clitic_aware_match(bm, mt_norm)
                )
            if not bm_in_mt:
                continue
            out.append({
                "source_row_id": row_id,
                "sub_subtype": p["sub_subtype"],
                "keyword": kw,
                "best_match": bm,
                "ar_ref": ar,
                "mt_output": mt,
                "source_en": row.get("source_en", ""),
            })
    return out
