"""
esv_gates.py — Compute the Error Signal Vector (ESV) for each candidate
and apply TRIVET v3.3's per-class structural gates.

ESV components (from the Raqeeb paper, §4.2):
  - OCS (Overlap with Constraint Specification): binary anchor presence check
  - CSR (Context Stability Ratio): how much the non-anchor portion shifted
  - CPS (Context Preservation Score): LaBSE cosine on non-anchor span
  - SFR (Semantic Fidelity Ratio): LaBSE cosine on full sentence
  - MPQS (Multi-Pivot Quality Score): aggregated structural quality

Gates: per the rubric (rubric_v33.csv), Regime F (fidelity, 22 classes) requires
OCS=1, CPS>=floor, SFR>=min. Regime D (Meaning Shift only) requires OCS=1,
CSR<=0.88, 0.45<=SFR<=0.78.

This module provides a *lightweight* implementation that skips the LaBSE
embedding step and relies on token-overlap proxies for CPS / SFR. Toolkit
users who want the full LaBSE-based scoring should call the methods in
trivet_validator.py directly with a loaded LaBSE model.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .trivet_validator import (
    calculate_ocs,
    classify_quality_tier,
)


def lightweight_esv(candidate: dict) -> dict:
    """
    Compute a lightweight ESV without LaBSE.

    Returns a dict with: ocs, cps_proxy, sfr_proxy, mpqs_lite, accepts_basic.

    For the full LaBSE-based scoring, use the validator's run_full_tier(...)
    function with `embedder=load_labse_model()` instead.
    """
    kw = str(candidate.get("keyword", ""))
    bm = str(candidate.get("best_match", ""))
    ar = str(candidate.get("ar_ref", ""))
    mt = str(candidate.get("mt_output", ""))

    # OCS: does the keyword appear in the reference (Regime F prerequisite)?
    ocs = 1 if (kw in ar or kw.strip() in ar) else 0

    # CPS proxy: token-overlap on the non-anchor portion
    ar_tok = set(ar.replace(kw, "").split())
    mt_tok = set(mt.replace(bm, "").split())
    if ar_tok and mt_tok:
        cps_proxy = len(ar_tok & mt_tok) / max(len(ar_tok), 1)
    else:
        cps_proxy = 0.0

    # SFR proxy: token-overlap on the full sentence
    ar_full = set(ar.split())
    mt_full = set(mt.split())
    if ar_full and mt_full:
        sfr_proxy = len(ar_full & mt_full) / max(len(ar_full | mt_full), 1)
    else:
        sfr_proxy = 0.0

    # MPQS-lite: simple weighted aggregate (real MPQS uses LaBSE-based CPS/SFR)
    mpqs_lite = 0.45 * cps_proxy + 0.30 * ocs + 0.25 * sfr_proxy

    # Basic accept rule: OCS=1 and MPQS-lite >= 0.55 (Clean_B)
    accepts_basic = (ocs == 1) and (mpqs_lite >= 0.55)

    return {
        "ocs": ocs,
        "cps_proxy": round(cps_proxy, 4),
        "sfr_proxy": round(sfr_proxy, 4),
        "mpqs_lite": round(mpqs_lite, 4),
        "accepts_basic": accepts_basic,
    }


def add_esv_columns(candidates: List[dict]) -> List[dict]:
    """Add ESV columns to each candidate dict in-place. Returns the same list."""
    for c in candidates:
        esv = lightweight_esv(c)
        c.update(esv)
    return candidates


def filter_clean(candidates: List[dict], strict: bool = False) -> List[dict]:
    """
    Filter candidates by ESV gate. By default: OCS=1 AND mpqs_lite>=0.55.
    With strict=True: also require mpqs_lite>=0.65 (Clean_A tier).
    """
    threshold = 0.65 if strict else 0.55
    return [c for c in candidates if c.get("ocs") == 1 and c.get("mpqs_lite", 0) >= threshold]
