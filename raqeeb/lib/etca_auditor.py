"""
etca_auditor.py — ETCA-Lite cross-vendor audit (Claude Sonnet 4 judge).

Runs the verbatim ETCA audit prompt from data/prompts/etca_audit_prompt.txt
against a list of candidates. Returns 5-axis scores + accept/reject decision
+ DQI (Data Quality Index).

Setup:
  1. pip install anthropic
  2. Set the env var ANTHROPIC_API_KEY=your_key_here
  3. Pass --etca to run_pipeline.py to invoke this module

If the anthropic library or API key is missing, the audit is skipped and a
warning is printed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

HERE = Path(__file__).resolve().parent.parent
PROMPT_PATH = HERE / "data" / "prompts" / "etca_audit_prompt.txt"

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_user_message(c: dict) -> str:
    """Format a candidate as the user message for the Claude judge."""
    return (
        f"error_label: {c.get('sub_subtype','')}\n"
        f"keyword: {c.get('keyword','')}\n"
        f"best_match: {c.get('best_match','')}\n"
        f"clean_ar: {c.get('ar_ref','')}\n"
        f"error_ar: {c.get('mt_output','')}\n"
        f"ocs_match_rule: {c.get('ocs_match_rule','exact')}\n\n"
        f"Return your evaluation as a JSON object with keys: "
        f"anchor_validity (1-5), phenomenon_clarity (1-5), arabic_naturalness (1-5), "
        f"collateral_severity (1-5), seed_recommendation (1-5), justification (string)."
    )


def _compute_dqi(scores: dict) -> float:
    """
    DQI: normalized mean of the 4 substantive dimensions
    (excluding seed_recommendation, which is the gate flag).
    """
    keys = ["anchor_validity", "phenomenon_clarity", "arabic_naturalness", "collateral_severity"]
    vals = [scores.get(k, 0) for k in keys]
    if any(v == 0 for v in vals):
        return 0.0
    return round(sum((v - 1) / 4 for v in vals) / 4, 4)


def audit_candidates(
    candidates: List[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    dqi_min: float = 0.75,
) -> List[dict]:
    """
    Audit each candidate via the Claude judge.

    Returns the input list with these added keys per candidate:
      anchor_validity, phenomenon_clarity, arabic_naturalness, collateral_severity,
      seed_recommendation, justification, dqi, etca_accepted

    If anthropic isn't installed or the API key is missing, prints a warning
    and adds 'etca_skipped': True to each candidate.
    """
    try:
        import anthropic
    except ImportError:
        print("WARNING: anthropic library not installed; skipping ETCA audit.\n"
              "  Install with: pip install anthropic", file=sys.stderr)
        for c in candidates:
            c["etca_skipped"] = True
        return candidates

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set; skipping ETCA audit.\n"
              "  Set with: export ANTHROPIC_API_KEY=your_key", file=sys.stderr)
        for c in candidates:
            c["etca_skipped"] = True
        return candidates

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = load_prompt()

    for c in candidates:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": _build_user_message(c)}],
            )
            text = resp.content[0].text
            # Try to parse JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            scores = json.loads(text[start:end]) if start >= 0 else {}
        except Exception as e:
            print(f"WARNING: ETCA audit failed for candidate: {e}", file=sys.stderr)
            scores = {}

        for k in ("anchor_validity", "phenomenon_clarity", "arabic_naturalness",
                  "collateral_severity", "seed_recommendation"):
            c[k] = int(scores.get(k, 0))
        c["justification"] = str(scores.get("justification", ""))
        c["dqi"] = _compute_dqi(scores)
        c["etca_accepted"] = c["dqi"] >= dqi_min and c.get("seed_recommendation", 0) >= 4

    return candidates
