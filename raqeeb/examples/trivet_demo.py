"""
trivet_demo.py - Three usage patterns for the TRIVET v3.3 validator.

Run from the repo root:
    python raqeeb/examples/trivet_demo.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows (Arabic text → console without UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Make the parent of `raqeeb/` importable (so `from raqeeb.trivet import ...`
# works without pip install). After pip install, this manipulation is harmless.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from raqeeb.trivet import (
    Trivet,
    clitic_aware_match,
    normalize_arabic_ocs,
    calculate_mpqs,
)

print("=" * 70)
print("TRIVET v3.3 demo")
print("=" * 70)

# ─── Pattern 1: low-level — clitic-aware match without ESV ────────────────
print("\n[Pattern 1] Low-level clitic-aware matching")
print("-" * 70)

# Note: clitic_aware_match operates on a TOKEN, not a sentence. To match
# inside a sentence, use the Trivet().match(...) wrapper (Pattern 2 below).
pairs = [
    ("الحالة",  "الحالة"),    # exact token match
    ("حالة",    "الحالة"),    # match via prefix strip (الـ)
    ("تابع",    "تابعين"),    # match via suffix strip (-ين)
    ("غير_موجود", "حالة"),    # no match
]
for kw, tok in pairs:
    matched, rule = clitic_aware_match(tok, kw)
    print(f"  '{kw}' vs '{tok}':  matched={matched}  rule={rule!r}")

# To match inside a sentence, use the wrapper:
print("\n  Trivet().match() finds keywords inside whole sentences:")
t_demo = Trivet()
sentence = "نعرض الحالة الراهنة"
for kw in ["حالة", "الحالة", "غير_موجود"]:
    print(f"    '{kw}' in '{sentence}':  {t_demo.match(kw, sentence)}")

# ─── Pattern 2: full ESV via Trivet wrapper ───────────────────────────────
print("\n[Pattern 2] Full ESV validation via the Trivet wrapper")
print("-" * 70)

t = Trivet()
print(f"  loaded: {t}")

result = t.validate(
    keyword="السلام",
    best_match="الأمن",
    ar_ref="مبدأ السلام في المنطقة",
    mt_output="مبدأ الأمن في المنطقة",
    sub_subtype="Terminology Substitution",
)
print(f"  OCS:    {result.ocs}")
print(f"  CSR:    {result.csr}")
print(f"  CPS:    {result.cps}")
print(f"  SFR:    {result.sfr}     (None = no embedder loaded)")
print(f"  MPQS:   {result.mpqs}")
print(f"  Tier:   {result.quality_tier}")
print(f"  Regime: {result.regime}")
print(f"  Match rule: {result.match_rule}")
print(f"  Accept (Clean_A): {result.accept_clean_a}")
print(f"  Accept (Clean_B): {result.accept_clean_b}")

# ─── Pattern 3: research / ablation — direct function calls ───────────────
print("\n[Pattern 3] Direct function access (research / ablation)")
print("-" * 70)

# Manually compute MPQS from arbitrary CSR/OCS/SFR values
# Weights = (alpha, beta, gamma) for (CSR, OCS, SFR); rubric_v33 defaults.
mpqs = calculate_mpqs(csr=0.92, ocs=1.0, sfr=0.85, weights=(0.45, 0.30, 0.25))
print(f"  calculate_mpqs(0.92, 1.0, 0.85, weights=(0.45,0.30,0.25)) = {mpqs:.4f}")

# Standard Arabic normalization
text = "السَّلامُ عَلَيكُم"
normalized = normalize_arabic_ocs(text, strip_diacritics=True)
print(f"  normalize_arabic_ocs('{text}', strip_diacritics=True) = '{normalized}'")

print("\nDone. See raqeeb/README.md for the full API reference.")
