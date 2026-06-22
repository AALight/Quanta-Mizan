"""
raqeeb.trivet — TRIVET v3.3 anchor-constrained validator.

Public API
----------
Ergonomic wrapper:
    from raqeeb.trivet import Trivet

Direct function access (research / ablation):
    from raqeeb.trivet import (
        clitic_aware_match,    # binary match with Arabic clitic stripping
        normalize_arabic_ocs,  # standard Arabic normalization
        calculate_ocs,         # anchor presence
        calculate_csr,         # context stability ratio
        calculate_cps,         # context preservation score
        calculate_sfr,         # semantic fidelity ratio (needs embedder)
        calculate_mpqs,        # multi-pivot quality score (aggregator)
        classify_quality_tier, # Clean_A / Clean_B / Noisy decision
        get_regime,            # F (fidelity) vs D (divergence)
        load_rubric_from_csv,  # per-class threshold loader
    )

What it does
------------
TRIVET v3.3 validates that an MT-error candidate (keyword, best_match) pair
corresponds to a real, structurally-valid Arabic linguistic shift, by:

1. Anchor presence — the keyword must appear in the reference (clitic-aware).
2. Context preservation — most of the sentence must remain unchanged.
3. Semantic fidelity — the overall semantic distance falls within bounds
   appropriate for the error type (per-class regime: F or D).

The cascade is the Arabic-specific innovation: 13 prefix patterns + 12 suffix
patterns + diacritic / Tanween / tatweel handling, applied as a 6-step
matching procedure.

Attribution
-----------
TRIVET's methodology (the ESV, the two-regime cascade, regimes, thresholds,
and per-class calibration) is the contribution of the companion methods paper,
not of this resource paper. This validator code is bundled here only as the
production substrate that built QUANTA, so the dataset can be reproduced. See
the companion methods paper for the full method description.
"""
from raqeeb.lib.trivet_validator import (
    calculate_csr,
    calculate_cps,
    calculate_mpqs,
    calculate_ocs,
    calculate_sfr,
    classify_quality_tier,
    clitic_aware_match,
    get_cps_floor,
    get_regime,
    load_rubric_from_csv,
    normalize_arabic_ocs,
    print_rubric_summary,
    strip_arabic_prefix,
    strip_arabic_suffix,
)
from .wrapper import Trivet, ValidationResult

__all__ = [
    # Wrapper class
    "Trivet",
    "ValidationResult",
    # Cascade primitives
    "clitic_aware_match",
    "normalize_arabic_ocs",
    "strip_arabic_prefix",
    "strip_arabic_suffix",
    # ESV components
    "calculate_ocs",
    "calculate_csr",
    "calculate_cps",
    "calculate_sfr",
    "calculate_mpqs",
    # Tier / regime
    "classify_quality_tier",
    "get_regime",
    "get_cps_floor",
    # Rubric
    "load_rubric_from_csv",
    "print_rubric_summary",
]
