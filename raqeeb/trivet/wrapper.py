"""
Trivet — ergonomic wrapper around the TRIVET v3.3 validator.

Provides a single class with high-level methods that hide the per-function
complexity of trivet_validator.py. For direct access to the underlying
functions (useful for research / ablation), import them from raqeeb.trivet
directly.

Example
-------
    from raqeeb.trivet import Trivet

    t = Trivet(rubric_path="path/to/rubric_v33.csv")

    # Just match (boolean — does this Arabic span match this keyword?)
    if t.match("الحالة", "نعرض الحالة الراهنة"):
        print("matched (clitic-aware)")

    # Full ESV validation
    result = t.validate(
        keyword="السلام",
        best_match="الأمن",
        ar_ref="مبدأ السلام في المنطقة",
        mt_output="مبدأ الأمن في المنطقة",
        sub_subtype="Terminology Substitution",
    )
    print(result.ocs, result.cps, result.sfr, result.mpqs, result.quality_tier)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from raqeeb.lib.trivet_validator import (
    calculate_ocs,
    calculate_csr,
    calculate_cps,
    calculate_sfr,
    calculate_mpqs,
    classify_quality_tier,
    clitic_aware_match,
    get_cps_floor,
    get_regime,
    load_rubric_from_csv,
    normalize_arabic_ocs,
)


@dataclass
class ValidationResult:
    """Output of Trivet.validate(). All ESV components + tier decision."""
    ocs: float
    csr: Optional[float]
    cps: Optional[float]
    sfr: Optional[float]
    mpqs: Optional[float]
    quality_tier: Optional[str]   # "Clean_A" / "Clean_B" / "Noisy" / None
    accept_clean_a: bool
    accept_clean_b: bool
    regime: str                   # "F" or "D"
    sub_subtype: Optional[str]
    keyword: str
    best_match: str
    match_rule: Optional[str]      # which clitic rule succeeded, e.g., "exact", "prefix_ال"

    def to_dict(self) -> dict:
        return asdict(self)


class Trivet:
    """High-level wrapper around the TRIVET v3.3 validator.

    Parameters
    ----------
    rubric_path : str | Path | None
        Path to the per-class threshold CSV (rubric_v33.csv). If None, uses
        a default rubric path bundled with the package, falling back to
        zero-thresholds if not found.
    embedder : object | None
        Optional sentence-embedder (e.g., LaBSE SentenceTransformer).
        If None, SFR (Semantic Fidelity Ratio) is skipped — the validator
        runs OCS + CPS only and MPQS is computed without the SFR contribution.
    """

    def __init__(self, rubric_path: Optional[str | Path] = None, embedder=None):
        if rubric_path is None:
            # Try the package's bundled rubric
            here = Path(__file__).resolve().parent.parent
            default = here / "data" / "rubric_v33.csv"
            rubric_path = default if default.exists() else None

        if rubric_path is not None and Path(rubric_path).exists():
            self.rubric = load_rubric_from_csv(str(rubric_path))
        else:
            self.rubric = {}

        self.embedder = embedder

    # ── Low-level helpers ────────────────────────────────────────────────

    def normalize(self, text: str, strip_diacritics: bool = False) -> str:
        """Standard Arabic normalization (Tanween, tatweel, optional diacritics)."""
        return normalize_arabic_ocs(text, strip_diacritics=strip_diacritics)

    def match(self, keyword: str, target: str) -> bool:
        """Clitic-aware match — does the keyword appear in the target text?

        Tokenises `target`, then checks whether any token matches `keyword`
        either literally or via a clitic-stripping rule (prefix ال/و/ب/ل/...
        or suffix ة/ات/ين/ون/...).
        """
        if not keyword or not target:
            return False
        # Literal containment first (fast path)
        if keyword in target:
            return True
        kw_norm = self.normalize(keyword)
        target_norm = self.normalize(target)
        if kw_norm in target_norm:
            return True
        # Clitic-aware token-level cascade
        for tok in target.split():
            matched, _rule = clitic_aware_match(tok, keyword)
            if matched:
                return True
        return False

    def regime(self, sub_subtype: str) -> str:
        """Return 'F' (fidelity) or 'D' (divergence) for this Mizan class."""
        return get_regime(sub_subtype)

    # ── Full pipeline ────────────────────────────────────────────────────

    def validate(
        self,
        keyword: str,
        best_match: str,
        ar_ref: str,
        mt_output: str,
        sub_subtype: Optional[str] = None,
    ) -> ValidationResult:
        """Compute the full ESV + MPQS + tier decision for one candidate.

        Parameters
        ----------
        keyword : str
            Span in the Arabic reference (the anchor expected to appear).
        best_match : str
            Span in the MT output (the corresponding error variant). Use the
            sentinel "[OMITTED]" if the keyword is absent from the MT output.
        ar_ref : str
            The full Arabic reference sentence.
        mt_output : str
            The full Arabic MT output sentence.
        sub_subtype : str | None
            The Mizan class label (e.g., "Terminology Substitution"). Used
            for per-class threshold lookup. If None, default thresholds
            apply.

        Returns
        -------
        ValidationResult
            Dataclass with ocs, csr, cps, sfr, mpqs, quality_tier, accept
            decisions, regime, and the cascade match rule.
        """
        regime = self.regime(sub_subtype) if sub_subtype else "F"
        cps_floor = get_cps_floor(sub_subtype) if sub_subtype else 0.55

        # OCS — anchor presence in reference + best_match presence in MT output
        # Underlying signature: calculate_ocs(clean_ar, error_ar, keyword, best_match, error_type)
        # → returns (ocs_score, failed_constraints, diagnostics)
        ocs_result = calculate_ocs(ar_ref, mt_output, keyword, best_match,
                                    sub_subtype or "Generic")
        if isinstance(ocs_result, tuple):
            ocs_val = float(ocs_result[0])
            ocs_diag = ocs_result[2] if len(ocs_result) > 2 else {}
            match_rule = ocs_diag.get("ocs_match_rule_kw", None) if isinstance(ocs_diag, dict) else None
        else:
            ocs_val = float(ocs_result)
            match_rule = None

        # CSR / CPS / SFR — semantic / context preservation
        csr = calculate_csr(ar_ref, mt_output)
        cps = calculate_cps(ar_ref, mt_output, keyword, best_match)
        sfr = calculate_sfr(ar_ref, mt_output, model=self.embedder) if self.embedder else None

        # MPQS aggregate. The validator signature is
        # calculate_mpqs(csr, ocs, sfr, weights=(alpha, beta, gamma))
        # Default weights from rubric_v33 (CSR=0.45, OCS=0.30, SFR=0.25).
        DEFAULT_MPQS_WEIGHTS = (0.45, 0.30, 0.25)
        weights = self.rubric.get(sub_subtype, {}).get(
            "mpqs_weights", DEFAULT_MPQS_WEIGHTS) if sub_subtype else DEFAULT_MPQS_WEIGHTS

        if csr is not None and sfr is not None:
            mpqs = calculate_mpqs(csr, ocs_val, sfr, weights)
        else:
            mpqs = None

        # Tier decision
        if cps is not None and sfr is not None:
            min_sfr = self.rubric.get(sub_subtype, {}).get("sfr_min_threshold", 0.62) \
                      if sub_subtype else 0.62
            tier = classify_quality_tier(cps, sfr, min_sfr=min_sfr, regime=regime)
        else:
            tier = None

        accept_a = (ocs_val == 1.0) and (mpqs is not None and mpqs >= 0.65)
        accept_b = (ocs_val == 1.0) and (mpqs is not None and mpqs >= 0.55)

        return ValidationResult(
            ocs=ocs_val, csr=csr, cps=cps, sfr=sfr, mpqs=mpqs,
            quality_tier=tier, accept_clean_a=accept_a, accept_clean_b=accept_b,
            regime=regime, sub_subtype=sub_subtype,
            keyword=keyword, best_match=best_match, match_rule=match_rule,
        )

    def __repr__(self) -> str:
        n = len(self.rubric)
        embedder_status = "with LaBSE" if self.embedder else "no embedder (CPS-only mode)"
        return f"Trivet(rubric_classes={n}, {embedder_status})"
