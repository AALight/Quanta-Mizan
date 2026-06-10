"""Test the public import API exposed at raqeeb.* and raqeeb.<submodule>.*"""
from __future__ import annotations


def test_top_level_imports():
    """from raqeeb import ... resolves the four documented entry points."""
    import raqeeb
    from raqeeb import Trivet, RaqeebClassifier, ETCAAuditor
    from raqeeb import calculate_mpqs, clitic_aware_match

    # __version__ exposed
    assert hasattr(raqeeb, "__version__")
    assert raqeeb.__version__ == "1.0.0"

    # Class types are usable
    assert isinstance(Trivet, type)
    assert isinstance(RaqeebClassifier, type)
    assert isinstance(ETCAAuditor, type)

    # Functions callable
    assert callable(calculate_mpqs)
    assert callable(clitic_aware_match)


def test_trivet_submodule_imports():
    """raqeeb.trivet exposes the wrapper + cascade primitives."""
    from raqeeb.trivet import (
        Trivet, ValidationResult,
        clitic_aware_match, normalize_arabic_ocs,
        calculate_ocs, calculate_csr, calculate_cps, calculate_sfr,
        calculate_mpqs, classify_quality_tier, get_regime,
    )
    # All callable / class
    for x in (Trivet, ValidationResult,
               clitic_aware_match, normalize_arabic_ocs,
               calculate_ocs, calculate_csr, calculate_cps,
               calculate_sfr, calculate_mpqs, classify_quality_tier,
               get_regime):
        assert callable(x)


def test_classifier_submodule_imports():
    from raqeeb.classifier import RaqeebClassifier
    assert isinstance(RaqeebClassifier, type)


def test_etca_submodule_imports():
    from raqeeb.etca import ETCAAuditor
    assert isinstance(ETCAAuditor, type)


def test_trivet_wrapper_match():
    """Trivet().match() correctly handles literal + clitic-prefix matching."""
    from raqeeb.trivet import Trivet
    t = Trivet()
    # literal containment
    assert t.match("الحالة", "نعرض الحالة الراهنة")
    # clitic prefix (الـ): keyword "حالة" should match token "الحالة"
    assert t.match("حالة", "نعرض الحالة الراهنة")
    # negative — keyword absent
    assert not t.match("غير_موجود", "نص آخر تماما")
    # empty inputs are False, not crashes
    assert not t.match("", "نص")
    assert not t.match("كلمة", "")


def test_trivet_wrapper_validate_returns_dataclass():
    """Trivet().validate() returns a populated ValidationResult."""
    from raqeeb.trivet import Trivet, ValidationResult
    t = Trivet()
    r = t.validate(
        keyword="السلام",
        best_match="الأمن",
        ar_ref="مبدأ السلام في المنطقة",
        mt_output="مبدأ الأمن في المنطقة",
        sub_subtype="Terminology Substitution",
    )
    assert isinstance(r, ValidationResult)
    assert r.regime in ("F", "D")
    assert isinstance(r.ocs, float)
    assert r.keyword == "السلام"
    assert r.best_match == "الأمن"
    # Round-trip dict serialisation
    assert isinstance(r.to_dict(), dict)


# ---------------------------------------------------------------------------
# Data loaders + version (the public toolkit API for Mizan / QUANTA).
# ---------------------------------------------------------------------------
def test_data_mizan_and_quanta():
    from raqeeb.data import mizan, load_quanta
    assert mizan["n_classes"] == 23
    assert len(mizan["label_map"]) == 23
    # severity present for a known class (bundled canonical_mizan_v7.json)
    assert mizan["classes"]["Total Omission"]["severity_score"] in (1, 2, 3, 4, 5)
    df = load_quanta()
    assert hasattr(df, "shape") and df.shape[0] > 0


def test_version_is_1_0_0():
    import raqeeb
    assert raqeeb.__version__ == "1.0.0"
