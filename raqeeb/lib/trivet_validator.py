"""
TIER1 VALIDATION IMPLEMENTATION - VERSION 3.3
==================================================

Version 3.3 Changes (Clitic-Aware Token Matching):
- TRV-021: Clitic-aware prefix stripping (ال، و، وال، ف، ب، ك، ل، لل)
- TRV-022: Inflection-aware suffix stripping (ة، ات، ين، ون، ان، تين)
- TRV-023: Match rule logging (ocs_match_rule diagnostic)
- TRV-024: Cascading match strategy (exact → prefix → suffix → both)

Previous Changes:
- v3.2: Arabic normalization (TRV-017 to TRV-020)
- v3.1: Token-aware OCS (TRV-013 to TRV-016)
- v3.0: Regime F/D, CPS, quality tiers (TRV-008 to TRV-012)

Key Fixes in v3.3:
- 'حالة' now matches 'الحالة' (definite article)
- 'حالة' now matches 'للحالة' (preposition + article)
- 'تابع' now matches 'تابعين' (plural inflection)
- 'خاص' now matches 'الخاصة' (article + feminine)

This is DETERMINISTIC clitic handling, not fuzzy matching.
All matches are logged with the rule that succeeded.

Author: Research Team
Date: 2026-01-24
Version: 3.3
"""

import pandas as pd
import numpy as np
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import Levenshtein


# ============================================================================
# TRV-008: REGIME MAPPING (SINGLE SOURCE OF TRUTH)
# ============================================================================

REGIME_F = {
    "Definiteness Shift",
    "Gender Disagreement",
    "Tanween Omission",
    "Spelling Error",
    "Register Mismatch",
    "Terminology Substitution",
    "Hypernym for Hyponym",
    "Hyponym for Hypernym",
    "Name Entity Error",
    "Total Omission",
    "Partial Translation",
    "Literal Translation",
    "Invalid Pattern",
    "Active to Passive Voice",
    "Passive to Active Voice",
    "Perfective to Progressive",
    "Progressive to Perfective",
    "Tense Shift Under Negation",
    "Adjective to Noun",
    "Noun to Adjective",
    "Wrong Word Order",
    "Wrong Structure",
}

REGIME_D = {
    "Meaning Shift",
}

STRUCTURE_TOLERANT_CLASSES = {
    "Wrong Word Order",
    "Wrong Structure",
    "Active to Passive Voice",
    "Passive to Active Voice",
    "Perfective to Progressive",
    "Progressive to Perfective",
    "Tense Shift Under Negation",
}

# TRV-012: CPS Floor Thresholds
CPS_FLOOR_DEFAULT = 0.55
CPS_FLOOR_STRUCTURE_TOLERANT = 0.45
CPS_FLOOR_CLEAN_A = 0.65


def get_regime(error_type: str) -> str:
    """Get validation regime for an error type."""
    if error_type in REGIME_D:
        return "D"
    elif error_type in REGIME_F:
        return "F"
    else:
        return "F"


def get_cps_floor(error_type: str) -> float:
    """Get CPS floor threshold for an error type."""
    if error_type in STRUCTURE_TOLERANT_CLASSES:
        return CPS_FLOOR_STRUCTURE_TOLERANT
    return CPS_FLOOR_DEFAULT


# ============================================================================
# TRV-017/018: ARABIC NORMALIZATION FOR OCS
# ============================================================================

_ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
_ALEF_VARIANTS = str.maketrans('أإآٱ', 'اااا')
_ALIF_MAQSURA = str.maketrans('ى', 'ي')
_TATWEEL = 'ـ'
_ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')


def normalize_arabic_ocs(s: str, strip_diacritics: bool = False) -> str:
    """
    Deterministic Arabic normalization for OCS matching.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(_TATWEEL, '')
    s = s.translate(_ALEF_VARIANTS)
    s = s.translate(_ALIF_MAQSURA)
    s = s.translate(_ARABIC_INDIC_DIGITS)
    s = re.sub(r'\s+', ' ', s).strip()
    if strip_diacritics:
        s = _ARABIC_DIACRITICS.sub('', s)
    return s


# ============================================================================
# TRV-021/022: CLITIC-AWARE TOKEN MATCHING
# ============================================================================

# Arabic prefixes (clitics + definite article)
# Order matters: longer prefixes first
ARABIC_PREFIXES = [
    'وبال', 'فبال', 'كال', 'وال', 'فال', 'بال', 'لل',  # compound
    'ال', 'و', 'ف', 'ب', 'ك', 'ل',  # simple
]

# Arabic suffixes (inflections)
# Order matters: longer suffixes first
ARABIC_SUFFIXES = [
    'تين', 'تان',  # dual feminine
    'ين', 'ون', 'ان', 'ات',  # plural
    'ية', 'ة', 'ه',  # feminine / ta marbuta
    'ي', 'ا',  # nisba / alef
]


def strip_arabic_prefix(token: str) -> Tuple[str, str]:
    """
    Strip Arabic prefix from token.
    Returns (stripped_token, prefix_found).
    """
    for prefix in ARABIC_PREFIXES:
        if token.startswith(prefix) and len(token) > len(prefix):
            return token[len(prefix):], prefix
    return token, ""


def strip_arabic_suffix(token: str) -> Tuple[str, str]:
    """
    Strip Arabic suffix from token.
    Returns (stripped_token, suffix_found).
    """
    for suffix in ARABIC_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[:-len(suffix)], suffix
    return token, ""


def clitic_aware_match(token: str, keyword: str) -> Tuple[bool, str]:
    """
    Check if token matches keyword with clitic-aware matching.
    
    Strategy (cascading):
    1. Exact match
    2. Strip prefix from token, match keyword
    3. Strip suffix from token, match keyword
    4. Strip both prefix and suffix from token, match keyword
    5. Strip prefix from keyword, match token (reverse)
    6. Strip suffix from keyword, match token (reverse)
    
    Returns:
        (matched: bool, match_rule: str)
        match_rule is one of: "exact", "prefix_X", "suffix_X", "both_X_Y", 
                              "kw_prefix_X", "kw_suffix_X", ""
    """
    # 1. Exact match
    if token == keyword:
        return True, "exact"
    
    # 2. Strip prefix from token
    token_no_prefix, prefix = strip_arabic_prefix(token)
    if prefix and token_no_prefix == keyword:
        return True, f"prefix_{prefix}"
    
    # 3. Strip suffix from token
    token_no_suffix, suffix = strip_arabic_suffix(token)
    if suffix and token_no_suffix == keyword:
        return True, f"suffix_{suffix}"
    
    # 4. Strip both prefix and suffix from token
    if prefix:
        token_no_both, suffix2 = strip_arabic_suffix(token_no_prefix)
        if suffix2 and token_no_both == keyword:
            return True, f"both_{prefix}_{suffix2}"
    
    # Also try suffix first, then prefix
    if suffix:
        token_no_both2, prefix2 = strip_arabic_prefix(token_no_suffix)
        if prefix2 and token_no_both2 == keyword:
            return True, f"both_{prefix2}_{suffix}"
    
    # 5. Strip prefix from keyword (reverse matching)
    kw_no_prefix, kw_prefix = strip_arabic_prefix(keyword)
    if kw_prefix and token == kw_no_prefix:
        return True, f"kw_prefix_{kw_prefix}"
    
    # 6. Strip suffix from keyword (reverse matching)
    kw_no_suffix, kw_suffix = strip_arabic_suffix(keyword)
    if kw_suffix and token == kw_no_suffix:
        return True, f"kw_suffix_{kw_suffix}"
    
    # No match
    return False, ""


# ============================================================================
# DATA STRUCTURES (Updated for v3.3)
# ============================================================================

@dataclass
class ValidationResult:
    """
    Result of validating a single candidate.
    
    v3.3 additions:
    - ocs_match_rule_kw: How keyword was matched (exact, prefix_ال, etc.)
    - ocs_match_rule_bm: How best_match was matched
    """
    passed: bool
    reject_reason: Optional[str]
    csr: float
    ocs: float
    sfr: float
    cps: float
    mpqs: float
    mpqs_mode: str
    regime: str
    quality_tier: Optional[str]
    flags: List[str]
    ocs_failed_constraints: List[str]
    cps_threshold: float = 0.0
    sfr_threshold_min: float = 0.0
    sfr_threshold_max: float = 1.0
    # v3.1 diagnostics
    kw_count_clean_tok: int = 0
    kw_count_error_tok: int = 0
    bm_count_error_tok: int = 0
    # v3.2 diagnostics
    ocs_norm_level: str = "light"
    kw_norm: str = ""
    bm_norm: str = ""
    ocs_rescued_by: str = ""
    # v3.3 diagnostics
    ocs_match_rule_kw: str = ""
    ocs_match_rule_bm: str = ""

    def to_dict(self):
        return {
            'passed': self.passed,
            'reject_reason': self.reject_reason,
            'csr': self.csr,
            'ocs': self.ocs,
            'sfr': self.sfr,
            'cps': self.cps,
            'mpqs': self.mpqs,
            'mpqs_mode': self.mpqs_mode,
            'regime': self.regime,
            'quality_tier': self.quality_tier,
            'flags': ','.join(self.flags) if self.flags else '',
            'ocs_failed_constraints': ';'.join(self.ocs_failed_constraints) if self.ocs_failed_constraints else '',
            'cps_threshold': self.cps_threshold,
            'sfr_threshold_min': self.sfr_threshold_min,
            'sfr_threshold_max': self.sfr_threshold_max,
            'kw_count_clean_tok': self.kw_count_clean_tok,
            'kw_count_error_tok': self.kw_count_error_tok,
            'bm_count_error_tok': self.bm_count_error_tok,
            'ocs_norm_level': self.ocs_norm_level,
            'kw_norm': self.kw_norm,
            'bm_norm': self.bm_norm,
            'ocs_rescued_by': self.ocs_rescued_by,
            'ocs_match_rule_kw': self.ocs_match_rule_kw,
            'ocs_match_rule_bm': self.ocs_match_rule_bm,
        }


# ============================================================================
# RUBRIC LOADING FROM CSV (SOURCE OF TRUTH)
# ============================================================================

def load_rubric_from_csv(csv_path: str = "threshold_reference_table_v2_OCS.csv") -> Dict:
    """Load RUBRIC_23 from CSV file."""
    csv_path = str(csv_path)
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Threshold CSV not found: {csv_path}\n"
            "OCS-native pipeline requires: threshold_reference_table_v2_OCS.csv"
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        "Error_Type", "Class", "CSR_Min", "OCS_Min", "SFR_Min", "SFR_Max",
        "Weight_CSR", "Weight_OCS", "Weight_SFR", "Inverted"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

    df["Error_Type"] = df["Error_Type"].astype(str).str.strip()
    df = df[df["Error_Type"].ne("")].copy()

    dup_vals = df["Error_Type"][df["Error_Type"].duplicated()].unique().tolist()
    if dup_vals:
        raise ValueError(f"Duplicate Error_Type entries: {dup_vals}")

    rubric: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        error_type = str(row["Error_Type"]).strip()
        inverted = bool(int(row["Inverted"])) if str(row["Inverted"]).strip() != "" else False

        rubric[error_type] = {
            "class": str(row["Class"]).strip(),
            "min_csr": float(row["CSR_Min"]),
            "min_ocs": float(row["OCS_Min"]),
            "min_sfr": float(row["SFR_Min"]),
            "max_sfr": float(row["SFR_Max"]),
            "weight_csr": float(row["Weight_CSR"]),
            "weight_ocs": float(row["Weight_OCS"]),
            "weight_sfr": float(row["Weight_SFR"]),
            "inverted": inverted,
            "regime": get_regime(error_type),
            "cps_floor": get_cps_floor(error_type),
        }

        if inverted:
            rubric[error_type]["max_csr"] = float(row["CSR_Min"])

    rubric["_rubric_source"] = csv_path
    rubric["_version"] = "3.3"

    error_keys = [k for k in rubric.keys() if not str(k).startswith("_")]
    if len(error_keys) != 23:
        raise ValueError(
            f"Expected 23 error classes, got {len(error_keys)}. "
            f"Keys={sorted(error_keys)}"
        )

    return rubric


def print_rubric_summary(rubric: dict) -> None:
    """Print a human-readable rubric summary."""
    print("\n" + "=" * 70)
    print("RUBRIC_23 LOADED FROM CSV (VERSION 3.3 - CLITIC-AWARE)")
    print("=" * 70)

    src = rubric.get("_rubric_source", "")
    version = rubric.get("_version", "unknown")
    print(f"Source: {src}")
    print(f"Version: {version}")

    rubric_items = {k: v for k, v in rubric.items() if isinstance(v, dict)}

    regime_f = {k: v for k, v in rubric_items.items() if v.get("regime") == "F"}
    regime_d = {k: v for k, v in rubric_items.items() if v.get("regime") == "D"}

    print(f"\n✓ Regime F (Fidelity): {len(regime_f)} classes")
    print(f"\n⚠ Regime D (Divergence): {len(regime_d)} classes")

    print(f"\n🔧 v3.3: Clitic-aware token matching")
    print(f"   Prefixes: {', '.join(ARABIC_PREFIXES[:6])}...")
    print(f"   Suffixes: {', '.join(ARABIC_SUFFIXES[:6])}...")


# ============================================================================
# TRV-023/024: CLITIC-AWARE OCS
# ============================================================================

_AR_PUNCT_SPLIT = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)


def _tokenize_ar(s: str) -> List[str]:
    """Tokenize Arabic text."""
    s = re.sub(r'\s+', ' ', (s or '').strip())
    toks = [t for t in _AR_PUNCT_SPLIT.split(s) if t]
    return toks


def _count_token_clitic_aware(tokens: List[str], keyword: str) -> Tuple[int, str]:
    """
    Count occurrences of keyword in tokens using clitic-aware matching.
    
    Returns:
        (count, best_match_rule)
    """
    if not keyword:
        return 0, ""
    
    kw_toks = _tokenize_ar(keyword)
    if not kw_toks:
        return 0, ""
    
    if len(kw_toks) == 1:
        # Single-token keyword
        kw = kw_toks[0]
        count = 0
        best_rule = ""
        for tok in tokens:
            matched, rule = clitic_aware_match(tok, kw)
            if matched:
                count += 1
                if not best_rule:
                    best_rule = rule
        return count, best_rule
    
    else:
        # Multi-token keyword - try exact sequence first
        n = len(kw_toks)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i+n] == kw_toks:
                return 1, "exact_seq"
        
        # Try clitic-aware on first/last token of sequence
        for i in range(len(tokens) - n + 1):
            # Check if first token matches with clitic stripping
            first_match, first_rule = clitic_aware_match(tokens[i], kw_toks[0])
            if first_match and tokens[i+1:i+n] == kw_toks[1:]:
                return 1, f"seq_{first_rule}"
            
            # Check if last token matches with clitic stripping
            last_match, last_rule = clitic_aware_match(tokens[i+n-1], kw_toks[-1])
            if last_match and tokens[i:i+n-1] == kw_toks[:-1]:
                return 1, f"seq_{last_rule}"
        
        return 0, ""


def calculate_ocs(
    clean_ar: str,
    error_ar: str,
    keyword: Optional[str],
    best_match: Optional[str],
    error_type: str
) -> Tuple[float, List[str], Dict]:
    """
    Option-2A Constraint Satisfaction (OCS) with Clitic-Aware Matching.
    
    v3.3 Changes:
    - Clitic-aware prefix stripping (ال، و، ف، ب، ك، ل، لل، etc.)
    - Inflection-aware suffix stripping (ة، ات، ين، ون، etc.)
    - Match rule logging for auditability
    - Cascading match strategy
    
    Returns:
        (ocs_score, failed_constraints, diagnostics)
    """
    failed: List[str] = []
    diagnostics = {
        'kw_count_clean_tok': 0,
        'kw_count_error_tok': 0,
        'bm_count_error_tok': 0,
        'ocs_norm_level': 'light',
        'kw_norm': '',
        'bm_norm': '',
        'ocs_rescued_by': '',
        'ocs_match_rule_kw': '',
        'ocs_match_rule_bm': '',
    }

    # --- Input validation ---
    if not isinstance(clean_ar, str) or not clean_ar.strip():
        failed.append("clean_ar_empty")
    if not isinstance(error_ar, str) or not error_ar.strip():
        failed.append("error_ar_empty")
    if failed:
        return 0.0, failed, diagnostics

    et = (error_type or "").strip()
    kw_raw = (keyword or "").strip()
    bm_raw = (best_match or "").strip()

    if not kw_raw:
        failed.append("keyword_missing")
    if not bm_raw:
        failed.append("best_match_missing")
    if failed:
        return 0.0, failed, diagnostics

    # --- Normalization (light pass) ---
    clean_norm = normalize_arabic_ocs(clean_ar, strip_diacritics=False)
    error_norm = normalize_arabic_ocs(error_ar, strip_diacritics=False)
    kw_norm = normalize_arabic_ocs(kw_raw, strip_diacritics=False)
    
    # Sentinel protection
    if bm_raw == "[OMITTED]":
        bm_norm = "[OMITTED]"
    else:
        bm_norm = normalize_arabic_ocs(bm_raw, strip_diacritics=False)

    diagnostics['kw_norm'] = kw_norm
    diagnostics['bm_norm'] = bm_norm

    # --- Tokenize ---
    clean_toks = _tokenize_ar(clean_norm)
    error_toks = _tokenize_ar(error_norm)

    # --- Count keyword in clean (clitic-aware) ---
    kw_clean, kw_match_rule = _count_token_clitic_aware(clean_toks, kw_norm)
    
    # --- Diacritics fallback for keyword ---
    rescued_by = []
    if kw_clean < 1:
        # Try with diacritics stripped
        clean_norm_diac = normalize_arabic_ocs(clean_ar, strip_diacritics=True)
        kw_norm_diac = normalize_arabic_ocs(kw_raw, strip_diacritics=True)
        clean_toks_diac = _tokenize_ar(clean_norm_diac)
        
        kw_clean_diac, kw_match_rule_diac = _count_token_clitic_aware(clean_toks_diac, kw_norm_diac)
        
        if kw_clean_diac >= 1:
            rescued_by.append("kw_diac")
            diagnostics['ocs_norm_level'] = 'diacritics'
            diagnostics['kw_norm'] = kw_norm_diac
            clean_toks = clean_toks_diac
            kw_clean = kw_clean_diac
            kw_match_rule = kw_match_rule_diac
            kw_norm = kw_norm_diac
            # Also update error tokens for consistency
            error_norm_diac = normalize_arabic_ocs(error_ar, strip_diacritics=True)
            error_toks = _tokenize_ar(error_norm_diac)

    diagnostics['kw_count_clean_tok'] = kw_clean
    diagnostics['ocs_match_rule_kw'] = kw_match_rule

    # --- Count keyword in error (clitic-aware) ---
    kw_err, _ = _count_token_clitic_aware(error_toks, kw_norm)
    diagnostics['kw_count_error_tok'] = kw_err

    # --- Keyword presence check ---
    if kw_clean < 1:
        failed.append(f"keyword_absent_in_clean_{kw_clean}")

    # --- Omission types handling ---
    omission_types = {"Total Omission", "Partial Translation"}

    if et in omission_types:
        if bm_norm != "[OMITTED]":
            failed.append("omission_best_match_not_OMITTED")
        diagnostics['bm_count_error_tok'] = 0

        if et == "Total Omission":
            if kw_err != 0:
                failed.append(f"keyword_present_in_error_{kw_err}")
        else:  # Partial Translation
            if kw_clean >= 1 and kw_err >= kw_clean:
                failed.append(f"keyword_not_reduced_in_error_clean{kw_clean}_err{kw_err}")

    else:
        # --- Non-omission substitution classes ---
        if kw_clean == 1 and kw_err != 0:
            failed.append(f"keyword_present_in_error_{kw_err}")
        if kw_clean > 1 and kw_err >= kw_clean:
            failed.append(f"keyword_not_reduced_in_error_clean{kw_clean}_err{kw_err}")

        if bm_norm == "[OMITTED]":
            failed.append("best_match_OMITTED_for_non_omission")
            diagnostics['bm_count_error_tok'] = 0
        else:
            # Count best_match in error (clitic-aware)
            bm_err, bm_match_rule = _count_token_clitic_aware(error_toks, bm_norm)
            
            # Diacritics fallback for best_match
            if bm_err < 1:
                if diagnostics['ocs_norm_level'] == 'light':
                    error_norm_diac = normalize_arabic_ocs(error_ar, strip_diacritics=True)
                    error_toks_diac = _tokenize_ar(error_norm_diac)
                else:
                    error_toks_diac = error_toks
                
                bm_norm_diac = normalize_arabic_ocs(bm_raw, strip_diacritics=True)
                bm_err_diac, bm_match_rule_diac = _count_token_clitic_aware(error_toks_diac, bm_norm_diac)
                
                if bm_err_diac >= 1:
                    rescued_by.append("bm_diac")
                    diagnostics['ocs_norm_level'] = 'diacritics'
                    diagnostics['bm_norm'] = bm_norm_diac
                    bm_err = bm_err_diac
                    bm_match_rule = bm_match_rule_diac
                else:
                    failed.append(f"best_match_absent_in_error_{bm_err}")
            
            diagnostics['bm_count_error_tok'] = bm_err
            diagnostics['ocs_match_rule_bm'] = bm_match_rule

    # --- Set rescued_by diagnostic ---
    if len(rescued_by) == 2:
        diagnostics['ocs_rescued_by'] = 'both'
    elif len(rescued_by) == 1:
        diagnostics['ocs_rescued_by'] = rescued_by[0]

    return (1.0 if not failed else 0.0), failed, diagnostics


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_csr(text1: str, text2: str) -> Optional[float]:
    """Calculate Character Similarity Ratio using Levenshtein distance."""
    try:
        if not text1 or not text2:
            return None
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        csr = 1.0 - (distance / max_len)
        return max(0.0, min(1.0, csr))
    except Exception as e:
        print(f"⚠ CSR calculation failed: {e}")
        return None


def calculate_cps(
    clean_ar: str,
    error_ar: str,
    keyword: Optional[str],
    best_match: Optional[str]
) -> Optional[float]:
    """Calculate Context Preservation Score (CPS)."""
    try:
        if not clean_ar or not error_ar:
            return None
        kw = (keyword or "").strip()
        bm = (best_match or "").strip()
        if kw:
            context_clean = clean_ar.replace(kw, "", 1)
        else:
            context_clean = clean_ar
        if bm and bm != "[OMITTED]":
            context_error = error_ar.replace(bm, "", 1)
        else:
            context_error = error_ar
        if not context_clean.strip() or not context_error.strip():
            return 0.90
        return calculate_csr(context_clean, context_error)
    except Exception as e:
        print(f"⚠ CPS calculation failed: {e}")
        return None


def calculate_sfr(text1: str, text2: str, model=None) -> Optional[float]:
    """Calculate Semantic Fluency Ratio using LaBSE embeddings."""
    try:
        if not text1 or not text2:
            return None
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        if model is None:
            model = SentenceTransformer('sentence-transformers/LaBSE')
        emb1 = model.encode([text1])
        emb2 = model.encode([text2])
        sfr = cosine_similarity(emb1, emb2)[0][0]
        return max(0.0, min(1.0, float(sfr)))
    except Exception as e:
        print(f"⚠ SFR calculation failed: {e}")
        return None


def calculate_mpqs(csr: float, ocs: float, sfr: float, 
                   weights: Tuple[float, float, float]) -> Optional[float]:
    """Calculate MPQS as weighted harmonic mean."""
    try:
        if csr is None or ocs is None or sfr is None:
            return None
        if csr <= 0 or ocs <= 0 or sfr <= 0:
            return 0.0
        alpha, beta, gamma = weights
        mpqs = 3.0 / ((alpha / csr) + (beta / ocs) + (gamma / sfr))
        return max(0.0, min(1.0, mpqs))
    except Exception as e:
        print(f"⚠ MPQS calculation failed: {e}")
        return None


# ============================================================================
# REGIME-AWARE VALIDATION
# ============================================================================

def validate_metrics_against_rubric(
    csr: Optional[float],
    ocs: Optional[float],
    sfr: Optional[float],
    cps: Optional[float],
    error_type: str,
    rubric: Dict
) -> Tuple[bool, Optional[str], Dict]:
    """Validate metrics against error-type-specific thresholds."""
    details = {}
    
    if error_type not in rubric:
        return False, "unknown_error_type", details
    
    t = rubric[error_type]
    regime = t.get("regime", get_regime(error_type))
    details["regime"] = regime
    
    if csr is None or ocs is None or sfr is None or cps is None:
        return False, "missing_metric", details
    
    if not (0 <= csr <= 1) or not (0 <= ocs <= 1) or not (0 <= sfr <= 1) or not (0 <= cps <= 1):
        return False, "metric_out_of_range", details
    
    if math.isnan(csr) or math.isnan(ocs) or math.isnan(sfr) or math.isnan(cps):
        return False, "metric_is_nan", details

    if ocs < 1.0:
        return False, "ocs_failed", details
    
    if regime == "F":
        cps_floor = t.get("cps_floor", CPS_FLOOR_DEFAULT)
        details["cps_threshold"] = cps_floor
        if cps < cps_floor:
            return False, "cps_below_floor", details
        min_sfr = t.get("min_sfr", 0.40)
        details["sfr_threshold_min"] = min_sfr
        if sfr < min_sfr:
            return False, "sfr_below_floor", details
        return True, None, details
    
    elif regime == "D":
        max_csr = t.get("max_csr", 1.0)
        details["csr_threshold"] = max_csr
        if max_csr < 1.0 and csr > max_csr:
            return False, "csr_exceeds_maximum", details
        min_sfr = t.get("min_sfr", 0.30)
        max_sfr = t.get("max_sfr", 0.80)
        details["sfr_threshold_min"] = min_sfr
        details["sfr_threshold_max"] = max_sfr
        if sfr < min_sfr:
            return False, "sfr_below_minimum", details
        if sfr > max_sfr:
            return False, "sfr_above_maximum", details
        return True, None, details
    
    return True, None, details


def classify_quality_tier(cps: float, sfr: float, min_sfr: float, regime: str) -> Optional[str]:
    """Classify a passing sample into quality tiers."""
    if regime == "F":
        if cps >= CPS_FLOOR_CLEAN_A:
            return "A"
        elif cps >= CPS_FLOOR_DEFAULT:
            return "B"
        return None
    elif regime == "D":
        return "B"
    return None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_tier1_candidate(
    row: pd.Series,
    rubric: Dict,
    tier: str = "TIER1",
    track: str = "clean",
    labse_model=None
) -> ValidationResult:
    """Validate a single Tier1 candidate with v3.3 clitic-aware matching."""
    flags = []
    reject_reason = None
    
    clean_ar = row.get('ar', row.get('clean_ar', ''))
    error_ar = row.get('mt_output', row.get('error_ar', ''))
    error_type = row.get('Sub-Subtype', row.get('error_type', ''))
    keyword = row.get('Keyword', row.get('keyword', None))
    best_match = row.get('Best_Match', row.get('best_match', None))
    
    regime = get_regime(error_type)
    
    default_result = lambda reason, fatal=False: ValidationResult(
        passed=False,
        reject_reason=reason,
        csr=0.0, ocs=0.0, sfr=0.0, cps=0.0, mpqs=0.0,
        mpqs_mode="diagnostic",
        regime=regime,
        quality_tier=None,
        flags=["fatal_error"] if fatal else [],
        ocs_failed_constraints=[reason] if fatal else [],
        cps_threshold=get_cps_floor(error_type),
        kw_count_clean_tok=0, kw_count_error_tok=0, bm_count_error_tok=0,
        ocs_norm_level='light', kw_norm='', bm_norm='', ocs_rescued_by='',
        ocs_match_rule_kw='', ocs_match_rule_bm='',
    )
    
    if not clean_ar or not error_ar:
        return default_result("empty_field", fatal=True)
    
    if clean_ar.strip() == error_ar.strip():
        return default_result("identical_sentences", fatal=True)
    
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in error_ar)
    if not has_arabic:
        return default_result("non_arabic", fatal=True)
    
    fidelity_classes = {
        "Tanween Omission", "Spelling Error", "Gender Disagreement",
        "Definiteness Shift", "Register Mismatch", "Active to Passive Voice",
        "Passive to Active Voice", "Perfective to Progressive",
        "Progressive to Perfective", "Tense Shift Under Negation",
        "Wrong Structure", "Wrong Word Order", "Noun to Adjective",
        "Adjective to Noun"
    }
    mpqs_mode = "score" if error_type in fidelity_classes else "diagnostic"

    # Calculate metrics
    csr = calculate_csr(clean_ar, error_ar)
    ocs, ocs_failed, ocs_diag = calculate_ocs(clean_ar, error_ar, keyword, best_match, error_type)
    sfr = calculate_sfr(clean_ar, error_ar, model=labse_model)
    cps = calculate_cps(clean_ar, error_ar, keyword, best_match)
    
    if csr is None or sfr is None or cps is None:
        return ValidationResult(
            passed=False,
            reject_reason="metric_calculation_failed",
            csr=csr or 0.0, ocs=ocs or 0.0, sfr=sfr or 0.0, cps=cps or 0.0,
            mpqs=0.0, mpqs_mode="diagnostic",
            regime=regime, quality_tier=None,
            flags=["metric_error"],
            ocs_failed_constraints=ocs_failed,
            cps_threshold=get_cps_floor(error_type),
            kw_count_clean_tok=ocs_diag.get('kw_count_clean_tok', 0),
            kw_count_error_tok=ocs_diag.get('kw_count_error_tok', 0),
            bm_count_error_tok=ocs_diag.get('bm_count_error_tok', 0),
            ocs_norm_level=ocs_diag.get('ocs_norm_level', 'light'),
            kw_norm=ocs_diag.get('kw_norm', ''),
            bm_norm=ocs_diag.get('bm_norm', ''),
            ocs_rescued_by=ocs_diag.get('ocs_rescued_by', ''),
            ocs_match_rule_kw=ocs_diag.get('ocs_match_rule_kw', ''),
            ocs_match_rule_bm=ocs_diag.get('ocs_match_rule_bm', ''),
        )
    
    if error_type in rubric:
        weights = (
            rubric[error_type]['weight_csr'],
            rubric[error_type]['weight_ocs'],
            rubric[error_type]['weight_sfr']
        )
    else:
        weights = (0.33, 0.34, 0.33)
    
    mpqs = calculate_mpqs(csr, ocs, sfr, weights)
    if mpqs is None:
        mpqs = 0.0
        flags.append("mpqs_calculation_failed")
    
    t = rubric.get(error_type, {})
    cps_threshold = t.get("cps_floor", get_cps_floor(error_type))
    sfr_threshold_min = t.get("min_sfr", 0.40)
    sfr_threshold_max = t.get("max_sfr", 1.0)
    
    quality_tier = None
    
    if track == "clean":
        passed, reason, details = validate_metrics_against_rubric(
            csr, ocs, sfr, cps, error_type, rubric
        )
        if not passed:
            reject_reason = reason
        else:
            reject_reason = None
            quality_tier = classify_quality_tier(cps, sfr, sfr_threshold_min, regime)
            if mpqs < 0.70:
                flags.append("low_mpqs")
    
    elif track == "noisy":
        passed = True
        reject_reason = None
        if error_type in rubric:
            violation_passed, violation_reason, _ = validate_metrics_against_rubric(
                csr, ocs, sfr, cps, error_type, rubric
            )
            if not violation_passed:
                flags.append("violates_clean_standard")
                reject_reason = violation_reason
        if mpqs < 0.50:
            flags.append("low_quality_suspected")
        if mpqs < 0.40:
            flags.append("multi_error_suspected")
    
    else:
        raise ValueError(f"Unknown track: {track}")
    
    return ValidationResult(
        passed=passed if track == "clean" else True,
        reject_reason=reject_reason,
        csr=csr, ocs=ocs, sfr=sfr, cps=cps, mpqs=mpqs,
        mpqs_mode=mpqs_mode,
        regime=regime,
        quality_tier=quality_tier,
        flags=flags,
        ocs_failed_constraints=ocs_failed,
        cps_threshold=cps_threshold,
        sfr_threshold_min=sfr_threshold_min,
        sfr_threshold_max=sfr_threshold_max,
        kw_count_clean_tok=ocs_diag.get('kw_count_clean_tok', 0),
        kw_count_error_tok=ocs_diag.get('kw_count_error_tok', 0),
        bm_count_error_tok=ocs_diag.get('bm_count_error_tok', 0),
        ocs_norm_level=ocs_diag.get('ocs_norm_level', 'light'),
        kw_norm=ocs_diag.get('kw_norm', ''),
        bm_norm=ocs_diag.get('bm_norm', ''),
        ocs_rescued_by=ocs_diag.get('ocs_rescued_by', ''),
        ocs_match_rule_kw=ocs_diag.get('ocs_match_rule_kw', ''),
        ocs_match_rule_bm=ocs_diag.get('ocs_match_rule_bm', ''),
    )


# ============================================================================
# SEQUENTIAL VALIDATION - v3.3
# ============================================================================

def run_full_tier1_validation(
    df_tier1: pd.DataFrame,
    rubric: Dict,
    labse_model=None,
    show_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """Run full Tier1 validation with v3.3 clitic-aware matching."""
    print("\n" + "="*70)
    print("TIER1 FULL VALIDATION v3.3 - CLITIC-AWARE MATCHING")
    print("="*70)
    print(f"Total candidates: {len(df_tier1):,}")
    print(f"Regime F classes: {len(REGIME_F)}")
    print(f"Regime D classes: {len(REGIME_D)}")
    print(f"v3.3: Clitic prefixes ({len(ARABIC_PREFIXES)}) + suffixes ({len(ARABIC_SUFFIXES)})")
    
    results = {
        'clean': [], 'clean_a': [], 'clean_b': [],
        'noisy': [], 'discarded': []
    }

    if labse_model is None:
        raise ValueError("labse_model must be provided.")

    metric_failures = defaultdict(int)
    regime_counts = {"F": {"clean": 0, "noisy": 0}, "D": {"clean": 0, "noisy": 0}}
    tier_counts = {"A": 0, "B": 0}
    rescue_counts = {"kw_diac": 0, "bm_diac": 0, "both": 0}
    match_rule_counts = defaultdict(int)
    
    iterator = tqdm(df_tier1.iterrows(), total=len(df_tier1), desc="Validating") if show_progress else df_tier1.iterrows()
    
    for idx, row in iterator:
        result = validate_tier1_candidate(row, rubric, tier="TIER1", track="clean", labse_model=labse_model)
        
        row_dict = row.to_dict()
        row_dict.update(result.to_dict())
        
        # Track rescues and match rules
        if result.ocs_rescued_by:
            rescue_counts[result.ocs_rescued_by] = rescue_counts.get(result.ocs_rescued_by, 0) + 1
        if result.ocs_match_rule_kw:
            match_rule_counts[f"kw_{result.ocs_match_rule_kw}"] += 1
        if result.ocs_match_rule_bm:
            match_rule_counts[f"bm_{result.ocs_match_rule_bm}"] += 1
        
        if result.passed:
            results['clean'].append(row_dict)
            regime_counts[result.regime]["clean"] += 1
            if result.quality_tier == "A":
                results['clean_a'].append(row_dict)
                tier_counts["A"] += 1
            elif result.quality_tier == "B":
                results['clean_b'].append(row_dict)
                tier_counts["B"] += 1
        elif result.reject_reason in ['empty_field', 'identical_sentences', 'non_arabic', 'metric_calculation_failed']:
            results['discarded'].append(row_dict)
            metric_failures[result.reject_reason] += 1
        else:
            noisy_result = validate_tier1_candidate(row, rubric, tier="TIER1", track="noisy", labse_model=labse_model)
            row_dict.update(noisy_result.to_dict())
            results['noisy'].append(row_dict)
            regime_counts[result.regime]["noisy"] += 1
    
    # Convert to DataFrames
    clean_df = pd.DataFrame(results['clean'])
    clean_a_df = pd.DataFrame(results['clean_a'])
    clean_b_df = pd.DataFrame(results['clean_b'])
    noisy_df = pd.DataFrame(results['noisy'])
    discarded_df = pd.DataFrame(results['discarded'])
    
    total = len(df_tier1)
    stats = {
        'total_candidates': total,
        'clean': len(clean_df),
        'clean_a': len(clean_a_df),
        'clean_b': len(clean_b_df),
        'noisy': len(noisy_df),
        'discarded': len(discarded_df),
        'clean_rate': len(clean_df) / total if total > 0 else 0,
        'clean_a_rate': len(clean_a_df) / total if total > 0 else 0,
        'clean_b_rate': len(clean_b_df) / total if total > 0 else 0,
        'noisy_rate': len(noisy_df) / total if total > 0 else 0,
        'discard_rate': len(discarded_df) / total if total > 0 else 0,
        'metric_failures': dict(metric_failures),
        'regime_counts': regime_counts,
        'tier_counts': tier_counts,
        'rescue_counts': rescue_counts,
        'match_rule_counts': dict(match_rule_counts),
    }
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY (v3.3)")
    print("="*70)
    print(f"\nTotal Candidates: {stats['total_candidates']:,}")
    print(f"\nResults:")
    print(f"  ✓ Clean:     {stats['clean']:,} ({stats['clean_rate']:.1%})")
    print(f"    ├─ Clean_A: {stats['clean_a']:,} ({stats['clean_a_rate']:.1%})")
    print(f"    └─ Clean_B: {stats['clean_b']:,} ({stats['clean_b_rate']:.1%})")
    print(f"  ⚠ Noisy:     {stats['noisy']:,} ({stats['noisy_rate']:.1%})")
    print(f"  ✗ Discarded: {stats['discarded']:,} ({stats['discard_rate']:.1%})")
    
    print(f"\nBy Regime:")
    for regime in ["F", "D"]:
        clean_r = regime_counts[regime]["clean"]
        noisy_r = regime_counts[regime]["noisy"]
        total_r = clean_r + noisy_r
        rate = clean_r / total_r if total_r > 0 else 0
        print(f"  Regime {regime}: {clean_r}/{total_r} clean ({rate:.1%})")
    
    print(f"\nClitic Match Rules (v3.3):")
    for rule, count in sorted(match_rule_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rule}: {count:,}")
    
    if rescue_counts:
        total_rescued = sum(rescue_counts.values())
        if total_rescued > 0:
            print(f"\nDiacritics Rescue: {total_rescued:,} rows")
    
    if len(noisy_df) > 0 and 'reject_reason' in noisy_df.columns:
        print(f"\nTop Rejection Reasons (Noisy Track):")
        top_reasons = noisy_df['reject_reason'].value_counts().head(10)
        for reason, count in top_reasons.items():
            print(f"  {reason}: {count:,} ({count/len(noisy_df):.1%})")
    
    if len(clean_df) > 0:
        print(f"\nClean Rate by Error Type:")
        error_col = 'Sub-Subtype' if 'Sub-Subtype' in df_tier1.columns else 'error_type'
        for et in sorted(df_tier1[error_col].unique()):
            total_et = len(df_tier1[df_tier1[error_col] == et])
            clean_et = len(clean_df[clean_df[error_col] == et]) if error_col in clean_df.columns else 0
            rate = clean_et / total_et if total_et > 0 else 0
            regime = get_regime(et)
            status = "✓" if rate >= 0.10 else "⚠" if rate > 0 else "✗"
            print(f"  {status} [{regime}] {et}: {clean_et}/{total_et} ({rate:.1%})")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'clean': clean_df,
        'clean_a': clean_a_df,
        'clean_b': clean_b_df,
        'noisy': noisy_df,
        'discarded': discarded_df,
        'stats': stats
    }


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Tier1 Validation Implementation v3.3")
    print("="*50)
    print("\nv3.3 Changes:")
    print("  - TRV-021: Clitic-aware prefix stripping")
    print("  - TRV-022: Inflection-aware suffix stripping")
    print("  - TRV-023: Match rule logging")
    print("  - TRV-024: Cascading match strategy")
    print(f"\nPrefixes: {ARABIC_PREFIXES}")
    print(f"Suffixes: {ARABIC_SUFFIXES}")
