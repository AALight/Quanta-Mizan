# =============================================================================
# CELL 15.2: RAQEEB EVALUATION FILE GENERATOR
# =============================================================================
"""
CELL 15.2: Generate Human Evaluation Workbooks for TRIVET/Raqeeb Framework

PURPOSE:
Create all annotation workbooks for the two-layer Raqeeb evaluation:
  Layer 1 — Blind sentence-level MQM assessment
  Layer 2 — Keyword-level validation (system labels revealed)

INPUTS (from earlier TRIVET cells — NOT hardcoded):
  - tier1_v3_3_final/tier1_clean_A_for_etca.csv   (Cell 16 output)
  - tier1_v3_3_final/tier1_clean_B_FROZEN.csv      (Cell 14 output)
  - tier1_v3_3_final/tier2_generated_raw.csv        (Cell 17A output, 8,044 rows)
  - tier1_v3_3_final/tier2_seed_audit_cell15.csv    (Cell 18 ETCA scores for A)
  - TAXONOMY_V3 from Cell 17B (23 classes)

OUTPUTS (per tier, per phase):
  Sampling CSVs:
    - TIER1_HUMANEVAL_MERGED.csv
    - TIER2_HUMANEVAL_STRATIFIED.csv
  Workbooks (×2 tiers):
    - Training Practice (T1/T2)
    - Phase 1 Gold Annotator A/B (T1/T2)
    - Phase 3 Batch Annotator A/B (T1/T2)
    - Phase 4 Review Annotator A/B (T1/T2)
    - IAA Report (T1/T2)
    - Phase 5-6 Adjudication (T1/T2)
    - Combined Results (master, both tiers)
    - sampling_metadata.json

DESIGN DECISIONS:
  - Separate workbooks per tier (different provenance: real MT vs synthetic)
  - Identical schema across all files
  - Tier column in every file for programmatic merging
  - Taxonomy from Cell 17B TAXONOMY_V3 (not hardcoded classes)
"""

import pandas as pd
import numpy as np
import json
import os
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy

try:
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                  numbers, Protection, NamedStyle)
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("⚠️ openpyxl not installed. Run: pip install openpyxl")

print("=" * 80)
print("CELL 15.2: RAQEEB EVALUATION FILE GENERATOR v1.0")
print("=" * 80)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

OUTPUT_DIR = "tier1_v3_3_final"
EVAL_DIR = f"{OUTPUT_DIR}/human_evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

# Input paths (from earlier cells)
TIER1_A_PATH = f"{OUTPUT_DIR}/tier1_clean_A_for_etca.csv"
TIER1_B_PATH = f"{OUTPUT_DIR}/tier1_clean_B_FROZEN.csv"
TIER2_PATH = f"{OUTPUT_DIR}/tier2_generated_raw.csv"
ETCA_AUDIT_PATH = f"{OUTPUT_DIR}/tier2_seed_audit_cell15.csv"

# Fallback paths
TIER1_A_FALLBACKS = [f"{OUTPUT_DIR}/tier1_clean_A_FROZEN.csv", f"{OUTPUT_DIR}/tier1_clean_A.csv"]
TIER1_B_FALLBACKS = [f"{OUTPUT_DIR}/tier1_clean_B.csv"]
TIER2_FALLBACKS = [f"{OUTPUT_DIR}/tier2_generated_enriched.csv", "tier2_generated_raw.csv"]

# Sampling configuration
RANDOM_SEED = 42
T1_BASE_RATE = 0.10          # 10% base sampling for Tier 1
T1_MIN_PER_CLASS = 5         # Minimum per class (rare class oversampling)
T1_EXHAUST_THRESHOLD = 10    # Classes with ≤ this many in pool: take ALL available
T2_TARGET = 425              # Fixed target for Tier 2
T2_MIN_PER_CLASS = 4         # Minimum per class for Tier 2
GOLD_PER_CLASS_T1 = 4        # Gold overlap: 4 per class for Tier 1
GOLD_PER_CLASS_T2 = 4        # Gold overlap: 4 per class for Tier 2
TRAINING_PER_TIER = 5        # Training samples per tier

# Error type column (consistent with pipeline)
ERROR_COL = 'Sub-Subtype'

# =============================================================================
# 2. TAXONOMY REFERENCE (from Cell 17B — DO NOT MODIFY)
# =============================================================================

TAXONOMY_V3 = {
    "Meaning Shift": {
        "Class": "Semantic Divergence", "Subtype": "Semantic Shift",
        "Definition": "The translation conveys an incorrect or distorted meaning compared to the source.",
        "Severity_Level": "Critical", "Severity_Score": 5,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.30, "Category_Weight": 0.40,
    },
    "Total Omission": {
        "Class": "Semantic Divergence", "Subtype": "Translation Gap",
        "Definition": "A source text element is completely missing from the translation.",
        "Severity_Level": "Critical", "Severity_Score": 5,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.30, "Category_Weight": 0.40,
    },
    "Partial Translation": {
        "Class": "Semantic Divergence", "Subtype": "Translation Gap",
        "Definition": "A named entity or key term is only partially translated.",
        "Severity_Level": "Severe", "Severity_Score": 4,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.20, "Category_Weight": 0.40,
    },
    "Name Entity Error": {
        "Class": "Semantic Divergence", "Subtype": "Translation Gap",
        "Definition": "Incorrect, distorted, or substituted rendering of a proper name.",
        "Severity_Level": "Severe", "Severity_Score": 4,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.20, "Category_Weight": 0.40,
    },
    "Literal Translation": {
        "Class": "Semantic Divergence", "Subtype": "Idiomatic Shift",
        "Definition": "An idiom is translated word-for-word, losing its figurative meaning.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.10, "Category_Weight": 0.40,
    },
    "Hypernym for Hyponym": {
        "Class": "Semantic Divergence", "Subtype": "Granularity Shift",
        "Definition": "Using a general term for a specific one.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.05, "Category_Weight": 0.40,
    },
    "Hyponym for Hypernym": {
        "Class": "Semantic Divergence", "Subtype": "Granularity Shift",
        "Definition": "Using a specific term for a general one.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.05, "Category_Weight": 0.40,
    },
    "Invalid Pattern": {
        "Class": "Morphosyntactic Shifts", "Subtype": "Morphological Error",
        "Definition": "Use of an incorrect form of a word.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Fluency", "Weight_WGT": 0.30, "Category_Weight": 0.10,
    },
    "Tanween Omission": {
        "Class": "Morphosyntactic Shifts", "Subtype": "Morphological Error",
        "Definition": "Omission of the Arabic nunation where required.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Fluency", "Weight_WGT": 0.20, "Category_Weight": 0.10,
    },
    "Gender Disagreement": {
        "Class": "Morphosyntactic Shifts", "Subtype": "Agreement Error",
        "Definition": "Mismatch in grammatical gender.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Fluency", "Weight_WGT": 0.30, "Category_Weight": 0.10,
    },
    "Definiteness Shift": {
        "Class": "Morphosyntactic Shifts", "Subtype": "Definiteness Shift",
        "Definition": "Incorrect use of the definite article 'al-' or its absence.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Fluency", "Weight_WGT": 0.20, "Category_Weight": 0.10,
    },
    "Perfective to Progressive": {
        "Class": "Syntactic Shift", "Subtype": "Aspect Shift",
        "Definition": "Completed to Ongoing event.",
        "Severity_Level": "Severe", "Severity_Score": 4,
        "TQA_Category": "Fluency", "Weight_WGT": 0.15, "Category_Weight": 0.10,
    },
    "Progressive to Perfective": {
        "Class": "Syntactic Shift", "Subtype": "Aspect Shift",
        "Definition": "Ongoing event to Completed.",
        "Severity_Level": "Severe", "Severity_Score": 4,
        "TQA_Category": "Fluency", "Weight_WGT": 0.15, "Category_Weight": 0.10,
    },
    "Tense Shift Under Negation": {
        "Class": "Syntactic Shift", "Subtype": "Tense Shift",
        "Definition": "Tense shift under negation.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Accuracy", "Weight_WGT": 0.20, "Category_Weight": 0.10,
    },
    "Wrong Structure": {
        "Class": "Syntactic Shift", "Subtype": "Structural Error",
        "Definition": "Use of a grammatically incorrect sentence construction.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Fluency", "Weight_WGT": 0.20, "Category_Weight": 0.10,
    },
    "Wrong Word Order": {
        "Class": "Syntactic Shift", "Subtype": "Structural Error",
        "Definition": "Words are arranged in an incorrect or unnatural order.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Fluency", "Weight_WGT": 0.20, "Category_Weight": 0.10,
    },
    "Noun to Adjective": {
        "Class": "Syntactic Shift", "Subtype": "Structural Error",
        "Definition": "A noun is translated as an adjective.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Fluency", "Weight_WGT": 0.025, "Category_Weight": 0.10,
    },
    "Adjective to Noun": {
        "Class": "Syntactic Shift", "Subtype": "Structural Error",
        "Definition": "An adjective is translated as a noun.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Fluency", "Weight_WGT": 0.025, "Category_Weight": 0.10,
    },
    "Active to Passive Voice": {
        "Class": "Syntactic Shift", "Subtype": "Voice Shift",
        "Definition": "A change in voice that may alter focus or style.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Style & Register", "Weight_WGT": 0.025, "Category_Weight": 0.10,
    },
    "Passive to Active Voice": {
        "Class": "Syntactic Shift", "Subtype": "Voice Shift",
        "Definition": "A change in voice that may alter focus or style.",
        "Severity_Level": "Moderate", "Severity_Score": 2,
        "TQA_Category": "Style & Register", "Weight_WGT": 0.025, "Category_Weight": 0.10,
    },
    "Register Mismatch": {
        "Class": "Pragmatic Divergence", "Subtype": "Register Mismatch",
        "Definition": "Using an informal expression in a formal text, or vice-versa.",
        "Severity_Level": "Major", "Severity_Score": 3,
        "TQA_Category": "Style & Register", "Weight_WGT": 0.20, "Category_Weight": 0.35,
    },
    "Terminology Substitution": {
        "Class": "Pragmatic Divergence", "Subtype": "Register Mismatch",
        "Definition": "Wrong domain-specific term.",
        "Severity_Level": "Severe", "Severity_Score": 4,
        "TQA_Category": "Terminology", "Weight_WGT": 0.80, "Category_Weight": 0.35,
    },
    "Spelling Error": {
        "Class": "Orthographic Shifts", "Subtype": "Orthographic Issues",
        "Definition": "A plain misspelling.",
        "Severity_Level": "Minor", "Severity_Score": 1,
        "TQA_Category": "Fluency", "Weight_WGT": 1.00, "Category_Weight": 0.05,
    },
}

RAQEEB_23_CLASSES = sorted(TAXONOMY_V3.keys())
assert len(RAQEEB_23_CLASSES) == 23, f"Expected 23 classes, got {len(RAQEEB_23_CLASSES)}"

# TQA Category weights (for Sentence_Score calculation)
TQA_WEIGHTS = {"Accuracy": 0.42, "Fluency": 0.225, "Terminology": 0.28, "Style & Register": 0.075}

# Severity multipliers (for MQM penalty)
SEVERITY_MULTIPLIER = {"Minor": 1, "Moderate": 2, "Major": 3, "Severe": 4, "Critical": 5}

print(f"✅ Taxonomy loaded: {len(RAQEEB_23_CLASSES)} Raqeeb classes")
print(f"   TQA Categories: {list(TQA_WEIGHTS.keys())}")

# =============================================================================
# 3. DATA LOADING
# =============================================================================

def load_with_fallbacks(primary, fallbacks, label):
    """Load CSV with fallback paths."""
    for path in [primary] + fallbacks:
        if Path(path).exists():
            df = pd.read_csv(path, encoding='utf-8-sig')
            print(f"✅ Loaded {label}: {path} ({len(df):,} rows)")
            return df
    print(f"❌ {label}: No file found at {primary} or fallbacks")
    return pd.DataFrame()

print("\n--- Loading Data ---")
df_tier1_a = load_with_fallbacks(TIER1_A_PATH, TIER1_A_FALLBACKS, "Tier1 Clean_A")
df_tier1_b = load_with_fallbacks(TIER1_B_PATH, TIER1_B_FALLBACKS, "Tier1 Clean_B")
df_tier2 = load_with_fallbacks(TIER2_PATH, TIER2_FALLBACKS, "Tier2 Generated")
df_etca = load_with_fallbacks(ETCA_AUDIT_PATH, [], "ETCA Audit")

# Normalize error column name
for df, label in [(df_tier1_a, "T1A"), (df_tier1_b, "T1B"), (df_tier2, "T2")]:
    if not df.empty and ERROR_COL not in df.columns:
        for alt in ['Sub_Subtype', 'error_type', 'error_label', 'sub_subtype']:
            if alt in df.columns:
                df.rename(columns={alt: ERROR_COL}, inplace=True)
                print(f"   ℹ️ {label}: Renamed '{alt}' → '{ERROR_COL}'")
                break

# =============================================================================
# 4. MERGE TIER 1 A+B (source column preserved)
# =============================================================================

print("\n--- Merging Tier 1 A+B ---")
if not df_tier1_a.empty:
    df_tier1_a['source'] = 'Clean_A'
if not df_tier1_b.empty:
    df_tier1_b['source'] = 'Clean_B'

dfs_to_merge = [df for df in [df_tier1_a, df_tier1_b] if not df.empty]
if dfs_to_merge:
    df_tier1_combined = pd.concat(dfs_to_merge, ignore_index=True)
    print(f"✅ Combined Tier 1: {len(df_tier1_combined):,} rows")
    print(f"   Clean_A: {len(df_tier1_a):,}  |  Clean_B: {len(df_tier1_b):,}")
    if ERROR_COL in df_tier1_combined.columns:
        t1_classes = df_tier1_combined[ERROR_COL].nunique()
        print(f"   Classes: {t1_classes}")
else:
    df_tier1_combined = pd.DataFrame()
    print("❌ No Tier 1 data available")

# Tier 2 source tag
if not df_tier2.empty:
    df_tier2['source'] = 'Tier2_Generated'
    if ERROR_COL in df_tier2.columns:
        t2_classes = df_tier2[ERROR_COL].nunique()
        print(f"\n✅ Tier 2: {len(df_tier2):,} rows, {t2_classes} classes")

# =============================================================================
# 5. STRATIFIED SAMPLING ENGINE
# =============================================================================

def stratified_sample(df, target_n, min_per_class, error_col=ERROR_COL, seed=RANDOM_SEED,
                      exhaust_threshold=10):
    """
    Stratified sampling with rare-class exhaustion.
    1. Classes with ≤ exhaust_threshold in pool: take ALL available
    2. Other classes: guarantee min_per_class
    3. Fill remaining quota proportionally from frequent classes
    """
    rng = np.random.RandomState(seed)

    if error_col not in df.columns:
        print(f"  ⚠️ Column '{error_col}' not found. Returning random sample.")
        return df.sample(n=min(target_n, len(df)), random_state=seed)

    class_counts = df[error_col].value_counts()
    classes = class_counts.index.tolist()

    # Step 1: Exhaust rare classes, guarantee minimum for frequent ones
    selected_indices = []
    for cls in classes:
        cls_df = df[df[error_col] == cls]
        pool_size = len(cls_df)
        if pool_size <= exhaust_threshold:
            n_take = pool_size  # Take ALL — rare class exhaustion
        else:
            n_take = min(min_per_class, pool_size)
        selected = cls_df.sample(n=n_take, random_state=seed)
        selected_indices.extend(selected.index.tolist())

    # Step 2: Fill remaining proportionally from frequent classes
    remaining = target_n - len(selected_indices)
    if remaining > 0:
        available = df[~df.index.isin(selected_indices)]
        if len(available) > 0:
            available_counts = available[error_col].value_counts()
            total_available = available_counts.sum()
            for cls in available_counts.index:
                cls_available = available[available[error_col] == cls]
                n_prop = max(0, int(round(available_counts[cls] / total_available * remaining)))
                n_take = min(n_prop, len(cls_available))
                if n_take > 0:
                    extra = cls_available.sample(n=n_take, random_state=seed)
                    selected_indices.extend(extra.index.tolist())

    # Deduplicate and trim to target
    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) > target_n:
        selected_indices = selected_indices[:target_n]

    result = df.loc[selected_indices].copy()
    return result


def split_gold_batch(df, gold_per_class, error_col=ERROR_COL, seed=RANDOM_SEED):
    """
    Split sampled data into gold overlap + batch A + batch B.
    Gold: stratified with min gold_per_class per class.
    Remainder: split evenly into batch A and batch B.
    """
    rng = np.random.RandomState(seed)

    if error_col not in df.columns:
        n_gold = min(len(df) // 3, 60)
        gold = df.sample(n=n_gold, random_state=seed)
        remainder = df[~df.index.isin(gold.index)]
        mid = len(remainder) // 2
        return gold, remainder.iloc[:mid].copy(), remainder.iloc[mid:].copy()

    classes = df[error_col].unique()

    # Select gold: min gold_per_class per class, up to available
    gold_indices = []
    for cls in classes:
        cls_df = df[df[error_col] == cls]
        n_take = min(gold_per_class, len(cls_df))
        selected = cls_df.sample(n=n_take, random_state=seed)
        gold_indices.extend(selected.index.tolist())

    gold = df.loc[gold_indices].copy()
    remainder = df[~df.index.isin(gold_indices)].copy()

    # Split remainder into batch A and B (stratified)
    batch_a_indices = []
    batch_b_indices = []
    for cls in classes:
        cls_rem = remainder[remainder[error_col] == cls]
        if len(cls_rem) == 0:
            continue
        shuffled = cls_rem.sample(frac=1.0, random_state=seed)
        mid = len(shuffled) // 2
        batch_a_indices.extend(shuffled.index[:mid].tolist())
        batch_b_indices.extend(shuffled.index[mid:].tolist())

    batch_a = remainder.loc[batch_a_indices].copy()
    batch_b = remainder.loc[batch_b_indices].copy()

    return gold, batch_a, batch_b


# =============================================================================
# 6. EXECUTE SAMPLING
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: STRATIFIED SAMPLING")
print("=" * 80)

# --- Tier 1 Sampling (two-pass: exhaust rare classes + proportional fill) ---
if not df_tier1_combined.empty:
    # PASS 1: Calculate rare-class exhaustive samples
    # Classes with ≤ T1_EXHAUST_THRESHOLD in pool: take ALL available
    # Classes with > threshold: take min T1_MIN_PER_CLASS
    t1_class_counts = df_tier1_combined[ERROR_COL].value_counts() if ERROR_COL in df_tier1_combined.columns else pd.Series()
    t1_n_classes = len(t1_class_counts)

    rare_total = 0
    for cls, cnt in t1_class_counts.items():
        if cnt <= T1_EXHAUST_THRESHOLD:
            rare_total += cnt  # Take all available
        else:
            rare_total += T1_MIN_PER_CLASS  # Guarantee minimum

    # PASS 2: Add proportional fill from frequent classes to reach ~202
    t1_base_target = max(1, int(len(df_tier1_combined) * T1_BASE_RATE))  # 170
    t1_target = max(t1_base_target, rare_total + 33)  # rare_total + proportional top-up
    # Ensure we hit ~200-205 range for per-class robustness
    t1_target = max(t1_target, 200)
    t1_target = min(t1_target, len(df_tier1_combined))  # Can't exceed pool

    print(f"\n--- Tier 1 ---")
    print(f"   Pool: {len(df_tier1_combined):,} | Base 10%: {t1_base_target} | Rare-class floor: {rare_total}")
    print(f"   Target: {t1_target} (~{t1_target/len(df_tier1_combined)*100:.1f}% of pool)")
    print(f"   Rationale: 10% base ({t1_base_target}) + ~{t1_target - t1_base_target} targeted rare-class oversamples")

    t1_sampled = stratified_sample(df_tier1_combined, t1_target, T1_MIN_PER_CLASS)
    t1_gold, t1_batch_a, t1_batch_b = split_gold_batch(t1_sampled, GOLD_PER_CLASS_T1)

    print(f"   ✅ Sampled: {len(t1_sampled)} | Gold: {len(t1_gold)} | Batch A: {len(t1_batch_a)} | Batch B: {len(t1_batch_b)}")
    if ERROR_COL in t1_sampled.columns:
        print(f"   Classes in sample: {t1_sampled[ERROR_COL].nunique()}")
        print(f"   Gold per class avg: {len(t1_gold) / t1_sampled[ERROR_COL].nunique():.1f}")
else:
    t1_sampled, t1_gold, t1_batch_a, t1_batch_b = [pd.DataFrame()] * 4

# --- Tier 2 Sampling ---
if not df_tier2.empty:
    t2_target = min(T2_TARGET, len(df_tier2))

    print(f"\n--- Tier 2 ---")
    print(f"   Pool: {len(df_tier2):,} | Target: {t2_target}")

    t2_sampled = stratified_sample(df_tier2, t2_target, T2_MIN_PER_CLASS)
    t2_gold, t2_batch_a, t2_batch_b = split_gold_batch(t2_sampled, GOLD_PER_CLASS_T2)

    print(f"   ✅ Sampled: {len(t2_sampled)} | Gold: {len(t2_gold)} | Batch A: {len(t2_batch_a)} | Batch B: {len(t2_batch_b)}")
    if ERROR_COL in t2_sampled.columns:
        print(f"   Classes in sample: {t2_sampled[ERROR_COL].nunique()}")
        print(f"   Gold per class avg: {len(t2_gold) / max(1, t2_sampled[ERROR_COL].nunique()):.1f}")
else:
    t2_sampled, t2_gold, t2_batch_a, t2_batch_b = [pd.DataFrame()] * 4

# --- Training samples (separate from main sample) ---
print(f"\n--- Training ---")
t1_training = pd.DataFrame()
t2_training = pd.DataFrame()
if not df_tier1_combined.empty:
    t1_remaining_pool = df_tier1_combined[~df_tier1_combined.index.isin(t1_sampled.index)]
    if len(t1_remaining_pool) >= TRAINING_PER_TIER:
        t1_training = stratified_sample(t1_remaining_pool, TRAINING_PER_TIER, 1, seed=RANDOM_SEED + 1)
    else:
        t1_training = t1_remaining_pool.head(min(TRAINING_PER_TIER, len(t1_remaining_pool)))
    print(f"   T1 Training: {len(t1_training)} samples")

if not df_tier2.empty:
    t2_remaining_pool = df_tier2[~df_tier2.index.isin(t2_sampled.index)]
    if len(t2_remaining_pool) >= TRAINING_PER_TIER:
        t2_training = stratified_sample(t2_remaining_pool, TRAINING_PER_TIER, 1, seed=RANDOM_SEED + 1)
    else:
        t2_training = t2_remaining_pool.head(min(TRAINING_PER_TIER, len(t2_remaining_pool)))
    print(f"   T2 Training: {len(t2_training)} samples")

# =============================================================================
# 7. GENERATE EVAL_IDs
# =============================================================================

def generate_eval_ids(df, prefix, tier_label):
    """Add unique eval_id and tier column to dataframe."""
    df = df.copy()
    df['tier'] = tier_label
    df['eval_id'] = [f"{prefix}_{i+1:04d}" for i in range(len(df))]
    return df

# Tag all splits
t1_gold = generate_eval_ids(t1_gold, "T1G", "Tier1")
t1_batch_a = generate_eval_ids(t1_batch_a, "T1A", "Tier1")
t1_batch_b = generate_eval_ids(t1_batch_b, "T1B", "Tier1")
t1_training = generate_eval_ids(t1_training, "T1TR", "Tier1")

t2_gold = generate_eval_ids(t2_gold, "T2G", "Tier2")
t2_batch_a = generate_eval_ids(t2_batch_a, "T2A", "Tier2")
t2_batch_b = generate_eval_ids(t2_batch_b, "T2B", "Tier2")
t2_training = generate_eval_ids(t2_training, "T2TR", "Tier2")

# =============================================================================
# 8. SAVE SAMPLING CSVs
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: SAVE SAMPLING CSVs")
print("=" * 80)

t1_all = pd.concat([t1_gold, t1_batch_a, t1_batch_b], ignore_index=True)
t2_all = pd.concat([t2_gold, t2_batch_a, t2_batch_b], ignore_index=True)

t1_csv_path = f"{EVAL_DIR}/TIER1_HUMANEVAL_MERGED.csv"
t2_csv_path = f"{EVAL_DIR}/TIER2_HUMANEVAL_STRATIFIED.csv"

t1_all.to_csv(t1_csv_path, index=False, encoding='utf-8-sig')
t2_all.to_csv(t2_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ {t1_csv_path}: {len(t1_all)} rows")
print(f"✅ {t2_csv_path}: {len(t2_all)} rows")

# =============================================================================
# 9. VALIDATION & SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: VALIDATION")
print("=" * 80)

# Check no contamination between gold and batches
for tier_label, gold, ba, bb in [("T1", t1_gold, t1_batch_a, t1_batch_b),
                                   ("T2", t2_gold, t2_batch_a, t2_batch_b)]:
    if gold.empty:
        continue
    g_ids = set(gold['eval_id'])
    a_ids = set(ba['eval_id'])
    b_ids = set(bb['eval_id'])
    assert g_ids.isdisjoint(a_ids), f"{tier_label}: Gold ∩ Batch A not empty!"
    assert g_ids.isdisjoint(b_ids), f"{tier_label}: Gold ∩ Batch B not empty!"
    assert a_ids.isdisjoint(b_ids), f"{tier_label}: Batch A ∩ Batch B not empty!"
    print(f"✅ {tier_label}: No contamination (Gold ∩ A = ∅, Gold ∩ B = ∅, A ∩ B = ∅)")

# Summary table
print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    SAMPLING SUMMARY                                   ║
╠════════════════════════╦══════════╦══════════╦══════════╦══════════════╣
║ Split                  ║  Tier 1  ║  Tier 2  ║  Total   ║ Notes        ║
╠════════════════════════╬══════════╬══════════╬══════════╬══════════════╣
║ Training               ║  {len(t1_training):>5}   ║  {len(t2_training):>5}   ║  {len(t1_training)+len(t2_training):>5}   ║ Not in total ║
║ Phase 1 Gold           ║  {len(t1_gold):>5}   ║  {len(t2_gold):>5}   ║  {len(t1_gold)+len(t2_gold):>5}   ║ Dual-annot.  ║
║ Phase 3 Batch A        ║  {len(t1_batch_a):>5}   ║  {len(t2_batch_a):>5}   ║  {len(t1_batch_a)+len(t2_batch_a):>5}   ║ Ann. A only  ║
║ Phase 3 Batch B        ║  {len(t1_batch_b):>5}   ║  {len(t2_batch_b):>5}   ║  {len(t1_batch_b)+len(t2_batch_b):>5}   ║ Ann. B only  ║
╠════════════════════════╬══════════╬══════════╬══════════╬══════════════╣
║ UNIQUE SAMPLES         ║  {len(t1_all):>5}   ║  {len(t2_all):>5}   ║  {len(t1_all)+len(t2_all):>5}   ║              ║
╚════════════════════════╩══════════╩══════════╩══════════╩══════════════╝
""")

# Class distribution in gold (for IAA reporting)
if ERROR_COL in t1_gold.columns and not t1_gold.empty:
    print("--- Tier 1 Gold: Class Distribution ---")
    for cls, cnt in t1_gold[ERROR_COL].value_counts().items():
        print(f"   {cls:<30} {cnt}")

    # CRITICAL: Report missing classes explicitly
    t1_present = set(t1_all[ERROR_COL].unique()) if ERROR_COL in t1_all.columns else set()
    t1_missing = set(RAQEEB_23_CLASSES) - t1_present
    if t1_missing:
        print(f"\n   ⚠️ Tier 1 covers {len(t1_present)}/23 classes. Missing {len(t1_missing)} (Tier 2-only):")
        for cls in sorted(t1_missing):
            print(f"      ✗ {cls}")
        print(f"   → These {len(t1_missing)} classes were NOT found by the TRIVET mining pipeline")
        print(f"     in the UN corpus. Per-class validation relies on Tier 2 synthetic data.")

    # Report singleton classes (present but not usable for IAA)
    singleton_classes = [cls for cls, cnt in t1_all[ERROR_COL].value_counts().items() if cnt <= 2]
    if singleton_classes:
        print(f"\n   ⚠️ {len(singleton_classes)} classes have ≤2 samples (no per-class IAA possible):")
        for cls in singleton_classes:
            cnt = t1_all[t1_all[ERROR_COL] == cls].shape[0]
            print(f"      △ {cls} ({cnt} sample{'s' if cnt > 1 else ''})")

if ERROR_COL in t2_gold.columns and not t2_gold.empty:
    print("\n--- Tier 2 Gold: Class Distribution ---")
    for cls, cnt in t2_gold[ERROR_COL].value_counts().items():
        print(f"   {cls:<30} {cnt}")

# =============================================================================
# 10. SAVE METADATA JSON
# =============================================================================

metadata = {
    "generated_at": datetime.now().isoformat(),
    "random_seed": RANDOM_SEED,
    "taxonomy_classes": len(RAQEEB_23_CLASSES),
    "tier1": {
        "pool_total": len(df_tier1_combined),
        "pool_clean_a": len(df_tier1_a),
        "pool_clean_b": len(df_tier1_b),
        "sampled": len(t1_all),
        "gold": len(t1_gold),
        "batch_a": len(t1_batch_a),
        "batch_b": len(t1_batch_b),
        "training": len(t1_training),
        "classes_in_sample": int(t1_all[ERROR_COL].nunique()) if ERROR_COL in t1_all.columns else 0,
        "gold_per_class": dict(t1_gold[ERROR_COL].value_counts()) if ERROR_COL in t1_gold.columns else {},
    },
    "tier2": {
        "pool_total": len(df_tier2),
        "sampled": len(t2_all),
        "gold": len(t2_gold),
        "batch_a": len(t2_batch_a),
        "batch_b": len(t2_batch_b),
        "training": len(t2_training),
        "classes_in_sample": int(t2_all[ERROR_COL].nunique()) if ERROR_COL in t2_all.columns else 0,
        "gold_per_class": dict(t2_gold[ERROR_COL].value_counts()) if ERROR_COL in t2_gold.columns else {},
    },
    "files_generated": [],  # Will be populated by workbook generator
}

meta_path = f"{EVAL_DIR}/sampling_metadata.json"
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\n✅ Metadata saved: {meta_path}")

print("\n" + "=" * 80)
print("STEP 1-3 COMPLETE: Sampling ready. Proceeding to workbook generation...")
print("=" * 80)
# =============================================================================
# =============================================================================
# CELL 15.2 — PART 2: PROFESSIONAL WORKBOOK GENERATION ENGINE
# =============================================================================
# All variables from Part 1 (t1_gold, t1_batch_a, etc.) are available.
# =============================================================================

if not OPENPYXL_AVAILABLE:
    raise ImportError("openpyxl required. Run: pip install openpyxl")

print("\n" + "=" * 80)
print("STEP 4: PROFESSIONAL WORKBOOK GENERATION")
print("=" * 80)

from openpyxl.formatting.rule import CellIsRule
from openpyxl.comments import Comment

# =============================================================================
# RAQEEB DESIGN SYSTEM
# =============================================================================

# Brand colors
RAQEEB_NAVY  = "1B2A4A"
RAQEEB_TEAL  = "2C6E6A"
RAQEEB_GOLD  = "C9A84C"
RAQEEB_LIGHT = "EDF2F7"

# Functional colors
CLR_PREFILL   = "F0F4F8"
CLR_ANNOTATOR = "FFF8E1"
CLR_AUTO      = "E8F5E9"
CLR_LOCKED    = "ECEFF1"
CLR_ARABIC_BG = "FFF9F0"
CLR_MT_BG     = "FFF3E0"

# Severity
CLR_SEV_MINOR = "A5D6A7"
CLR_SEV_MOD   = "FFF176"
CLR_SEV_MAJOR = "FFB74D"
CLR_SEV_CRIT  = "EF5350"

# TQA categories
TQA_CLR = {
    "Accuracy": "BBDEFB", "Fluency": "C8E6C9",
    "Terminology": "FFE0B2", "Style & Register": "E1BEE7",
}

# Status
CLR_ACCEPT = "C8E6C9"
CLR_REVIEW = "FFCDD2"

# Fonts
F_BRAND    = Font(name='Arial', bold=True, size=20, color="FFFFFF")
F_BRAND_SUB= Font(name='Arial', size=11, color=RAQEEB_GOLD, italic=True)
F_SECTION  = Font(name='Arial', bold=True, size=12, color=RAQEEB_TEAL)
F_SEC_HDR  = Font(name='Arial', bold=True, size=9, color="FFFFFF")
F_HEADER   = Font(name='Arial', bold=True, size=10, color="FFFFFF")
F_BODY     = Font(name='Arial', size=10)
F_BODY_AR  = Font(name='Arial', size=12)
F_BODY_SM  = Font(name='Arial', size=9, color="78909C")
F_GOLD_ACC = Font(name='Arial', bold=True, size=10, color=RAQEEB_GOLD)

# Fills
FILL_HEADER    = PatternFill('solid', fgColor=RAQEEB_NAVY)
FILL_BRAND_BAR = PatternFill('solid', fgColor=RAQEEB_NAVY)
FILL_GOLD_BAR  = PatternFill('solid', fgColor=RAQEEB_GOLD)
FILL_PREFILL   = PatternFill('solid', fgColor=CLR_PREFILL)
FILL_ANNOTATOR = PatternFill('solid', fgColor=CLR_ANNOTATOR)
FILL_AUTO      = PatternFill('solid', fgColor=CLR_AUTO)
FILL_LOCKED    = PatternFill('solid', fgColor=CLR_LOCKED)
FILL_ARABIC_BG = PatternFill('solid', fgColor=CLR_ARABIC_BG)
FILL_MT_BG     = PatternFill('solid', fgColor=CLR_MT_BG)
FILL_ALT_ROW   = PatternFill('solid', fgColor=RAQEEB_LIGHT)

# Alignment
A_CENTER   = Alignment(horizontal='center', vertical='center', wrap_text=True)
A_LEFT     = Alignment(horizontal='left', vertical='center', wrap_text=True)
A_RIGHT_AR = Alignment(horizontal='right', vertical='center', wrap_text=True, readingOrder=2)

# Border
B_THIN = Border(
    left=Side('thin', color='CFD8DC'), right=Side('thin', color='CFD8DC'),
    top=Side('thin', color='CFD8DC'), bottom=Side('thin', color='CFD8DC'))
B_HEADER = Border(
    left=Side('thin', color=RAQEEB_NAVY), right=Side('thin', color=RAQEEB_NAVY),
    top=Side('medium', color=RAQEEB_NAVY), bottom=Side('medium', color=RAQEEB_NAVY))

# Arabic class names
AR_CLASS = {
    "Meaning Shift": "تحوّل المعنى", "Total Omission": "حذف كامل",
    "Partial Translation": "ترجمة جزئية", "Name Entity Error": "خطأ في الكيان المسمّى",
    "Literal Translation": "ترجمة حرفية", "Hypernym for Hyponym": "استخدام الأعم بدل الأخص",
    "Hyponym for Hypernym": "استخدام الأخص بدل الأعم",
    "Invalid Pattern": "نمط صرفي غير صحيح", "Tanween Omission": "حذف التنوين",
    "Gender Disagreement": "عدم مطابقة الجنس", "Definiteness Shift": "تغيّر التعريف",
    "Perfective to Progressive": "من الماضي إلى المضارع",
    "Progressive to Perfective": "من المضارع إلى الماضي",
    "Tense Shift Under Negation": "تغيّر الزمن تحت النفي",
    "Wrong Structure": "تركيب خاطئ", "Wrong Word Order": "ترتيب خاطئ للكلمات",
    "Noun to Adjective": "اسم إلى صفة", "Adjective to Noun": "صفة إلى اسم",
    "Active to Passive Voice": "من المبني للمعلوم إلى المجهول",
    "Passive to Active Voice": "من المبني للمجهول إلى المعلوم",
    "Register Mismatch": "عدم تطابق السجل اللغوي",
    "Terminology Substitution": "استبدال مصطلحي",
    "Spelling Error": "خطأ إملائي",
}
AR_TQA = {"Accuracy": "الدقة", "Fluency": "الطلاقة",
           "Terminology": "المصطلحات", "Style & Register": "الأسلوب والسجل"}

CLASS_HIERARCHY = {
    "Semantic Divergence | التباعد الدلالي (40%)": [
        "Meaning Shift", "Total Omission", "Partial Translation",
        "Name Entity Error", "Literal Translation",
        "Hypernym for Hyponym", "Hyponym for Hypernym"],
    "Morphosyntactic Shifts | التحولات الصرفية (10%)": [
        "Invalid Pattern", "Tanween Omission",
        "Gender Disagreement", "Definiteness Shift"],
    "Syntactic Shift | التحولات النحوية (10%)": [
        "Perfective to Progressive", "Progressive to Perfective",
        "Tense Shift Under Negation", "Wrong Structure", "Wrong Word Order",
        "Noun to Adjective", "Adjective to Noun",
        "Active to Passive Voice", "Passive to Active Voice"],
    "Pragmatic Divergence | التباعد التداولي (35%)": [
        "Register Mismatch", "Terminology Substitution"],
    "Orthographic Shifts | التحولات الإملائية (5%)": [
        "Spelling Error"],
}


def sc(ws, row, col, value, font=None, fill=None, alignment=None, border=None,
       number_format=None, comment_text=None):
    cell = ws.cell(row=row, column=col, value=value)
    if font: cell.font = font
    if fill: cell.fill = fill
    if alignment: cell.alignment = alignment
    if border: cell.border = border
    if number_format: cell.number_format = number_format
    if comment_text: cell.comment = Comment(comment_text, "Raqeeb")
    return cell


def get_tax(cls_name):
    return TAXONOMY_V3.get(cls_name, {"Class": "Unknown", "Subtype": "Unknown",
        "Definition": "", "Severity_Level": "Unknown", "Severity_Score": 0,
        "TQA_Category": "Unknown", "Weight_WGT": 0, "Category_Weight": 0})


def abs_weight(cls_name):
    t = get_tax(cls_name)
    return round(t["Weight_WGT"] * t["Category_Weight"], 6)


# =============================================================================
# DASHBOARD
# =============================================================================

def build_dashboard(wb, phase_name, annotator_id, tier_label, n_samples,
                    n_gold=0, n_classes=0):
    ws = wb.create_sheet("Dashboard")
    ws.sheet_properties.tabColor = RAQEEB_GOLD
    for col, w in [(1,3),(2,28),(3,30),(4,18),(5,18),(6,3)]:
        ws.column_dimensions[get_column_letter(col)].width = w

    # Brand bar
    for c in range(1, 7):
        sc(ws, 1, c, '', fill=FILL_BRAND_BAR)
        sc(ws, 2, c, '', fill=FILL_BRAND_BAR)
    sc(ws, 1, 2, "رقيب  RAQEEB", F_BRAND, FILL_BRAND_BAR, A_LEFT)
    sc(ws, 2, 2, "TRIVET Translation Quality Evaluation Framework",
       F_BRAND_SUB, FILL_BRAND_BAR, A_LEFT)
    sc(ws, 2, 4, "EN → AR  |  UN Domain",
       Font(name='Arial', size=10, color="90A4AE"), FILL_BRAND_BAR,
       Alignment(horizontal='right', vertical='center'))
    for c in range(1, 7):
        sc(ws, 3, c, '', fill=FILL_GOLD_BAR)

    r = 5
    info = [
        ("Phase | المرحلة", phase_name),
        ("Annotator | المُقيّم", annotator_id),
        ("Data Tier | مستوى البيانات", tier_label),
        ("Total Samples | عدد العيّنات", n_samples),
        ("Gold Overlap | التداخل الذهبي", n_gold if n_gold else "—"),
        ("Classes | التصنيفات", f"{n_classes} Raqeeb classes"),
        ("Date | التاريخ", "(fill on start)"),
    ]
    for label, value in info:
        sc(ws, r, 2, label, F_SECTION, alignment=A_LEFT, border=B_THIN)
        sc(ws, r, 3, value, F_BODY, alignment=A_LEFT, border=B_THIN)
        r += 1

    r += 1
    sc(ws, r, 2, "PROGRESS TRACKER | متابعة التقدم", F_SECTION)
    r += 1
    sc(ws, r, 2, "Task", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    sc(ws, r, 3, "Completed", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    sc(ws, r, 4, "Progress %", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    r += 1
    for name, ar, count_f in [
        ("Layer 1 — Blind", "الطبقة 1",
         f'=COUNTA(\'Raqeeb Layer 1\'!I4:I{n_samples+3})'),
        ("Layer 2 — Keyword", "الطبقة 2",
         f'=COUNTA(\'Raqeeb Layer 2\'!M4:M{n_samples+3})'),
    ]:
        sc(ws, r, 2, f"{name}\n{ar}", F_BODY, FILL_PREFILL, A_LEFT, B_THIN)
        sc(ws, r, 3, count_f, F_BODY, alignment=A_CENTER, border=B_THIN)
        sc(ws, r, 4, f'=IFERROR(C{r}/{n_samples}*100,0)',
           F_GOLD_ACC, alignment=A_CENTER, border=B_THIN, number_format='0.0"%"')
        r += 1

    r += 2
    sc(ws, r, 2, "⚠️  WORKFLOW | سير العمل", F_SECTION)
    r += 1
    for num, text in [
        ("1.", "Complete ALL of Layer 1 FIRST | أكمل الطبقة 1 أولاً"),
        ("2.", "THEN open Layer 2 | ثم افتح الطبقة 2"),
        ("3.", "Fill ONLY yellow cells | املأ الخلايا الصفراء فقط"),
        ("4.", "Do NOT edit grey/green cells | لا تعدّل الرمادية/الخضراء"),
        ("5.", "Save frequently | احفظ بشكل متكرر"),
    ]:
        sc(ws, r, 2, num, F_GOLD_ACC, alignment=A_CENTER)
        sc(ws, r, 3, text, F_BODY, alignment=A_LEFT)
        r += 1

    r += 1
    sc(ws, r, 2, "COLOR GUIDE | دليل الألوان", F_SECTION)
    r += 1
    for fill, desc in [
        (FILL_ANNOTATOR, "Your input — fill this | أدخل هنا"),
        (FILL_PREFILL, "Pre-filled — do not edit | بيانات مسبقة"),
        (FILL_AUTO, "Auto-calculated | محسوب تلقائياً"),
        (FILL_LOCKED, "Locked / N/A | مقفل"),
    ]:
        sc(ws, r, 2, "     ", fill=fill, border=B_THIN)
        sc(ws, r, 3, desc, F_BODY, alignment=A_LEFT)
        r += 1
    return ws


# =============================================================================
# LAYER 1 — BLIND SENTENCE EVALUATION
# =============================================================================

def build_layer1(wb, samples_df, sheet_name="Raqeeb Layer 1"):
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = "2196F3"
    n = len(samples_df)
    N_COLS = 20

    # Row 1: Brand
    for c in range(1, N_COLS+1):
        sc(ws, 1, c, '', fill=FILL_BRAND_BAR)
    sc(ws, 1, 1, "رقيب  Raqeeb Layer 1 — Blind Sentence-Level MQM | الطبقة 1 — تقييم الجملة الأعمى",
       Font(name='Arial', bold=True, size=13, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)

    # Row 2: Section labels
    for start, end, label, color in [
        (1,8,"A: METADATA | البيانات الوصفية", RAQEEB_NAVY),
        (9,12,"B: ERROR DETECTION | كشف الأخطاء", "E65100"),
        (13,16,"C: MQM HOLISTIC | التقييم الشامل", "1565C0"),
        (17,18,"D: AUTO-CALCULATED | محسوب تلقائياً", "2E7D32"),
        (19,20,"E: ANNOTATOR META | بيانات المُقيّم", "546E7A"),
    ]:
        for c in range(start, end+1):
            sc(ws, 2, c, '', fill=PatternFill('solid', fgColor=color))
        sc(ws, 2, start, label, F_SEC_HDR,
           PatternFill('solid', fgColor=color), A_LEFT, B_HEADER)
        if end > start:
            ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)

    # Row 3: Headers (bilingual)
    hdrs = [
        ('No.\nرقم',5), ('Eval_ID\nمعرّف',12), ('Sentence_ID\nمعرّف الجملة',15),
        ('Tier\nالمستوى',6), ('Source\nالمصدر',8),
        ('English (Source)\nالإنجليزية',40), ('Arabic (Ref)\nالعربية',40),
        ('MT Output\nمخرجات الترجمة',40),
        ('Error Found?\nخطأ؟ (Yes/No)',11), ('Error Type\nنوع الخطأ',22),
        ('Severity\nالحدّة',13), ('Confidence\nالثقة (1-5)',10),
        ('Accuracy\nالدقة (1-5)',11), ('Fluency\nالطلاقة (1-5)',11),
        ('Terminology\nالمصطلحات (1-5)',13), ('Style\nالأسلوب (1-5)',11),
        ('Sentence Score\nدرجة الجملة',13), ('Binary Error\nخطأ ثنائي',10),
        ('Duration (min)\nالمدة',10), ('Notes\nملاحظات',25),
    ]
    for c, (h, w) in enumerate(hdrs, 1):
        sc(ws, 3, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[3].height = 45

    # Data rows
    for i, (_, row) in enumerate(samples_df.iterrows()):
        r = 4 + i
        alt = FILL_ALT_ROW if i % 2 == 0 else None
        # A: Metadata
        sc(ws, r, 1, i+1, F_BODY, alt or FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 2, row.get('eval_id',''), F_BODY, alt or FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 3, str(row.get('Sentence_ID','')), F_BODY_SM, alt or FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 4, row.get('tier',''), F_BODY, alt or FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 5, row.get('source',''), F_BODY, alt or FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 6, str(row.get('en','')), F_BODY, alt or FILL_PREFILL, A_LEFT, B_THIN)
        sc(ws, r, 7, str(row.get('ar','')), F_BODY_AR, FILL_ARABIC_BG, A_RIGHT_AR, B_THIN)
        sc(ws, r, 8, str(row.get('mt_output','')), F_BODY_AR, FILL_MT_BG, A_RIGHT_AR, B_THIN)
        # B: Error detection (annotator)
        for c in range(9, 13):
            sc(ws, r, c, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        # C: MQM (annotator)
        for c in range(13, 17):
            sc(ws, r, c, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        # D: Auto
        sc(ws, r, 17, f'=IFERROR((M{r}*0.42+N{r}*0.225+O{r}*0.28+P{r}*0.075)/5*100,"")',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        sc(ws, r, 18, f'=IF(I{r}="Yes",1,IF(I{r}="No",0,""))',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN)
        # E: Meta
        sc(ws, r, 19, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        sc(ws, r, 20, '', F_BODY, FILL_ANNOTATOR, A_LEFT, B_THIN)

    last = 3 + n
    # Validations
    dv_yn = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    dv_yn.prompt = "Select Yes or No | اختر نعم أو لا"
    ws.add_data_validation(dv_yn)
    dv_yn.add(f"I4:I{last}")

    dv_sev = DataValidation(type="list", formula1='"Minor,Moderate,Major,Critical"', allow_blank=True)
    dv_sev.prompt = "Minor | Moderate | Major | Critical"
    ws.add_data_validation(dv_sev)
    dv_sev.add(f"K4:K{last}")

    dv_15 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dv_15.prompt = "Rate 1-5 | قيّم من 1 إلى 5"
    ws.add_data_validation(dv_15)
    for col in ['L','M','N','O','P']:
        dv_15.add(f"{col}4:{col}{last}")

    # Conditional formatting: Severity
    for val, clr in [("Minor", CLR_SEV_MINOR), ("Moderate", CLR_SEV_MOD),
                     ("Major", CLR_SEV_MAJOR), ("Critical", CLR_SEV_CRIT)]:
        ws.conditional_formatting.add(f"K4:K{last}",
            CellIsRule(operator='equal', formula=[f'"{val}"'],
                       fill=PatternFill('solid', fgColor=clr)))

    # Conditional formatting: Sentence Score bands
    ws.conditional_formatting.add(f"Q4:Q{last}",
        CellIsRule(operator='lessThan', formula=['50'],
                   fill=PatternFill('solid', fgColor=CLR_SEV_CRIT),
                   font=Font(color="FFFFFF", bold=True)))
    ws.conditional_formatting.add(f"Q4:Q{last}",
        CellIsRule(operator='between', formula=['50','75'],
                   fill=PatternFill('solid', fgColor=CLR_SEV_MOD)))
    ws.conditional_formatting.add(f"Q4:Q{last}",
        CellIsRule(operator='greaterThan', formula=['75'],
                   fill=PatternFill('solid', fgColor=CLR_SEV_MINOR)))

    ws.freeze_panes = 'I4'
    return ws


# =============================================================================
# LAYER 2 — KEYWORD-LEVEL VALIDATION
# =============================================================================

def build_layer2(wb, samples_df, tier_label, etca_lookup=None,
                 sheet_name="Raqeeb Layer 2"):
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = "4CAF50"
    n = len(samples_df)
    is_t2 = (tier_label == "Tier2")
    N_COLS = 36

    # Row 1: Brand
    for c in range(1, N_COLS+1):
        sc(ws, 1, c, '', fill=FILL_BRAND_BAR)
    sc(ws, 1, 1, "رقيب  Raqeeb Layer 2 — Keyword Validation | الطبقة 2 — التحقق من الكلمات المفتاحية",
       Font(name='Arial', bold=True, size=13, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)

    # Row 2: Section labels
    for start, end, label, color in [
        (1,12,"A: REVEALED DATA | البيانات المكشوفة", RAQEEB_NAVY),
        (13,19,"B: CORE VALIDATION | التحقق الأساسي", "E65100"),
        (20,22,"C: SYNTHETIC | جودة التوليد" if is_t2 else "C: N/A", "7B1FA2" if is_t2 else "9E9E9E"),
        (23,28,"D: AUTO-SCORES | الدرجات التلقائية", "2E7D32"),
        (29,33,"E: ETCA COMPARISON | مقارنة ETCA", "0277BD"),
        (34,36,"F: ANNOTATOR | المُقيّم", "546E7A"),
    ]:
        for c in range(start, end+1):
            sc(ws, 2, c, '', fill=PatternFill('solid', fgColor=color))
        sc(ws, 2, start, label, F_SEC_HDR,
           PatternFill('solid', fgColor=color), A_LEFT, B_HEADER)
        if end > start:
            ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)

    # Row 3: Headers
    hdrs = [
        ('No.\nرقم',5), ('Eval_ID\nمعرّف',12), ('Tier\nالمستوى',6), ('Source\nالمصدر',8),
        ('Sub-Subtype\nالتصنيف الفرعي',22), ('Keyword\nالكلمة المفتاحية',18),
        ('Best_Match\nأفضل تطابق',18), ('Arabic (Ref)\nالعربية',35),
        ('MT Output\nمخرجات الترجمة',35), ('Severity\nالحدّة',8),
        ('Weight\nالوزن',8), ('TQA Cat.\nفئة TQA',14),
        ('Q1 Error\nCorrect\nخطأ صحيح؟',10), ('Q2 Anchor\nCorrect\nمرساة صحيحة؟',10),
        ('Q3 Severity\nحدّة الكلمة',10), ('Q4 Minimal\nPair\nزوج أدنى؟',10),
        ('Q5 Label\nCorrect\nتصنيف صحيح؟',10), ('Q6 Natural\nness\nطبيعية',10),
        ('Q7 UN Reg.\nمناسب للأمم المتحدة؟',10),
        ('S1 Fluency\nطلاقة',10), ('S2 Realism\nواقعية',10), ('S3 Domain\nمجال',10),
        ('KPS\nدرجة KPS',10), ('Tech Score\nالدرجة التقنية',10), ('DQI\nمؤشر الجودة',8),
        ('TRIVET\nScore',10), ('Quality\nScore',10), ('Status\nالحالة',10),
        ('ETCA\nAnchor',9), ('ETCA\nClarity',9), ('ETCA\nNatural',9),
        ('ETCA\nCollat.',9), ('ETCA\nSeed',9),
        ('Duration\nالمدة',8), ('Cog Load\nالعبء',10), ('Notes\nملاحظات',22),
    ]
    for c, (h, w) in enumerate(hdrs, 1):
        sc(ws, 3, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[3].height = 55

    # Data rows
    for i, (_, row) in enumerate(samples_df.iterrows()):
        r = 4 + i
        cls = str(row.get(ERROR_COL, ''))
        tax = get_tax(cls)
        wgt = abs_weight(cls)
        tqa_fill = PatternFill('solid', fgColor=TQA_CLR.get(tax["TQA_Category"], "F2F2F2"))

        # A: Metadata
        sc(ws, r, 1, i+1, F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 2, row.get('eval_id',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 3, row.get('tier',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 4, row.get('source',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 5, cls, Font(name='Arial', bold=True, size=10), tqa_fill, A_CENTER, B_THIN)
        sc(ws, r, 6, str(row.get('Keyword','')), F_BODY_AR, FILL_ARABIC_BG, A_RIGHT_AR, B_THIN)
        sc(ws, r, 7, str(row.get('Best_Match','')), F_BODY_AR, FILL_ARABIC_BG, A_RIGHT_AR, B_THIN)
        sc(ws, r, 8, str(row.get('ar','')), F_BODY_AR, FILL_ARABIC_BG, A_RIGHT_AR, B_THIN)
        sc(ws, r, 9, str(row.get('mt_output','')), F_BODY_AR, FILL_MT_BG, A_RIGHT_AR, B_THIN)
        sc(ws, r, 10, tax["Severity_Score"], F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, r, 11, wgt, F_BODY, FILL_PREFILL, A_CENTER, B_THIN, '0.0000')
        sc(ws, r, 12, tax["TQA_Category"], F_BODY, tqa_fill, A_CENTER, B_THIN)

        # B: Core Q (annotator)
        for c in range(13, 20):
            sc(ws, r, c, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        # C: Synthetic
        s_fill = FILL_ANNOTATOR if is_t2 else FILL_LOCKED
        for c in range(20, 23):
            sc(ws, r, c, '' if is_t2 else 'N/A', F_BODY, s_fill, A_CENTER, B_THIN)

        # D: Auto formulas
        sc(ws, r, 23, f'=IFERROR(MAX(0,100-MIN(100,O{r}*K{r}*20/MAX(1,LEN(TRIM(I{r}))-LEN(SUBSTITUTE(TRIM(I{r})," ",""))+1)*100)),"")',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        sc(ws, r, 24, f'=IFERROR((M{r}+N{r}+P{r}+Q{r})/4*100,"")',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        sc(ws, r, 25, f'=IFERROR((M{r}+N{r}+P{r}+Q{r})/4,"")',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.00')
        sc(ws, r, 26, f'=IFERROR(W{r},"")', F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        if is_t2:
            sc(ws, r, 27, f'=IFERROR(AVERAGE(R{r},T{r},U{r},V{r}),"")',
               F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        else:
            sc(ws, r, 27, f'=IFERROR(R{r},"")', F_BODY, FILL_AUTO, A_CENTER, B_THIN, '0.0')
        sc(ws, r, 28, f'=IF(Y{r}="","",IF(Y{r}>=0.75,"ACCEPT","REVIEW"))',
           F_BODY, FILL_AUTO, A_CENTER, B_THIN)

        # E: ETCA
        etca = {}
        if etca_lookup and row.get('Sentence_ID') in etca_lookup:
            etca = etca_lookup[row['Sentence_ID']]
        for j, key in enumerate(['anchor_validity','phenomenon_clarity',
                                  'arabic_naturalness','collateral_severity',
                                  'tier2_seed_recommendation_final']):
            sc(ws, r, 29+j, etca.get(key,''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)

        # F: Meta
        sc(ws, r, 34, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        sc(ws, r, 35, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        sc(ws, r, 36, '', F_BODY, FILL_ANNOTATOR, A_LEFT, B_THIN)

    last = 3 + n
    # Validations
    dv01 = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    dv01.prompt = "0=No | 1=Yes"
    ws.add_data_validation(dv01)
    for c in [13,14,16,17,19]:
        dv01.add(f"{get_column_letter(c)}4:{get_column_letter(c)}{last}")

    dvsv = DataValidation(type="list", formula1='"0,0.3,0.7,1.0"', allow_blank=True)
    dvsv.prompt = "0=None | 0.3=Minor | 0.7=Major | 1.0=Critical"
    ws.add_data_validation(dvsv)
    dvsv.add(f"O4:O{last}")

    dv15 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dv15.prompt = "Rate 1-5 | قيّم 1-5"
    ws.add_data_validation(dv15)
    dv15.add(f"R4:R{last}")
    dv15.add(f"AI4:AI{last}")
    if is_t2:
        for c in ['T','U','V']:
            dv15.add(f"{c}4:{c}{last}")

    # Conditional: Status
    ws.conditional_formatting.add(f"AB4:AB{last}",
        CellIsRule(operator='equal', formula=['"ACCEPT"'],
                   fill=PatternFill('solid', fgColor=CLR_ACCEPT),
                   font=Font(bold=True, color="1B5E20")))
    ws.conditional_formatting.add(f"AB4:AB{last}",
        CellIsRule(operator='equal', formula=['"REVIEW"'],
                   fill=PatternFill('solid', fgColor=CLR_REVIEW),
                   font=Font(bold=True, color="B71C1C")))

    # Conditional: DQI bands
    for op, args, clr in [
        ('lessThan', ['0.5'], CLR_SEV_CRIT),
        ('between', ['0.5','0.74'], CLR_SEV_MOD),
        ('greaterThanOrEqual', ['0.75'], CLR_SEV_MINOR)]:
        ws.conditional_formatting.add(f"Y4:Y{last}",
            CellIsRule(operator=op, formula=args,
                       fill=PatternFill('solid', fgColor=clr)))

    ws.freeze_panes = 'M4'
    return ws


# =============================================================================
# TAXONOMY SHEET (with tree diagram + bilingual reference table)
# =============================================================================

def build_taxonomy(wb, sheet_name="Raqeeb Taxonomy"):
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = RAQEEB_GOLD

    for c in range(1, 14):
        sc(ws, 1, c, '', fill=FILL_BRAND_BAR)
    sc(ws, 1, 1, "رقيب  Raqeeb V4 Taxonomy — 23 Classes | تصنيف رقيب — ٢٣ فئة",
       Font(name='Arial', bold=True, size=13, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)
    for c in range(1, 14):
        sc(ws, 2, c, '', fill=FILL_GOLD_BAR)

    # Tree
    sc(ws, 4, 1, "TAXONOMY TREE | شجرة التصنيف", F_SECTION)
    r = 5
    for parent, children in CLASS_HIERARCHY.items():
        sc(ws, r, 2, parent, Font(name='Arial', bold=True, size=11, color=RAQEEB_TEAL))
        r += 1
        for i, child in enumerate(children):
            conn = "└──" if i == len(children)-1 else "├──"
            tax = TAXONOMY_V3[child]
            tf = PatternFill('solid', fgColor=TQA_CLR.get(tax["TQA_Category"], "F2F2F2"))
            sc(ws, r, 1, conn, Font(name='Consolas', size=10, color="90A4AE"))
            sc(ws, r, 2, f"{child} | {AR_CLASS.get(child,'')}", F_BODY, tf, A_LEFT, B_THIN)
            sc(ws, r, 3, tax["Severity_Level"], F_BODY, alignment=A_CENTER)
            sc(ws, r, 4, tax["TQA_Category"], F_BODY_SM, alignment=A_CENTER)
            r += 1
        r += 1

    # Reference table
    r += 1
    sc(ws, r, 1, "FULL REFERENCE | الجدول الكامل", F_SECTION)
    r += 1
    for c, (h, w) in enumerate([('#',4), ('Sub-Subtype',25), ('العربية',22), ('Class',22),
            ('Subtype',20), ('TQA Category',16), ('Severity',12), ('Score',6),
            ('WGT',8), ('Cat.WGT',8), ('Abs.Weight',10), ('Definition',55)], 1):
        sc(ws, r, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
        ws.column_dimensions[get_column_letter(c)].width = w
    r += 1

    for i, cls in enumerate(RAQEEB_23_CLASSES):
        tax = TAXONOMY_V3[cls]
        tf = PatternFill('solid', fgColor=TQA_CLR.get(tax["TQA_Category"], "F2F2F2"))
        aw = round(tax["Weight_WGT"] * tax["Category_Weight"], 4)
        sc(ws, r, 1, i+1, F_BODY, None, A_CENTER, B_THIN)
        sc(ws, r, 2, cls, Font(name='Arial', bold=True, size=10), tf, A_LEFT, B_THIN)
        sc(ws, r, 3, AR_CLASS.get(cls,''), F_BODY_AR, None, A_RIGHT_AR, B_THIN)
        sc(ws, r, 4, tax["Class"], F_BODY, None, A_LEFT, B_THIN)
        sc(ws, r, 5, tax["Subtype"], F_BODY, None, A_LEFT, B_THIN)
        sc(ws, r, 6, f"{tax['TQA_Category']} | {AR_TQA.get(tax['TQA_Category'],'')}", F_BODY, tf, A_CENTER, B_THIN)
        sc(ws, r, 7, tax["Severity_Level"], F_BODY, None, A_CENTER, B_THIN)
        sc(ws, r, 8, tax["Severity_Score"], F_BODY, None, A_CENTER, B_THIN)
        sc(ws, r, 9, tax["Weight_WGT"], F_BODY, None, A_CENTER, B_THIN, '0.000')
        sc(ws, r, 10, tax["Category_Weight"], F_BODY, None, A_CENTER, B_THIN, '0.00')
        sc(ws, r, 11, aw, F_BODY, None, A_CENTER, B_THIN, '0.0000')
        sc(ws, r, 12, tax["Definition"], F_BODY, None, A_LEFT, B_THIN)
        r += 1
    return ws


# =============================================================================
# INSTRUCTIONS SHEET (bilingual)
# =============================================================================

def build_instructions(wb, phase_name, tier_label, sheet_name="Instructions"):
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = "607D8B"
    ws.column_dimensions['A'].width = 5
    ws.column_dimensions['B'].width = 55
    ws.column_dimensions['C'].width = 55

    for c in range(1, 4):
        sc(ws, 1, c, '', fill=FILL_BRAND_BAR)
    sc(ws, 1, 2, "رقيب  Raqeeb Evaluation Guide",
       Font(name='Arial', bold=True, size=16, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)

    r = 3
    def sect(en, ar="", title=False):
        nonlocal r
        f = Font(name='Arial', bold=True, size=13 if title else 11,
                 color=RAQEEB_NAVY if title else RAQEEB_TEAL)
        sc(ws, r, 2, en, f)
        if ar: sc(ws, r, 3, ar, Font(name='Arial', bold=True, size=13 if title else 11,
                                       color=RAQEEB_TEAL), alignment=A_RIGHT_AR)
        r += 1
    def ln(en, ar=""):
        nonlocal r
        sc(ws, r, 2, en, F_BODY, alignment=A_LEFT)
        if ar: sc(ws, r, 3, ar, F_BODY_AR, alignment=A_RIGHT_AR)
        r += 1
    def gap():
        nonlocal r; r += 1

    sect(f"INSTRUCTIONS — {phase_name} ({tier_label})",
         f"تعليمات التقييم — {phase_name}", title=True)
    gap()

    for c in [2, 3]:
        sc(ws, r, c, '', fill=PatternFill('solid', fgColor="FFCDD2"))
    sc(ws, r, 2, "⚠️ CRITICAL: Complete ALL of Layer 1 BEFORE opening Layer 2.",
       Font(name='Arial', bold=True, size=11, color="B71C1C"),
       PatternFill('solid', fgColor="FFCDD2"))
    sc(ws, r, 3, "⚠️ مهم: أكمل الطبقة 1 بالكامل قبل فتح الطبقة 2.",
       Font(name='Arial', bold=True, size=11, color="B71C1C"),
       PatternFill('solid', fgColor="FFCDD2"), A_RIGHT_AR)
    r += 2

    sect("LAYER 1 — BLIND SENTENCE EVALUATION", "الطبقة 1 — تقييم الجملة الأعمى")
    for en, ar in [
        ("1. Read Source (EN), Reference (AR), and MT Output.", "1. اقرأ المصدر والمرجع ومخرجات الترجمة."),
        ("2. Compare MT Output against the Reference.", "2. قارن المخرجات بالمرجع."),
        ("3. Fill: Error Found, Error Type, Severity, Confidence.", "3. املأ: وجود الخطأ، النوع، الحدّة، الثقة."),
        ("4. Rate: Accuracy, Fluency, Terminology, Style (1-5).", "4. قيّم: الدقة، الطلاقة، المصطلحات، الأسلوب."),
    ]: ln(en, ar)

    gap()
    sect("Severity Scale | مقياس الحدّة")
    for en, ar in [
        ("Minor — No meaning change", "طفيف — لا تغيير في المعنى"),
        ("Moderate — Minor meaning impact", "متوسط — تأثير طفيف"),
        ("Major — Clear meaning error", "كبير — خطأ دلالي واضح"),
        ("Critical — Complete distortion", "حرج — تشويه كامل"),
    ]: ln(en, ar)

    gap()
    sect("LAYER 2 — KEYWORD VALIDATION", "الطبقة 2 — التحقق من الكلمات المفتاحية")
    ln("System labels are now revealed. Validate:", "الآن تُكشف تصنيفات النظام. تحقق:")
    gap()
    for en, ar in [
        ("Q1 Error_Correct (0/1): Error present in mt_output?", "هل الخطأ موجود في المخرجات؟"),
        ("Q2 Anchor_Correct (0/1): Keyword→Match correct?", "هل الكلمة←التطابق صحيحة؟"),
        ("Q3 Severity: 0=None | 0.3=Minor | 0.7=Major | 1.0=Crit", "0=لا | 0.3=طفيف | 0.7=كبير | 1.0=حرج"),
        ("Q4 Minimal_Pair (0/1): Identical except the error?", "متطابقان عدا الخطأ المقصود؟"),
        ("Q5 Label_Correct (0/1): Sub-subtype correct?", "التصنيف الفرعي صحيح؟"),
        ("Q6 Naturalness (1-5): Arabic quality", "طبيعية العربية (1-5)"),
        ("Q7 UN_Appropriate (0/1): Fits UN register?", "مناسب للسجل الأممي؟"),
    ]: ln(en, ar)

    if tier_label == "Tier2":
        gap()
        sect("SYNTHETIC QUALITY (Tier 2)", "جودة التوليد (المستوى 2)")
        for en, ar in [
            ("S1 Fluency (1-5)", "طلاقة العربية"),
            ("S2 Realism (1-5)", "واقعية المثال"),
            ("S3 Domain (1-5)", "ملاءمة المجال"),
        ]: ln(en, ar)

    gap()
    sect("SCORING FORMULAS | صيغ الحساب")
    for f in [
        "Sentence_Score = (Acc×0.42 + Flu×0.225 + Term×0.28 + Style×0.075)/5 × 100",
        "KPS = MAX(0, 100 − MIN(100, Q3 × Weight × 20 / WordCount × 100))",
        "DQI = (Q1 + Q2 + Q4 + Q5) / 4  →  ≥0.75 = ACCEPT",
        "TRIVET_Score = Sentence_Score × 0.70 + KPS × 0.30",
    ]: ln(f)

    gap()
    sc(ws, r, 2, "© Raqeeb V4.0 | TRIVET Pipeline | EN→AR | UN Domain", F_BODY_SM)
    return ws


# =============================================================================
# HIDDEN AUDIT SHEET
# =============================================================================

def build_audit(wb, n_samples, sheet_name="_Audit"):
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = "F44336"
    ws.sheet_state = 'hidden'

    sc(ws, 1, 1, "AUDIT METADATA — DO NOT EDIT",
       Font(name='Arial', bold=True, size=12, color="B71C1C"))
    sc(ws, 4, 1, "Metric", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    sc(ws, 4, 2, "Value", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    ws.column_dimensions['A'].width = 35
    ws.column_dimensions['B'].width = 20

    last = n_samples + 3
    metrics = [
        ("L1 Completion Rate", f'=COUNTA(\'Raqeeb Layer 1\'!I4:I{last})/{n_samples}', '0.0%'),
        ("L2 Completion Rate", f'=COUNTA(\'Raqeeb Layer 2\'!M4:M{last})/{n_samples}', '0.0%'),
        ("L1 Mean Duration", f'=IFERROR(AVERAGE(\'Raqeeb Layer 1\'!S4:S{last}),"")', '0.0'),
        ("L2 Mean Duration", f'=IFERROR(AVERAGE(\'Raqeeb Layer 2\'!AH4:AH{last}),"")', '0.0'),
        ("L1 Duration StdDev", f'=IFERROR(STDEV(\'Raqeeb Layer 1\'!S4:S{last}),"")', '0.00'),
        ("L2 Mean Cog Load", f'=IFERROR(AVERAGE(\'Raqeeb Layer 2\'!AI4:AI{last}),"")', '0.0'),
        ("Error Found Rate", f'=IFERROR(COUNTIF(\'Raqeeb Layer 1\'!I4:I{last},"Yes")/{n_samples},"")', '0.0%'),
        ("Mean Sentence Score", f'=IFERROR(AVERAGE(\'Raqeeb Layer 1\'!Q4:Q{last}),"")', '0.0'),
        ("Mean DQI", f'=IFERROR(AVERAGE(\'Raqeeb Layer 2\'!Y4:Y{last}),"")', '0.00'),
        ("ACCEPT Rate", f'=IFERROR(COUNTIF(\'Raqeeb Layer 2\'!AB4:AB{last},"ACCEPT")/{n_samples},"")', '0.0%'),
        ("Mean KPS", f'=IFERROR(AVERAGE(\'Raqeeb Layer 2\'!W4:W{last}),"")', '0.0'),
        ("Mean Confidence", f'=IFERROR(AVERAGE(\'Raqeeb Layer 1\'!L4:L{last}),"")', '0.0'),
    ]
    for i, (name, formula, fmt) in enumerate(metrics):
        r = 5 + i
        sc(ws, r, 1, name, F_BODY, alignment=A_LEFT, border=B_THIN)
        sc(ws, r, 2, formula, F_BODY, FILL_AUTO, A_CENTER, B_THIN, fmt)
    return ws


# =============================================================================
# MASTER BUILDER
# =============================================================================

def build_workbook(samples_df, phase_name, annotator_id, tier_label,
                   output_path, etca_lookup=None, n_gold=0, n_classes=0):
    wb = Workbook()
    wb.remove(wb.active)
    build_dashboard(wb, phase_name, annotator_id, tier_label, len(samples_df),
                    n_gold, n_classes)
    build_layer1(wb, samples_df)
    build_layer2(wb, samples_df, tier_label, etca_lookup)
    build_taxonomy(wb)
    build_instructions(wb, phase_name, tier_label)
    build_audit(wb, len(samples_df))
    wb.save(output_path)
    print(f"   ✅ {os.path.basename(output_path)} ({len(samples_df)} samples)")
    return output_path


# =============================================================================
# BUILD ETCA LOOKUP
# =============================================================================

etca_lookup = {}
if not df_etca.empty:
    for _, row in df_etca.iterrows():
        sid = row.get('Sentence_ID', '')
        if sid:
            etca_lookup[sid] = {k: row.get(k, '') for k in
                ['anchor_validity', 'phenomenon_clarity', 'arabic_naturalness',
                 'collateral_severity', 'tier2_seed_recommendation_final']}
    print(f"\n✅ ETCA lookup: {len(etca_lookup):,} entries")

# Class counts for dashboard
t1_n_cls = int(t1_all[ERROR_COL].nunique()) if ERROR_COL in t1_all.columns and not t1_all.empty else 0
t2_n_cls = int(t2_all[ERROR_COL].nunique()) if ERROR_COL in t2_all.columns and not t2_all.empty else 0

# =============================================================================
# GENERATE ALL 19 WORKBOOKS
# =============================================================================

generated_files = []

# Training
print("\n--- Training ---")
for tl, df in [("Tier1", t1_training), ("Tier2", t2_training)]:
    if df.empty: continue
    p = f"{EVAL_DIR}/TRIVET_Training_{tl}.xlsx"
    build_workbook(df, "Training (Calibration)", "BOTH", tl, p,
                   etca_lookup if tl == "Tier1" else None, 0,
                   t1_n_cls if tl == "Tier1" else t2_n_cls)
    generated_files.append(p)

# Phase 1: Gold
print("\n--- Phase 1: Gold ---")
for tl, gdf, nc in [("Tier1", t1_gold, t1_n_cls), ("Tier2", t2_gold, t2_n_cls)]:
    if gdf.empty: continue
    gs = gdf.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    for ann in ["A", "B"]:
        p = f"{EVAL_DIR}/TRIVET_Phase1_Gold_{tl}_Annotator_{ann}.xlsx"
        build_workbook(gs, "Phase 1 — Gold Overlap", f"Annotator {ann}", tl, p,
                       etca_lookup if tl == "Tier1" else None, len(gs), nc)
        generated_files.append(p)

# Phase 3: Batches
print("\n--- Phase 3: Batches ---")
for tl, ba, bb, nc in [("Tier1", t1_batch_a, t1_batch_b, t1_n_cls),
                         ("Tier2", t2_batch_a, t2_batch_b, t2_n_cls)]:
    for ann, bdf in [("A", ba), ("B", bb)]:
        if bdf.empty: continue
        bs = bdf.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        p = f"{EVAL_DIR}/TRIVET_Phase3_Batch_{tl}_Annotator_{ann}.xlsx"
        build_workbook(bs, "Phase 3 — Unique Batch", f"Annotator {ann}", tl, p,
                       etca_lookup if tl == "Tier1" else None, 0, nc)
        generated_files.append(p)

# Phase 4: Review templates
print("\n--- Phase 4: Review Templates ---")
for tl, ba, bb in [("Tier1", t1_batch_a, t1_batch_b),
                     ("Tier2", t2_batch_a, t2_batch_b)]:
    for reviewer, orig in [("A", bb), ("B", ba)]:
        if orig.empty: continue
        wb = Workbook()
        wb.remove(wb.active)
        build_dashboard(wb, "Phase 4 — Cross-Review", f"Reviewer {reviewer}", tl, len(orig))
        ws = wb.create_sheet("Review Layer 1")
        sc(ws, 1, 1, "Phase 4 — Original answers shown + review columns",
           Font(name='Arial', bold=True, size=13, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)
        sc(ws, 3, 1, "[Rows populated after Phase 3 collection]",
           Font(name='Arial', italic=True, color="999999"))
        ws2 = wb.create_sheet("Review Layer 2")
        sc(ws2, 1, 1, "Phase 4 — Layer 2 review",
           Font(name='Arial', bold=True, size=13, color="FFFFFF"), FILL_BRAND_BAR, A_LEFT)
        sc(ws2, 3, 1, "[Rows populated after Phase 3 collection]",
           Font(name='Arial', italic=True, color="999999"))
        build_taxonomy(wb)
        build_instructions(wb, "Phase 4 — Cross-Review", tl)
        p = f"{EVAL_DIR}/TRIVET_Phase4_Review_{tl}_Reviewer_{reviewer}.xlsx"
        wb.save(p)
        print(f"   ✅ {os.path.basename(p)} (template)")
        generated_files.append(p)

# Phase 5-6: Adjudication
print("\n--- Phase 5-6: Adjudication ---")
for tl, gdf in [("Tier1", t1_gold), ("Tier2", t2_gold)]:
    if gdf.empty: continue
    wb = Workbook()
    wb.remove(wb.active)
    build_dashboard(wb, "Phase 5-6 — Adjudication", "BOTH", tl, len(gdf))
    ws = wb.create_sheet("Adjudication")
    for c, h in enumerate(['Eval_ID', 'Sub-Subtype', 'A_Error', 'B_Error', 'Agree?',
        'A_Severity', 'B_Severity', 'A_MQM_Acc', 'B_MQM_Acc',
        'Final_Error', 'Final_Severity', 'Final_MQM_Acc', 'Resolution_Notes'], 1):
        sc(ws, 1, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    for i, (_, row) in enumerate(gdf.iterrows()):
        sc(ws, i+2, 1, row.get('eval_id',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, i+2, 2, str(row.get(ERROR_COL,'')), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        for c in range(3, 14):
            sc(ws, i+2, c, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
    ws_gt = wb.create_sheet("Ground Truth")
    for c, h in enumerate(['Eval_ID','Sub-Subtype','Final_Error','Final_Type',
        'Final_Severity','Final_MQM_Acc','Final_MQM_Flu','Final_MQM_Term',
        'Final_MQM_Style','Final_SentScore','Final_Q1','Final_Q2','Final_Q3',
        'Final_Q4','Final_Q5','Final_Q6','Final_Q7','Final_KPS','Final_DQI','Final_Status'], 1):
        sc(ws_gt, 1, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    for i, (_, row) in enumerate(gdf.iterrows()):
        sc(ws_gt, i+2, 1, row.get('eval_id',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws_gt, i+2, 2, str(row.get(ERROR_COL,'')), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
    build_taxonomy(wb)
    p = f"{EVAL_DIR}/TRIVET_Phase56_Adjudication_{tl}.xlsx"
    wb.save(p)
    print(f"   ✅ {os.path.basename(p)} ({len(gdf)} gold)")
    generated_files.append(p)

# IAA Reports
print("\n--- IAA Reports ---")
for tl, gdf in [("Tier1", t1_gold), ("Tier2", t2_gold)]:
    if gdf.empty: continue
    wb = Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("IAA Data")
    for c, h in enumerate(['Eval_ID','Sub-Subtype',
        'A_Error','B_Error','A_Sev','B_Sev','A_MQM_Acc','B_MQM_Acc',
        'A_MQM_Flu','B_MQM_Flu','A_Q1','B_Q1','A_Q2','B_Q2','A_Q3','B_Q3',
        'A_Q4','B_Q4','A_Q5','B_Q5','A_Q6','B_Q6','A_Q7','B_Q7'], 1):
        sc(ws, 1, c, h, F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    for i, (_, row) in enumerate(gdf.iterrows()):
        sc(ws, i+2, 1, row.get('eval_id',''), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
        sc(ws, i+2, 2, str(row.get(ERROR_COL,'')), F_BODY, FILL_PREFILL, A_CENTER, B_THIN)
    wss = wb.create_sheet("IAA Statistics")
    for i, (m, _, interp) in enumerate([
        ("Cohen's κ (Binary Error)", "", ""), ("Cohen's κ (Severity)", "", ""),
        ("Weighted κ (MQM Accuracy)", "", ""), ("Spearman ρ (Sentence Score)", "", ""),
        ("ICC (KPS)", "", ""), ("ICC (TRIVET Score)", "", ""),
        ("% Agreement (Q1)", "", ""), ("% Agreement (Q4)", "", ""),
        ("% Agreement (Q5)", "", ""),
    ]):
        r = i + 2
        sc(wss, r, 1, m, F_BODY, alignment=A_LEFT, border=B_THIN)
        sc(wss, r, 2, '', F_BODY, FILL_ANNOTATOR, A_CENTER, B_THIN)
        sc(wss, r, 3, '', F_BODY, alignment=A_LEFT, border=B_THIN)
    sc(wss, 1, 1, "Metric", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    sc(wss, 1, 2, "Value", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    sc(wss, 1, 3, "Interpretation", F_HEADER, FILL_HEADER, A_CENTER, B_HEADER)
    ws.column_dimensions['A'].width = 12
    wss.column_dimensions['A'].width = 30
    wss.column_dimensions['B'].width = 15
    wss.column_dimensions['C'].width = 30
    p = f"{EVAL_DIR}/TRIVET_IAA_Report_{tl}.xlsx"
    wb.save(p)
    print(f"   ✅ {os.path.basename(p)}")
    generated_files.append(p)

# Combined Results
print("\n--- Combined Results ---")
all_s = pd.concat([t1_all, t2_all], ignore_index=True)
if not all_s.empty:
    p = f"{EVAL_DIR}/TRIVET_Combined_Results.xlsx"
    build_workbook(all_s, "Combined Results (Master)", "ALL", "Combined", p, etca_lookup,
                   n_classes=23)
    generated_files.append(p)

# =============================================================================
# UPDATE METADATA & FINAL SUMMARY
# =============================================================================

metadata['files_generated'] = generated_files
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"""
{'='*80}
CELL 15.2 COMPLETE — ALL FILES GENERATED
{'='*80}

📁 Output: {EVAL_DIR}/
📄 Files: {len(generated_files)}
""")
for f in generated_files:
    print(f"   📄 {os.path.basename(f)}")

print(f"""
📊 Sampling:
   Tier 1: {len(t1_all)} ({len(t1_gold)} gold + {len(t1_batch_a)} A + {len(t1_batch_b)} B) | {t1_n_cls}/23 classes
   Tier 2: {len(t2_all)} ({len(t2_gold)} gold + {len(t2_batch_a)} A + {len(t2_batch_b)} B) | {t2_n_cls}/23 classes
   Training: {len(t1_training)} T1 + {len(t2_training)} T2
   Total unique: {len(t1_all)+len(t2_all)}

🎨 Professional features:
   ✓ Raqeeb branding (navy/teal/gold)
   ✓ Bilingual headers (EN/AR)
   ✓ Conditional formatting (severity, scores, status)
   ✓ Data validations with Arabic prompts
   ✓ Hidden _Audit sheet (12 metrics)
   ✓ Taxonomy tree + reference table
   ✓ Color-coded TQA categories
   ✓ Arabic RTL alignment

✅ Ready for annotation.
""")
