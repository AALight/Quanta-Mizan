"""
Data preparation script for Raqeep classifier training.

Usage:
    python prepare_splits.py

    Run from any directory — the script auto-detects its location
    and creates all needed folders and files automatically.

    Note: processed files are already in data/processed/. Run this only to
    next to this script (or in the repo's data/raw/):
      - gold_446_manual_curated.csv
      - tier1_clean_A_FROZEN.csv
      - tier2_generated_raw.csv
      - candidates_tiered_ALL.csv
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_FOLDS = 5

# Auto-detect base directory (where this script lives)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Look for source files in data/processed/
if (SCRIPT_DIR / "data" / "raw").exists():
    BASE_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent.parent / "data" / "raw").exists():
    BASE_DIR = SCRIPT_DIR.parent.parent
else:
    BASE_DIR = SCRIPT_DIR

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"

# Standard column names
STANDARD_COLS = [
    "sample_id", "Sentence_ID", "en", "ar", "mt_output",
    "Sub-Subtype", "Keyword", "Best_Match", "source_tier"
]

# ============================================================
# EMBEDDED LABEL MAP (no external file needed)
# ============================================================

LABEL_MAP = {
    "Meaning Shift": 0,
    "Total Omission": 1,
    "Partial Translation": 2,
    "Name Entity Error": 3,
    "Terminology Substitution": 4,
    "Literal Translation": 5,
    "Hypernym for Hyponym": 6,
    "Hyponym for Hypernym": 7,
    "Tanween Omission": 8,
    "Gender Disagreement": 9,
    "Definiteness Shift": 10,
    "Perfective to Progressive": 11,
    "Progressive to Perfective": 12,
    "Tense Shift Under Negation": 13,
    "Wrong Structure": 14,
    "Wrong Word Order": 15,
    "Active to Passive Voice": 16,
    "Passive to Active Voice": 17,
    "Noun to Adjective": 18,
    "Adjective to Noun": 19,
    "Register Mismatch": 20,
    "Spelling Error": 21,
    "Invalid Pattern": 22,
}

PARENT_MAP = {
    "Accuracy": [
        "Meaning Shift", "Total Omission", "Partial Translation",
        "Name Entity Error", "Terminology Substitution", "Literal Translation",
        "Hypernym for Hyponym", "Hyponym for Hypernym"
    ],
    "Fluency_Morphological": [
        "Tanween Omission", "Gender Disagreement", "Definiteness Shift",
        "Perfective to Progressive", "Progressive to Perfective"
    ],
    "Fluency_Syntactic": [
        "Tense Shift Under Negation", "Wrong Structure", "Wrong Word Order",
        "Active to Passive Voice", "Passive to Active Voice"
    ],
    "Terminology": [
        "Noun to Adjective", "Adjective to Noun", "Register Mismatch",
        "Spelling Error", "Invalid Pattern"
    ],
}


# ============================================================
# LABEL NORMALIZATION (old names → canonical names)
# ============================================================

LABEL_NORMALIZE = {
    "Definiteness":                "Definiteness Shift",
    "Definiteness Shift":          "Definiteness Shift",
    "Formality Level":             "Register Mismatch",
    "Register Mismatch":           "Register Mismatch",
    "Terminological Substitution": "Terminology Substitution",
    "Terminology Substitution":    "Terminology Substitution",
    "Negation":                    "Tense Shift Under Negation",
    "Tense shift under negation":  "Tense Shift Under Negation",
    "Tense Shift Under Negation":  "Tense Shift Under Negation",
    "Wrong Word Order ":           "Wrong Word Order",   # trailing space
    "Adjective to Noun ":          "Adjective to Noun",  # trailing space
    "Total Omission ":             "Total Omission",     # trailing space
}

def normalize_labels(df: pd.DataFrame, col: str = "Sub-Subtype") -> pd.DataFrame:
    """Normalize old/variant label names to canonical 23-class names.
    
    Handles hidden Unicode characters, trailing spaces, and mixed encodings
    that cause str.replace() to silently fail.
    """
    if col not in df.columns:
        return df

    # Step 1: force to string
    s = df[col].astype(str)

    # Step 2: strip all whitespace including non-breaking spaces (\xa0)
    s = s.str.strip().str.replace(r'\s+', ' ', regex=True)

    # Step 3: drop invisible non-ASCII characters (zero-width spaces etc.)
    s = s.apply(lambda x: ''.join(c for c in x if c.isprintable()))

    # Step 4: strip again after cleaning
    s = s.str.strip()

    # Step 5: apply the normalization map
    s = s.replace(LABEL_NORMALIZE)

    # Step 6: second pass with stripped keys to catch any remaining variants
    strip_map = {k.strip(): v for k, v in LABEL_NORMALIZE.items()}
    s = s.str.strip().replace(strip_map)

    df[col] = s
    return df


# ============================================================
# LOADING FUNCTIONS
# ============================================================

def load_gold_446(path: Path) -> pd.DataFrame:
    """Load expert gold test set (438 curated samples after quality audit)."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Standardize columns
    df = df.rename(columns={
        "MT Tool": "mt_tool",
        "Linguistic Shift Type": "shift_type",
        "Subtype": "subtype",
        "Severity_Level": "severity_level",
        "Severity_Score": "severity_score",
        "SubSubtype_WGT": "subsubtype_wgt",
        "TQA_Category": "tqa_category",
        "Shift_Group_Weight": "shift_group_weight",
        "TQA_Category_Weight": "tqa_category_weight",
    })

    df["source_tier"] = "gold_446"
    df["sample_id"] = [f"GOLD_{i:04d}" for i in range(len(df))]

    # Normalize labels (old names → canonical)
    df = normalize_labels(df)

    return df


def load_tier1(path: Path) -> pd.DataFrame:
    """Load Tier 1 clean validated real MT errors."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_tier"] = "tier1"
    df["sample_id"] = [f"T1_{i:04d}" for i in range(len(df))]
    df = normalize_labels(df)
    return df


def load_tier2(path: Path) -> pd.DataFrame:
    """Load Tier 2 synthetic generated samples."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_tier"] = "tier2"
    df["sample_id"] = [f"T2_{i:04d}" for i in range(len(df))]
    df = normalize_labels(df)
    return df


# ============================================================
# VALIDATION
# ============================================================

def verify_no_overlap(gold: pd.DataFrame, t1: pd.DataFrame, t2: pd.DataFrame):
    """Ensure gold_446 has zero sentence overlap with training data."""
    gold_sids = set(gold["Sentence_ID"].dropna().unique())
    t1_sids = set(t1["Sentence_ID"].dropna().unique())
    t2_sids = set(t2["Sentence_ID"].dropna().unique())

    overlap_t1 = gold_sids & t1_sids
    overlap_t2 = gold_sids & t2_sids

    if overlap_t1:
        print(f"⚠️  WARNING: {len(overlap_t1)} Sentence_IDs overlap between gold_446 and T1!")
        print(f"   Overlapping IDs: {list(overlap_t1)[:10]}...")
        print(f"   ACTION: Removing these from T1.")

    if overlap_t2:
        print(f"⚠️  WARNING: {len(overlap_t2)} Sentence_IDs overlap between gold_446 and T2!")
        print(f"   Overlapping IDs: {list(overlap_t2)[:10]}...")
        print(f"   ACTION: Removing these from T2.")

    if not overlap_t1 and not overlap_t2:
        print("✅ No overlap between gold_446 and T1/T2.")

    return overlap_t1, overlap_t2


def verify_label_coverage(df: pd.DataFrame, name: str, label_map: dict):
    """Check all 23 classes are present."""
    present = set(df["Sub-Subtype"].unique())
    expected = set(label_map.keys())
    missing = expected - present
    extra = present - expected

    print(f"\n--- {name} ---")
    print(f"   Samples: {len(df)}")
    print(f"   Classes: {len(present & expected)}/23")

    if missing:
        print(f"   ⚠️  Missing classes: {missing}")
    if extra:
        print(f"   ⚠️  Unknown classes: {extra}")

    counts = df["Sub-Subtype"].value_counts()
    valid_counts = counts[counts.index.isin(expected)]
    if len(valid_counts) > 0:
        print(f"   Min per class: {valid_counts.min()} ({valid_counts.idxmin()})")
        print(f"   Max per class: {valid_counts.max()} ({valid_counts.idxmax()})")


# ============================================================
# SPLIT CREATION
# ============================================================

def create_stratified_group_splits(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    name: str
) -> dict:
    """Create GroupStratifiedKFold splits ensuring no Sentence_ID leakage."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    labels = df["Sub-Subtype"].values
    groups = df["Sentence_ID"].values

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        splits[f"fold_{fold}"] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist()
        }
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups), \
            f"FATAL: Fold {fold} has Sentence_ID leakage!"

    print(f"\n✅ {name}: {n_folds}-fold splits created (no Sentence_ID leakage)")
    for fold in range(n_folds):
        n_train = len(splits[f"fold_{fold}"]["train"])
        n_val = len(splits[f"fold_{fold}"]["val"])
        print(f"   Fold {fold}: Train {n_train} | Val {n_val}")

    return splits


def compute_split_statistics(df: pd.DataFrame, splits: dict) -> dict:
    """Compute per-fold, per-class statistics."""
    stats = {}
    for fold_name, fold_data in splits.items():
        train_df = df.iloc[fold_data["train"]]
        val_df = df.iloc[fold_data["val"]]
        stats[fold_name] = {
            "train_total": len(train_df),
            "val_total": len(val_df),
            "train_classes": train_df["Sub-Subtype"].nunique(),
            "val_classes": val_df["Sub-Subtype"].nunique(),
            "train_distribution": train_df["Sub-Subtype"].value_counts().to_dict(),
            "val_distribution": val_df["Sub-Subtype"].value_counts().to_dict(),
        }
    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("RAQEEP CLASSIFIER — DATA PREPARATION")
    print("=" * 70)

    for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Base directory: {BASE_DIR}")
    print(f"Raw data:       {RAW_DIR}")
    print(f"Processed:      {PROCESSED_DIR}")
    print(f"Splits:         {SPLITS_DIR}")

    label_map = LABEL_MAP
    print(f"Label map: {len(label_map)} classes (embedded)")

    with open(PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(PROCESSED_DIR / "parent_map.json", "w") as f:
        json.dump(PARENT_MAP, f, indent=2)
    print(f"   Saved: label_map.json, parent_map.json")

    # ---- Check required files ----
    required_files = {
        "gold_446_manual_curated.csv": "Expert gold test set (438 samples after audit)",
        "tier1_clean_A_FROZEN.csv": "Tier 1 validated real MT errors",
        "tier2_generated_raw.csv": "Tier 2 synthetic samples",
    }
    optional_files = {
        "candidates_tiered_ALL.csv": "Full candidates (for No-Error extraction)",
    }

    missing = []
    for fname, desc in required_files.items():
        if not (RAW_DIR / fname).exists():
            missing.append(f"   ❌ {fname} — {desc}")

    if missing:
        print(f"\n🚨 MISSING REQUIRED FILES in {RAW_DIR}/:")
        for m in missing:
            print(m)
        print(f"\nPlace the files in: {RAW_DIR}/")
        sys.exit(1)

    for fname, desc in optional_files.items():
        if not (RAW_DIR / fname).exists():
            print(f"\n⚠️  Optional file not found: {fname}")
            print(f"   {desc}")
            print(f"   No-Error extraction will be skipped.")

    # ---- Load data ----
    print("\n📂 Loading data...")

    # Gold — single canonical filename only
    GOLD_PATH = RAW_DIR / "gold_446_manual_curated.csv"
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold file not found: {GOLD_PATH}\n"
            f"Place gold_446_manual_curated.csv in {RAW_DIR}/"
        )
    gold = load_gold_446(GOLD_PATH)
    print(f"   gold:     {len(gold)} samples ← {GOLD_PATH.name}")

    # Tier 1
    T1_CANDIDATES = [
        RAW_DIR / "tier1_clean_A_FROZEN.csv",
        RAW_DIR / "tier1_clean1.csv",
    ]
    t1 = None
    for t1_path in T1_CANDIDATES:
        if t1_path.exists():
            t1 = load_tier1(t1_path)
            print(f"   Tier 1:   {len(t1)} samples ← {t1_path.name}")
            break
    if t1 is None:
        raise FileNotFoundError(f"No Tier 1 file found! Tried: {[p.name for p in T1_CANDIDATES]}")

    # Tier 2
    TIER2_CANDIDATES = [
        # Raw source file alternatives (any one of these is accepted)
        RAW_DIR / "tier2_generated_enriched.csv",
        RAW_DIR / "tier2_generated_raw.csv",
        RAW_DIR / "tier2_llm_synthetic.csv",
        RAW_DIR / "TIER2_GENERATED_FINAL.csv",
        RAW_DIR / "tier2_synthetic_v1.csv",
    ]
    t2 = None
    for t2_path in TIER2_CANDIDATES:
        if t2_path.exists():
            t2 = load_tier2(t2_path)
            print(f"   Tier 2:   {len(t2)} samples ← {t2_path.name}")
            break
    if t2 is None:
        raise FileNotFoundError(f"No Tier 2 file found! Tried: {[p.name for p in TIER2_CANDIDATES]}")

    # ---- Verify no overlap ----
    print("\n🔍 Checking for overlap...")
    overlap_t1, overlap_t2 = verify_no_overlap(gold, t1, t2)

    if overlap_t1:
        t1 = t1[~t1["Sentence_ID"].isin(overlap_t1)]
        print(f"   T1 after removal: {len(t1)}")
    if overlap_t2:
        t2 = t2[~t2["Sentence_ID"].isin(overlap_t2)]
        print(f"   T2 after removal: {len(t2)}")

    # ---- Verify label coverage ----
    print("\n📊 Label coverage:")
    verify_label_coverage(gold, "gold_446 (TEST)", label_map)
    verify_label_coverage(t1, "Tier 1 (TRANSFER TEST)", label_map)
    verify_label_coverage(t2, "Tier 2 (TRAINING)", label_map)

    # ---- Add numeric labels ----
    for df in [gold, t1, t2]:
        df["label_id"] = df["Sub-Subtype"].map(label_map)
        unmapped = df["label_id"].isna().sum()
        if unmapped > 0:
            bad_labels = df[df["label_id"].isna()]["Sub-Subtype"].unique()
            print(f"⚠️  {unmapped} samples with unmapped labels: {bad_labels}")

    # ---- Extract No-Error samples from candidates ----
    print("\n🔍 Extracting No-Error samples from candidates...")

    CANDIDATES_PATHS = [
        RAW_DIR / "candidates_tiered_ALL.csv",
        # Raw candidates file alternatives
        RAW_DIR / "candidates_tiered_all.csv",
        RAW_DIR / "candidates_ALL.csv",
    ]

    CANDIDATES_PATH = None
    for cp in CANDIDATES_PATHS:
        if cp.exists():
            CANDIDATES_PATH = cp
            break

    NO_ERROR_TARGET = len(t2)

    if CANDIDATES_PATH is not None:
        cand = pd.read_csv(CANDIDATES_PATH, encoding="utf-8-sig")
        cand = normalize_labels(cand)
        print(f"   Candidates loaded: {len(cand)} rows (labels normalized)")

        omission_ne = cand[
            cand["retrieval_signal"] == "OMISSION_CONTEXT_REF_HAS_KW_MT_HAS_KW"
        ].copy()
        print(f"   Omission No-Error (raw): {len(omission_ne)}")

        sub_ne_raw = cand[
            cand["retrieval_signal"] == "REF_HAS_KEYWORD_MT_MISSING_BESTMATCH"
        ].copy()

        sub_ne_raw["_kw_in_mt"] = sub_ne_raw.apply(
            lambda r: str(r["Keyword"]).strip() in str(r["mt_output"]), axis=1
        )
        sub_ne_verified = sub_ne_raw[sub_ne_raw["_kw_in_mt"]].drop(columns=["_kw_in_mt"])
        sub_ne_rejected = len(sub_ne_raw) - len(sub_ne_verified)

        print(f"   Substitution No-Error (raw): {len(sub_ne_raw)}")
        print(f"   Substitution No-Error (keyword verified in MT): {len(sub_ne_verified)}")
        print(f"   Substitution rejected (keyword NOT in MT): {sub_ne_rejected}")

        all_no_error = pd.concat([omission_ne, sub_ne_verified], ignore_index=True)
        all_no_error["source_tier"] = "no_error"
        print(f"   Total No-Error pool: {len(all_no_error)}")

        per_class_target = NO_ERROR_TARGET // 23
        remainder = NO_ERROR_TARGET % 23

        no_error_sampled = []

        for cls in sorted(label_map.keys()):
            cls_pool = all_no_error[all_no_error["Sub-Subtype"] == cls]
            n_sample = per_class_target + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)

            if len(cls_pool) >= n_sample:
                sampled = cls_pool.sample(n=n_sample, random_state=SEED)
            elif len(cls_pool) > 0:
                sampled = cls_pool
                print(f"   ⚠️  {cls}: only {len(cls_pool)} No-Error available (needed {n_sample})")
            else:
                print(f"   ❌ {cls}: NO No-Error samples available!")
                continue

            no_error_sampled.append(sampled)

        no_error_df = pd.concat(no_error_sampled, ignore_index=True)

        ne_sids = set(no_error_df["Sentence_ID"].dropna().unique())
        ne_gold_overlap = ne_sids & set(gold["Sentence_ID"].dropna().unique())
        if ne_gold_overlap:
            print(f"   ⚠️  Removing {len(ne_gold_overlap)} No-Error samples overlapping with gold_446")
            no_error_df = no_error_df[~no_error_df["Sentence_ID"].isin(ne_gold_overlap)]

        no_error_df["sample_id"] = [f"NE_{i:04d}" for i in range(len(no_error_df))]
        no_error_df["binary_label"] = "No-Error"

        print(f"\n   ✅ No-Error sampled: {len(no_error_df)} (target was {NO_ERROR_TARGET})")
        print(f"   Classes represented: {no_error_df['Sub-Subtype'].nunique()}/23")
        print(f"   Per-class range: {no_error_df['Sub-Subtype'].value_counts().min()}"
              f"–{no_error_df['Sub-Subtype'].value_counts().max()}")

        if "retrieval_signal" in no_error_df.columns:
            sig_counts = no_error_df["retrieval_signal"].value_counts()
            for sig, cnt in sig_counts.items():
                print(f"   {sig}: {cnt}")
    else:
        print(f"   ⚠️  No candidates file found — skipping No-Error extraction")
        print(f"   Tried: {[p.name for p in CANDIDATES_PATHS]}")
        no_error_df = pd.DataFrame()

    # ---- Save processed files ----
    print("\n💾 Saving processed files...")

    gold.to_csv(PROCESSED_DIR / "test_gold.csv", index=False, encoding="utf-8-sig")
    t1.to_csv(PROCESSED_DIR / "test_transfer.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(PROCESSED_DIR / "train_t2.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat([t1, t2], ignore_index=True)
    combined.to_csv(PROCESSED_DIR / "train_t1t2.csv", index=False, encoding="utf-8-sig")

    print(f"   ✅ test_gold.csv: {len(gold)} samples")
    print(f"   ✅ test_transfer.csv: {len(t1)} samples")
    print(f"   ✅ train_t2.csv: {len(t2)} samples")
    print(f"   ✅ train_t1t2.csv: {len(combined)} samples")

    if len(no_error_df) > 0:
        no_error_df.to_csv(PROCESSED_DIR / "no_error_train.csv", index=False, encoding="utf-8-sig")
        print(f"   ✅ no_error_train.csv: {len(no_error_df)} samples")

        t2_with_binary = t2.copy()
        t2_with_binary["binary_label"] = "Error"

        binary_train = pd.concat([t2_with_binary, no_error_df], ignore_index=True)
        binary_train.to_csv(PROCESSED_DIR / "binary_train.csv", index=False, encoding="utf-8-sig")
        print(f"   ✅ binary_train.csv: {len(binary_train)} samples "
              f"({len(t2_with_binary)} Error + {len(no_error_df)} No-Error)")

    # ---- Create splits ----
    print("\n🔀 Creating cross-validation splits...")

    t2_splits = create_stratified_group_splits(t2, N_FOLDS, SEED, "T2")
    t1t2_splits = create_stratified_group_splits(combined, N_FOLDS, SEED, "T1+T2")

    t2_stats = compute_split_statistics(t2, t2_splits)
    t1t2_stats = compute_split_statistics(combined, t1t2_splits)

    with open(SPLITS_DIR / "t2_5fold_splits.json", "w") as f:
        json.dump(t2_splits, f, indent=2)

    with open(SPLITS_DIR / "t1t2_5fold_splits.json", "w") as f:
        json.dump(t1t2_splits, f, indent=2)

    with open(SPLITS_DIR / "split_statistics.json", "w") as f:
        json.dump({
            "t2_splits": t2_stats,
            "t1t2_splits": t1t2_stats,
            "gold_446": {
                "total": len(gold),
                "classes": gold["Sub-Subtype"].nunique(),
                "distribution": gold["Sub-Subtype"].value_counts().to_dict()
            },
            "tier1_transfer": {
                "total": len(t1),
                "classes": t1["Sub-Subtype"].nunique(),
                "distribution": t1["Sub-Subtype"].value_counts().to_dict()
            }
        }, f, indent=2)

    print(f"\n   ✅ t2_5fold_splits.json")
    print(f"   ✅ t1t2_5fold_splits.json")
    print(f"   ✅ split_statistics.json")

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gold test set:     {len(gold):>6} samples ({gold['Sub-Subtype'].nunique()}/23 classes)")
    print(f"  Transfer test set: {len(t1):>6} samples ({t1['Sub-Subtype'].nunique()}/23 classes)")
    print(f"  Training (T2):     {len(t2):>6} samples ({t2['Sub-Subtype'].nunique()}/23 classes)")
    print(f"  Training (T1+T2):  {len(combined):>6} samples ({combined['Sub-Subtype'].nunique()}/23 classes)")
    if len(no_error_df) > 0:
        print(f"  No-Error:          {len(no_error_df):>6} samples ({no_error_df['Sub-Subtype'].nunique()}/23 source classes)")
        print(f"  Binary training:   {len(t2) + len(no_error_df):>6} samples (1:1 balanced)")
    print(f"  CV folds:          {N_FOLDS}")
    print(f"  Random seed:       {SEED}")
    print(f"\n✅ Data preparation complete.")
    print(f"\n📋 NEXT STEP: Run validation")
    print(f"   python src/data/validate_data.py")
    print(f"\n📋 THEN: Upload ai_check_*.csv files to Claude for spot-checking")


if __name__ == "__main__":
    main()
