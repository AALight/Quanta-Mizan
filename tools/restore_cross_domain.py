#!/usr/bin/env python
"""
Restore the full cross-domain TRIVET validation files from the extraData/ backup,
enriching them with the domain column via a join against trivet_input.csv.

This reverses the corruption where tier1_clean_A.csv was overwritten with a
5-candidate smoke-test output on Apr 14.

Run once from project root:
    python tools/restore_cross_domain.py
"""
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
BACKUP = PROJ / "extraData"
TARGET = PROJ / "cross_domain_validation" / "tier1_v3_3_cross_domain"
TRIVET_INPUT = PROJ / "cross_domain_validation" / "trivet_input.csv"

# Build domain lookup keyed on (Keyword, ar) -> domain
trivet_in = pd.read_csv(TRIVET_INPUT)
domain_lookup = (
    trivet_in[["keyword", "clean_ar", "domain"]]
    .drop_duplicates(subset=["keyword", "clean_ar"])
    .rename(columns={"keyword": "Keyword", "clean_ar": "ar"})
)
print(f"domain lookup: {len(domain_lookup)} unique (Keyword, ar) keys")

for stem in ["tier1_clean_A", "tier1_clean_B", "tier1_noisy"]:
    src = BACKUP / f"{stem}.csv"
    dst = TARGET / f"{stem}.csv"
    df = pd.read_csv(src)
    before = len(df)
    enriched = df.merge(domain_lookup, on=["Keyword", "ar"], how="left")
    n_missing = enriched["domain"].isna().sum()
    if n_missing > 0:
        print(f"  WARN {stem}: {n_missing} rows have no domain, filling 'unknown'")
        enriched["domain"] = enriched["domain"].fillna("unknown")
    enriched.to_csv(dst, index=False)
    print(f"  wrote {dst.relative_to(PROJ)}: {len(enriched)} rows with domain")

# Rebuild tier1_clean.csv as A+B combined
a = pd.read_csv(TARGET / "tier1_clean_A.csv")
b = pd.read_csv(TARGET / "tier1_clean_B.csv")
combined = pd.concat([a, b], ignore_index=True)
combined.to_csv(TARGET / "tier1_clean.csv", index=False)
print(f"  wrote tier1_clean.csv: {len(combined)} rows (A+B combined)")

# Update validation_stats.json
import json
n_clean_a = len(a)
n_clean_b = len(b)
n_noisy = len(pd.read_csv(TARGET / "tier1_noisy.csv"))
total = n_clean_a + n_clean_b + n_noisy
stats = {
    "total_candidates": total,
    "clean": n_clean_a + n_clean_b,
    "clean_a": n_clean_a,
    "clean_b": n_clean_b,
    "noisy": n_noisy,
    "discarded": 0,
    "clean_rate": round((n_clean_a + n_clean_b) / total, 4),
    "clean_a_rate": round(n_clean_a / total, 4),
    "clean_b_rate": round(n_clean_b / total, 4),
    "noisy_rate": round(n_noisy / total, 4),
    "discard_rate": 0.0,
}
with open(TARGET / "validation_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  wrote validation_stats.json: total={total}, clean_rate={stats['clean_rate']}")

print("\nRestore complete.")
