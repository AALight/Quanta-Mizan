# ETCA Audits

ETCA (Error Taxonomy Cross-vendor Audit) is a cross-vendor LLM-as-judge quality audit
(GPT-4o generates; Claude Sonnet 4 judges). This folder holds the two ETCA runs reported
in the paper. **Neither is the T2 selection gate** — the released 7,856 T2 instances are
selected by ESV structural validation plus the Partial-Translation/Total-Omission
re-classification audit (see the top-level `README.md` *QUANTA T2 selection* section and
`paper_data/audit_manifests/`). In no case is `DQI ≥ 0.75` applied as a filter to the
released T2 split. The verbatim audit prompt is at `raqeeb/data/prompts/etca_audit_prompt.txt`.

## 1. Clean_A seed audit (selects Tier2 generation seeds)

| File | Rows | Description |
|------|-----:|-------------|
| `etca_seed_audit_cleanA_794.csv` | 794 | Every Clean_A candidate with its five ETCA scores (`anchor_validity`, `phenomenon_clarity`, `arabic_naturalness`, `collateral_severity`, `tier2_seed_recommendation`) plus structural metrics (`ocs`, `cps`, `sfr`, `csr`). |
| `etca_seeds_recommended_664.csv` | 664 | The subset recommended as Tier2 generation seeds (`tier2_seed_recommendation_final = 1`). |
| `etca_seed_audit_stats.json` | — | Run summary: 794 audited → 664 recommended (130 rejected), recommendation rate 0.836; score means anchor 4.17 / phenomenon 4.45 / naturalness 4.40 / collateral 2.15; run 2026-01-29, 0 parse/API errors. |

## 2. T2 audit subset (quality audit of the synthetic split)

| File | Rows | Description |
|------|-----:|-------------|
| `etca_t2_audit_414.csv` | 414 | A stratified ~5% sample of T2 (all 23 classes), scored by **ETCA only**. ETCA per-dimension means: anchor 4.70 / phenomenon 4.74 / naturalness 4.62 / collateral 1.02 (of 5; collateral lower = better). |

**This is an ETCA (LLM) audit, not a human evaluation.** The file retains human-evaluation
*template* columns (`Q1`–`Q6`, `Annotator_ID`, `Technical_Score`, `Quality_Score`,
`Overall_Score`, `Status`), but they were never filled: the `Q*`/`Annotator_ID` columns are
empty, and `Technical/Quality/Overall_Score`/`Status` hold constant placeholder values
(`0.0` / `0.3` / `0.12` / `REVIEW`) produced by the scoring formula on empty human input.
Only the ETCA score columns carry real values. (The original filename for this artifact was
`TIER2_HUMANEVAL_WITH_ETCA.csv`, a misnomer — it is renamed here to reflect its actual content.)

## What the *human* evaluation is

The real human evaluation is the separate **147-instance dual-annotator IAA** (55 T1 + 92 T2),
reported in the paper (Appendix on dual-annotator validation). ETCA's external validation
against that human consensus reaches **Gwet's AC1 = 0.62** on a **43-instance Clean_A** subset —
this is *not* the 414-instance T2 audit subset above.
