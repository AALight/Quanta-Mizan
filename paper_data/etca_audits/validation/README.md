# ETCA external validation (Paper 1, §4.3)

Reproduces the paper's claim that the cross-vendor ETCA judge (Claude Sonnet 4)
agrees with the dual-annotator **human consensus** at Gwet's **AC1 = 0.62**
(substantial), on a 43-instance Clean_A subset.

## Files
- `etca_joined_raw.csv` - 43 Clean_A instances; ETCA dimension scores
  (`etca_anchor/clarity/natural`, 1-5) joined to the two annotators' per-criterion
  means (`q1_mean..q7_mean`; 1.0 = both said yes, 0.5 = split, 0 = both no).
- `compute_etca_validation.py` - recomputes the AC1 from that CSV.
- `etca_validation_headline.json` - the full result (per-dimension AC1 + 95% CIs).

## Reproduced (`python compute_etca_validation.py`)
| Comparison | n | AC1 |
|---|---|---|
| ETCA anchor >=4 vs Q2 (anchor) consensus | 43 | **0.616 -> 0.62** |
| ETCA clarity >=4 vs Q1 (error) consensus | 43 | 0.616 -> 0.62 |

(95% CI ~ [0.35, 0.81]; naturalness-vs-Q7 is much lower, 0.04, and is not the
cited number.)

## Notes
- This is the **external validation of ETCA**, distinct from the per-dimension
  ETCA means on the 414-instance T2 subset (parent directory).
- ETCA is a **parallel quality audit, not a selection gate**: neither T1 (738,
  structural + gold-overlap removal) nor T2 (7,856, structural re-classification
  audit) is gated by ETCA / DQI.
- Gwet's AC1 (not Cohen's kappa) because the binary criteria are high-prevalence
  by design (see App B).
