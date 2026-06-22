"""
compute_etca_validation.py - reproduce the ETCA external-validation number in
Paper 1 (Gwet's AC1 = 0.62 of the cross-vendor ETCA judge against the
dual-annotator human consensus, n = 43 Clean_A instances).

Input (same directory):
    etca_joined_raw.csv   43 Clean_A instances, each with the ETCA dimension
                          scores (etca_anchor/clarity/natural, 1-5) joined to
                          the two annotators' criteria means (q1_mean..q7_mean;
                          1.0 = both annotators said yes, 0.5 = split, 0 = both no).

ETCA is binarised at >=4 ("high confidence"); the human side is "consensus yes"
(both annotators = 1, i.e. mean == 1.0). We report Gwet's AC1 (high-prevalence
regime; see App B). Anchor-vs-Q2 and clarity-vs-Q1 both reproduce 0.616 -> 0.62.

Run:  python compute_etca_validation.py
"""
import csv


def gwet_ac1(a, b):
    n = len(a)
    p_a = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return 1.0
    pi = {k: (a.count(k) + b.count(k)) / (2 * n) for k in cats}
    p_e = sum(pi[k] * (1 - pi[k]) for k in cats) / (len(cats) - 1)
    return (p_a - p_e) / (1 - p_e)


def main():
    rows = list(csv.DictReader(open("etca_joined_raw.csv", encoding="utf-8-sig")))
    print(f"n = {len(rows)} Clean_A instances")
    anchor = [1 if float(r["etca_anchor"]) >= 4 else 0 for r in rows]
    clarity = [1 if float(r["etca_clarity"]) >= 4 else 0 for r in rows]
    q2_cons = [1 if float(r["q2_mean"]) == 1.0 else 0 for r in rows]   # anchor-correct consensus
    q1_cons = [1 if float(r["q1_mean"]) == 1.0 else 0 for r in rows]   # error-correct consensus
    print(f"ETCA anchor>=4  vs  Q2 (anchor) consensus : AC1 = {gwet_ac1(anchor, q2_cons):.3f}")
    print(f"ETCA clarity>=4 vs  Q1 (error)  consensus : AC1 = {gwet_ac1(clarity, q1_cons):.3f}")


if __name__ == "__main__":
    main()
