"""
compute_iaa_ac1.py - reproduce the dual-annotator inter-annotator agreement
(Gwet's AC1) reported in Paper 1, Appendix B (tab:iaa-main), from the released
paired annotator files.

Inputs (same directory):
    phase1_T1_paired_AvsB.csv   55 T1 instances, Annotator A vs B, criteria Q1-Q7
    phase1_T2_paired_AvsB.csv   92 T2 instances, same columns

Criteria: Q1 Error, Q2 Anchor, Q3 Severity (0-5, normalised), Q4 Minimal-pair,
Q5 Label, Q6 Naturalness, Q7 UN-register. The paper reports 5 of 7 (error,
anchor, label, severity, UN-register); minimal-pair (Q4) and naturalness (Q6)
are omitted for systematic annotator interpretation divergence (disclosed in
App B). Core-three = error + anchor + label (Q1, Q2, Q5).

Gwet's AC1 is used (not Cohen's kappa) because the binary criteria are
high-prevalence by design, where kappa collapses artifactually (kappa paradox;
Feinstein & Cicchetti 1990; Gwet 2008).

Run:  python compute_iaa_ac1.py
"""
import csv

CRITERIA = {"Q1": "Error", "Q2": "Anchor", "Q3": "Severity",
            "Q5": "Label", "Q7": "UN-register"}   # 5 reported (Q4, Q6 omitted)
CORE = ["Q1", "Q2", "Q5"]


def gwet_ac1(a, b):
    """Gwet's AC1 for two raters over the same items (categorical)."""
    pairs = [(x, y) for x, y in zip(a, b) if x != "" and y != ""]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    A = [float(x) for x, _ in pairs]
    B = [float(y) for _, y in pairs]
    p_a = sum(1 for x, y in zip(A, B) if x == y) / n
    cats = sorted(set(A) | set(B))
    if len(cats) < 2:
        return 1.0, n
    pi = {k: (A.count(k) + B.count(k)) / (2 * n) for k in cats}
    p_e = sum(pi[k] * (1 - pi[k]) for k in cats) / (len(cats) - 1)
    return (p_a - p_e) / (1 - p_e), n


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    for tier, path in [("T1", "phase1_T1_paired_AvsB.csv"),
                       ("T2", "phase1_T2_paired_AvsB.csv")]:
        rows = load(path)
        print(f"--- {tier} (n={len(rows)}) ---")
        ac = {}
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]:
            a = [r[f"{q}_A"].strip() for r in rows]
            b = [r[f"{q}_B"].strip() for r in rows]
            ac[q], _ = gwet_ac1(a, b)
        for q, name in CRITERIA.items():
            if tier == "T1" and q == "Q7":   # UN-register is T2-only
                continue
            print(f"  {name:12s} AC1 = {ac[q]:.3f}")
        core = sum(ac[q] for q in CORE) / len(CORE)
        rep = [ac[q] for q in CRITERIA if not (tier == "T1" and q == "Q7")]
        print(f"  core-three  = {core:.3f}")
        print(f"  all-reported = {sum(rep) / len(rep):.3f}\n")


if __name__ == "__main__":
    main()
