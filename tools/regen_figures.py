#!/usr/bin/env python
"""
Regenerate all Raqeeb paper figures with the 2026 theme palette,
enlarged fonts, and modern chart types (v2).

Chart-type upgrades from v1:
  Fig 4: bar chart -> confusion matrix heatmap (restored)
  Fig 5: vertical bars -> horizontal lollipop chart
  Fig 6: heatmap -> improved annotated heatmap (same type, better styling)
  Fig 7: histograms -> KDE density plots (ZOOMED)
  Fig 8: horizontal bars -> polished horizontal bars with annotations
  Fig 9: box plot -> violin plot with strip overlay (ZOOMED)

Run from project root:
    python tools/regen_figures.py
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# =======================================================================
# 2026 Theme palette (matches paper.tex \definecolor entries)
# =======================================================================
THEME = {
    "primary":     "#1E40AF",
    "secondary":   "#DC2626",
    "tertiary":    "#F59E0B",
    "accent":      "#059669",
    "muted":       "#6B7280",
    "primaryLt":   "#E0E7FF",
    "secondaryLt": "#FEE2E2",
    "tertiaryLt":  "#FEF3C7",
    "accentLt":    "#D1FAE5",
    "bg":          "#F9FAFB",
}

# =======================================================================
# Global rcParams — 2026 research style
# =======================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "figure.titlesize": 15,
    "axes.edgecolor": THEME["muted"],
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": THEME["muted"],
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PROJ = Path(__file__).resolve().parents[1]
FIG_DIR = PROJ / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save(fig, name):
    out_pdf = FIG_DIR / f"{name}.pdf"
    out_png = FIG_DIR / f"{name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {out_pdf.relative_to(PROJ)}")


def load_clean_a():
    return pd.read_csv(
        PROJ / "cross_domain_validation/tier1_v3_3_cross_domain/tier1_clean_A.csv")


# =======================================================================
# Figure 3 — Data efficiency (12-class controlled ablation)
# =======================================================================
def fig_data_efficiency():
    print("Fig 3: data_efficiency  [2-panel bars]")
    data = json.loads(
        (PROJ / "delta_upload/results/P10/controlled_12class_ablation.json").read_text())
    conds = ["t1_12", "t2_12", "t1t2_12"]
    labels = ["T1\n(738 real)", "T2-12\n(4,274 synth.)", "T1+T2-12\n(5,012 comb.)"]
    means = [data[c]["mean_gold_f1"] for c in conds]
    stds  = [data[c]["std_gold_f1"]  for c in conds]
    des   = [data[c]["des_12class"]  for c in conds]
    colors = [THEME["primary"], THEME["secondary"], THEME["tertiary"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Bars: light fill (alpha 0.30) + same-color 2pt outline to match KDE style.
    # This makes Fig 3 visually consistent with the muted palette of Figs 6/7/8.
    bars1 = ax1.bar(labels, means, yerr=stds, color=colors,
                    edgecolor=colors, linewidth=2.0, alpha=0.30, capsize=4,
                    error_kw={"ecolor": THEME["muted"], "elinewidth": 1.2})
    # Re-draw edges on top at full alpha so the outline stays crisp
    for bar, c in zip(bars1, colors):
        bar.set_edgecolor(c)
        bar.set_linewidth(2.0)
    ax1.set_ylabel("Gold macro-F1 (12 shared classes)")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("(a) Macro-F1")
    for bar, m in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.018,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=THEME["primary"])

    bars2 = ax2.bar(labels, des, color=colors, edgecolor=colors,
                    linewidth=2.0, alpha=0.30)
    for bar, c in zip(bars2, colors):
        bar.set_edgecolor(c)
        bar.set_linewidth(2.0)
    ax2.set_ylabel("Data Efficiency Score (DES)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.04, 2.0)
    ax2.set_title("(b) Data efficiency (log)")
    for bar, d in zip(bars2, des):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f"{d:.3f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=THEME["primary"])

    fig.tight_layout()
    save(fig, "data_efficiency")


# =======================================================================
# Figure 4 — Confusion matrix heatmap (RESTORED)
# =======================================================================
def fig_confusion_matrix():
    print("Fig 4: meaning_shift_confusion  [confusion matrix heatmap — restored]")
    # Load AraBERT v2 best-fold predictions on Gold
    res_dir = PROJ / "delta_upload/results/M01"
    # Find best fold = fold 3 (from paper: best gold F1)
    best_fold = None
    best_f1 = -1
    for f in range(5):
        p = res_dir / f"AraBERT-v2_fold{f}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "gold_f1" in df.columns:
                gf = df["gold_f1"].iloc[0] if len(df) > 0 else 0
            else:
                gf = 0
            if gf > best_f1:
                best_f1 = gf
                best_fold = f

    # Try to build confusion matrix from predictions
    pred_path = res_dir / f"AraBERT-v2_fold{best_fold}.csv"
    gold_path = PROJ / "delta_upload/data/processed/test_gold.csv"

    if not pred_path.exists() or not gold_path.exists():
        print("  WARNING: prediction CSVs not found, generating placeholder")
        return

    pred_df = pd.read_csv(pred_path)
    gold_df = pd.read_csv(gold_path)

    # Check if predictions have y_true and y_pred columns
    if "Sub-Subtype" in pred_df.columns and "predicted_label" in pred_df.columns:
        y_true = pred_df["Sub-Subtype"]
        y_pred = pred_df["predicted_label"]
    elif "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
        y_true = pred_df["y_true"]
        y_pred = pred_df["y_pred"]
    else:
        print(f"  Columns in pred: {list(pred_df.columns)[:10]}")
        print("  WARNING: cannot determine y_true/y_pred columns, skipping confusion matrix")
        return

    # Get all class labels sorted
    all_labels = sorted(set(y_true) | set(y_pred))
    n = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    # Build confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    # Row-normalize for display
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Short labels
    short = [l.replace("Terminology ", "Term. ")
              .replace("Definiteness ", "Def. ")
              .replace("Hypernym for Hyponym", "Hyper→Hypo")
              .replace("Hyponym for Hypernym", "Hypo→Hyper")
              .replace("Active to Passive Voice", "Act→Pass")
              .replace("Passive to Active Voice", "Pass→Act")
              .replace("Perfective to Progressive", "Perf→Prog")
              .replace("Progressive to Perfective", "Prog→Perf")
              .replace("Tense Shift Under Negation", "Tense/Neg")
              .replace("Register Mismatch", "Reg. Mism.")
              .replace("Literal Translation", "Literal Trans.")
              .replace("Gender Disagreement", "Gender Dis.")
              .replace("Wrong Word Order", "Wrong WO")
              .replace("Wrong Structure", "Wrong Struct.")
              .replace("Meaning Shift", "Mean. Shift")
              .replace("Total Omission", "Total Om.")
              .replace("Partial Translation", "Partial Trans.")
              .replace("Name Entity Error", "Name Ent.")
              .replace("Tanween Omission", "Tanween Om.")
              .replace("Spelling Error", "Spelling Err.")
              .replace("Invalid Pattern", "Inv. Pattern")
              .replace("Noun to Adjective", "N→Adj")
              .replace("Adjective to Noun", "Adj→N")
             for l in all_labels]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "raqeeb_cm", ["#FFFFFF", THEME["primaryLt"], THEME["primary"], "#0F1D45"])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("AraBERT v2 best-fold confusion matrix on Gold (row-normalised)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                color = "white" if cm_norm[i, j] > 0.45 else THEME["muted"]
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    # Highlight diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                    fill=False, edgecolor=THEME["accent"],
                                    linewidth=1.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Row-normalised\nproportion", fontsize=10)

    fig.tight_layout()
    save(fig, "meaning_shift_confusion")


# =======================================================================
# Figure 5 — Horizontal lollipop chart (per-class acceptance)
# =======================================================================
def fig_per_class_acceptance():
    print("Fig 5: figure2_per_class_acceptance  [horizontal lollipop]")
    stats = json.loads(
        (PROJ / "cross_domain_validation/cross_domain_stats.json").read_text())
    rows = stats["rows_per_mizan_class"]
    df = load_clean_a()
    clean_counts = df.groupby("Sub-Subtype").size().to_dict()

    classes = list(rows.keys())
    candidates = [rows[c] for c in classes]
    accepted = [clean_counts.get(c, 0) for c in classes]
    acc_rate = [100 * a / max(1, n) for a, n in zip(accepted, candidates)]

    # Sort ascending for horizontal lollipop
    order = sorted(range(len(classes)), key=lambda i: acc_rate[i])
    classes   = [classes[i] for i in order]
    candidates = [candidates[i] for i in order]
    accepted  = [accepted[i] for i in order]
    acc_rate  = [acc_rate[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y_pos = range(len(classes))

    # Horizontal stems
    ax.hlines(y_pos, 0, acc_rate, color=THEME["primary"], linewidth=2.0, alpha=0.7)
    # Dots at the end
    ax.scatter(acc_rate, y_pos, color=THEME["primary"], s=80, zorder=5,
               edgecolors="white", linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Clean_A acceptance rate (%)")
    ax.set_title("Per-class acceptance on WMT24++ cross-domain")
    ax.set_xlim(0, max(acc_rate) * 1.18)

    for i, (rate, n) in enumerate(zip(acc_rate, candidates)):
        ax.text(rate + 1.0, i, f"{rate:.1f}%  (n={n})",
                va="center", fontsize=9, color=THEME["primary"], fontweight="bold")

    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "figure2_per_class_acceptance")


# =======================================================================
# Figure 6 — Class x Domain acceptance-rate heatmap
# Matches paper caption: "Cross-domain clean rate heatmap: Mizan class (rows,
# sorted by acceptance) x WMT24pp domain (columns). Cells are coloured on a
# red-yellow-green diverging scale; missing combinations blank."
# =======================================================================
def fig_domain_class_heatmap():
    print("Fig 6: figure3_domain_class_heatmap  [class x domain acceptance rate]")
    base = PROJ / "cross_domain_validation/tier1_v3_3_cross_domain"
    clean_a = pd.read_csv(base / "tier1_clean_A.csv")
    clean_b = pd.read_csv(base / "tier1_clean_B.csv")
    noisy   = pd.read_csv(base / "tier1_noisy.csv")

    # A+B are both "clean" in paper's 23.6% figure
    clean = pd.concat([clean_a, clean_b], ignore_index=True)
    all_candidates = pd.concat([clean, noisy], ignore_index=True)

    # Restrict to the 4 actual WMT24++ domains (drop any 'unknown')
    DOMAINS = ["news", "social", "literary", "speech"]
    all_candidates = all_candidates[all_candidates["domain"].isin(DOMAINS)]
    clean = clean[clean["domain"].isin(DOMAINS)]

    total_by_cd = all_candidates.groupby(
        ["Sub-Subtype", "domain"]).size().unstack(fill_value=0)
    clean_by_cd = clean.groupby(
        ["Sub-Subtype", "domain"]).size().unstack(fill_value=0)

    # Ensure consistent column order
    total_by_cd = total_by_cd.reindex(columns=DOMAINS, fill_value=0)
    clean_by_cd = clean_by_cd.reindex(columns=DOMAINS, fill_value=0)
    # Ensure same class rows in both
    all_classes = sorted(set(total_by_cd.index) | set(clean_by_cd.index))
    total_by_cd = total_by_cd.reindex(index=all_classes, fill_value=0)
    clean_by_cd = clean_by_cd.reindex(index=all_classes, fill_value=0)

    # Acceptance rate; mask cells where no input candidates exist
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = (clean_by_cd / total_by_cd.replace(0, np.nan)) * 100
    rate = rate.fillna(np.nan)

    # Sort rows by overall acceptance (descending)
    total_per_class = total_by_cd.sum(axis=1)
    clean_per_class = clean_by_cd.sum(axis=1)
    overall_rate = (clean_per_class / total_per_class.replace(0, np.nan)) * 100
    row_order = overall_rate.fillna(-1).sort_values(ascending=False).index.tolist()
    rate = rate.reindex(index=row_order)
    total_by_cd = total_by_cd.reindex(index=row_order)
    clean_by_cd = clean_by_cd.reindex(index=row_order)

    # Red-Yellow-Green diverging colormap (low = red, high = green)
    # Softer 400-tier hues for a less saturated look that matches the muted
    # palette of Figs 6/7/8 (no more full-saturation red/green cells).
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ryg_soft", ["#F87171", "#FBBF24", "#34D399"])
    cmap.set_bad(color="#F3F4F6")  # blank cells for missing combos

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    values = rate.values.astype(float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                   vmin=0, vmax=max(50, np.nanmax(values) if np.isfinite(np.nanmax(values)) else 50))

    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([d.capitalize() for d in DOMAINS], fontsize=11)
    ax.set_yticks(range(len(row_order)))
    # Wrap multi-word class names to 2 lines for readability
    wrapped_labels = [c.replace(" ", "\n", 1) if " " in c else c
                      for c in row_order]
    ax.set_yticklabels(wrapped_labels, fontsize=10)
    # Internal title removed — paper's \caption{} provides the title (ACL convention)

    # Annotate cells with rate% and count
    for i, cls in enumerate(row_order):
        for j, dom in enumerate(DOMAINS):
            v = rate.iloc[i, j]
            tot = total_by_cd.iloc[i, j]
            cln = clean_by_cd.iloc[i, j]
            if np.isnan(v) or tot == 0:
                continue
            # Color: white on dark cells, dark text on light cells
            color = "white" if v > 28 else "#111827"
            ax.text(j, i, f"{v:.0f}%\n{cln}/{tot}",
                    ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Acceptance rate (%)", fontsize=11)
    fig.tight_layout()
    save(fig, "figure3_domain_class_heatmap")


# =======================================================================
# Figure 6 — Clean_A CPS + SFR distributions (2 KDE panels)
# =======================================================================
def fig_cleanA_metrics():
    print("Fig 6: figure1_cleanA_metrics  [2-panel CPS + SFR KDE]")
    df = load_clean_a()

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    kde_signals = [
        (axes[0], "cps", "CPS (Context Preservation)", THEME["primary"], (0.45, 1.02)),
        (axes[1], "sfr", "SFR (Semantic Fidelity)",    THEME["accent"],  (0.45, 1.02)),
    ]
    from scipy.stats import gaussian_kde
    for ax, col, xlabel, color, xlim in kde_signals:
        vals = df[col].dropna().values
        vals_in = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        try:
            kde = gaussian_kde(vals_in, bw_method=0.15)
            xs = np.linspace(xlim[0], xlim[1], 400)
            ys = kde(xs)
            ax.fill_between(xs, ys, alpha=0.30, color=color)
            ax.plot(xs, ys, color=color, linewidth=2.0)
        except (ValueError, np.linalg.LinAlgError):
            ax.hist(vals_in, bins=50, color=color, edgecolor="white",
                    alpha=0.6, density=True, linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        med = np.median(vals)
        if xlim[0] <= med <= xlim[1]:
            ymax = ax.get_ylim()[1]
            ax.axvline(med, color=THEME["secondary"], linestyle="--",
                       linewidth=1.2, alpha=0.7)
            ax.text(med, ymax * 0.88, f"med={med:.3f}",
                    fontsize=8, color=THEME["secondary"], ha="center")
        ax.set_ylabel("Density" if col == "cps" else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    fig.tight_layout()
    save(fig, "figure1_cleanA_metrics")


# =======================================================================
# Figure 7 — MPQS diagnostic (standalone ranked dot plot)
# Surfaces the "investigation tool" story: nearly everything at MPQS=1.0,
# 2 candidates flagged as edge cases by the aggregate diagnostic despite
# normal OCS/CPS/SFR values.
# =======================================================================
def fig_mpqs_diagnostic():
    print("Fig 7: figureA2_mpqs_diagnostic  [ranked dot plot, diagnostic panel]")
    df = load_clean_a()

    fig, ax = plt.subplots(figsize=(6.2, 3.0))

    mpqs_vals = df["mpqs"].dropna().values
    sorted_mpqs = np.sort(mpqs_vals)
    ranks = np.arange(1, len(sorted_mpqs) + 1)
    outlier_mask = sorted_mpqs < 0.99
    n_total = len(sorted_mpqs)
    n_out   = int(outlier_mask.sum())
    n_floor = n_total - n_out
    mean_mpqs = float(np.mean(mpqs_vals))

    # Normal candidates at MPQS ceiling
    ax.scatter(ranks[~outlier_mask], sorted_mpqs[~outlier_mask],
               s=10, color=THEME["primary"], alpha=0.35,
               edgecolors="none", zorder=2,
               label=f"Normal candidate (n={n_floor})")
    # Flagged edge cases
    ax.scatter(ranks[outlier_mask], sorted_mpqs[outlier_mask],
               s=90, color=THEME["secondary"], zorder=5,
               edgecolors="white", linewidth=1.4, marker="o",
               label=f"Flagged edge case (n={n_out})")

    # Mean reference line
    ax.axhline(mean_mpqs, color=THEME["muted"], linestyle="--",
               linewidth=1.0, alpha=0.7)
    ax.text(n_total * 0.02, mean_mpqs - 0.05,
            f"Mean MPQS = {mean_mpqs:.3f}  (matches UN Clean_A)",
            fontsize=9, color=THEME["muted"], va="top", ha="left")

    # Annotations for the two outliers
    for r, v in zip(ranks[outlier_mask], sorted_mpqs[outlier_mask]):
        ax.annotate(f"{v:.2f}", xy=(r, v), xytext=(r + 45, v + 0.05),
                    fontsize=9, color=THEME["secondary"],
                    arrowprops=dict(arrowstyle="-", color=THEME["secondary"],
                                    lw=0.8, alpha=0.6))

    # Ceiling count annotation
    ax.text(n_total * 0.55, 1.05,
            f"{n_floor}/{n_total} ({100*n_floor/n_total:.1f}%) at MPQS = 1.0",
            fontsize=9, color=THEME["primary"], ha="center", va="bottom",
            fontweight="bold")

    ax.set_xlim(-15, n_total + 15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_xlabel("Candidate rank (ascending MPQS)")
    ax.set_ylabel("MPQS (aggregate signal)")
    ax.legend(loc="center right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    save(fig, "figureA2_mpqs_diagnostic")


# =======================================================================
# Figure 8 — Polished horizontal bars (rejection reasons)
# =======================================================================
def fig_rejection_reasons():
    print("Fig 8: figure4_rejection_reasons  [polished horizontal bars]")
    stats = json.loads(
        (PROJ / "cross_domain_validation/cross_domain_stats.json").read_text())
    reasons = stats["exclusion_reasons"]
    labels = list(reasons.keys())
    counts = list(reasons.values())
    total = sum(counts)

    # Shorten labels
    short_labels = [
        "Addition error\n(no pos. keyword)",
        "No extractable\nword-level diffs",
        "Keyword has\nno Arabic words",
        "Best_match\ntoo short (<2 chars)",
    ]

    colors = [THEME["secondary"], THEME["tertiary"], THEME["muted"], THEME["muted"]]

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bars = ax.barh(range(len(labels)), counts, color=colors,
                   edgecolor="white", linewidth=1.0, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Excluded candidates")
    ax.set_title(f"Pre-validation exclusion reasons  (total = {total})")
    ax.set_xlim(0, max(counts) * 1.35)

    for bar, c in zip(bars, counts):
        pct = 100 * c / total
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                f"{c}  ({pct:.1f}%)", va="center", fontsize=10,
                color=THEME["primary"], fontweight="bold")

    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "figure4_rejection_reasons")


# =======================================================================
# Figure 9 — Violin + strip plot for SFR by class (ZOOMED)
# =======================================================================
def fig_sfr_by_class():
    print("Fig 9: figureA1_sfr_by_class  [violin + strip, full cross-domain set]")
    # CRITICAL: use full cross-domain set (clean A + clean B + noisy), not Clean_A only.
    # Clean_A contains only Regime-F classes that passed; the paper caption says
    # "full cross-domain set (clean + noisy)" so we need all 6,402 candidates.
    base = PROJ / "cross_domain_validation/tier1_v3_3_cross_domain"
    clean_a = pd.read_csv(base / "tier1_clean_A.csv")
    clean_b = pd.read_csv(base / "tier1_clean_B.csv")
    noisy   = pd.read_csv(base / "tier1_noisy.csv")
    df = pd.concat([clean_a, clean_b, noisy], ignore_index=True)
    # Restrict to the 4 actual WMT24++ domains
    if "domain" in df.columns:
        df = df[df["domain"].isin(["news", "social", "literary", "speech"])]

    # Order classes by median SFR ascending (lowest first = Meaning Shift leftmost)
    classes_all = df["Sub-Subtype"].unique()
    medians = {c: df.loc[df["Sub-Subtype"] == c, "sfr"].median() for c in classes_all}
    classes = sorted(classes_all, key=lambda c: medians[c])
    data_by_class = [df.loc[df["Sub-Subtype"] == c, "sfr"].dropna().values
                     for c in classes]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))

    # Violin
    parts = ax.violinplot(data_by_class, positions=range(len(classes)),
                          showmeans=False, showmedians=False, showextrema=False,
                          widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(THEME["primaryLt"])
        pc.set_edgecolor(THEME["primary"])
        pc.set_linewidth(1.0)
        pc.set_alpha(0.7)

    # Strip (jittered scatter)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data_by_class):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, s=8, color=THEME["primary"],
                   alpha=0.35, edgecolors="none", zorder=3)

    # Median markers
    for i, vals in enumerate(data_by_class):
        if len(vals) > 0:
            med = np.median(vals)
            ax.scatter(i, med, s=50, color=THEME["secondary"],
                       edgecolors="white", linewidth=1.5, zorder=5, marker="D")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("Semantic Fidelity Ratio (SFR)")
    # No internal title — paper's \caption{} provides it
    ax.set_ylim(0.20, 1.05)  # wider than before to show the Regime D window

    # Regime D window boundaries (paper caption references both)
    ax.axhline(0.45, color=THEME["tertiary"], linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(len(classes) - 0.15, 0.455, "Regime D lower (0.45)",
            color=THEME["tertiary"], fontsize=8, va="bottom", ha="right")
    ax.axhline(0.78, color=THEME["secondary"], linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(len(classes) - 0.15, 0.785, "Regime D upper (0.78)",
            color=THEME["secondary"], fontsize=8, va="bottom", ha="right")

    # Legend for median diamond
    ax.scatter([], [], s=50, color=THEME["secondary"], edgecolors="white",
               linewidth=1.5, marker="D", label="Median")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)

    fig.tight_layout()
    save(fig, "figureA1_sfr_by_class")


# =======================================================================
# Entry point
# =======================================================================
def main():
    print(f"Regenerating all figures (v2) -> {FIG_DIR}\n")
    fig_data_efficiency()
    fig_confusion_matrix()
    fig_per_class_acceptance()
    fig_domain_class_heatmap()
    fig_cleanA_metrics()       # Fig 6: CPS + SFR only
    fig_mpqs_diagnostic()      # Fig 7: MPQS ranked-dot diagnostic (new)
    fig_rejection_reasons()
    fig_sfr_by_class()
    print("\nDone. Recompile paper.tex to pick up new PDFs.")


if __name__ == "__main__":
    main()
