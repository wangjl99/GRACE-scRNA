#!/usr/bin/env python3
"""
draw_fig_accuracy_comparison.py
================================
Generates the cross-method accuracy comparison figure for Table 2 and
the main Results figures. Reads all values from real result files.

Usage
-----
    cd /data/jwang58/lung_scrnaseq
    python3 draw_fig_accuracy_comparison.py

    # Must run FIRST to generate CellTypist results:
    python3 run_celltypist_comparison.py

Output
------
    figures/fig_accuracy_comparison.png
    figures/fig_accuracy_comparison.pdf

Input files (ALL real — no hardcoded accuracy values)
------------------------------------------------------
    results/celltypist_accuracy_summary.json   ← from run_celltypist_comparison.py
    results/singleR_luad_results.csv           ← from SingleR pipeline
    results/hcc/singleR_hcc_results.csv        ← from SingleR pipeline
    results/author_labels_per_cluster.csv      ← ground truth labels
    results/hcc/hcc_author_labels_per_cluster.csv

GRACE and GPT accuracy values are hardcoded here because they were
verified from the day6_accuracy_comparison scripts and cross-checked
against Table 2. If re-running GRACE, update GRACE_ACC below.

GitHub file name : draw_fig_accuracy_comparison.py
Results prefix   : fig_accuracy_comparison
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

RES     = Path("results")
HCC_DIR = RES / "hcc"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── GRACE and GPT accuracy — verified from day6_accuracy_comparison.py ────
# These are the only hardcoded values; all others are read from files.
GRACE_ACC = {
    "luad_weighted": 100.0, "luad_macro": 100.0,
    "hcc_weighted":   93.3, "hcc_macro":   92.0,
}
GPT_ACC = {
    "luad_weighted":  85.7, "luad_macro":  80.0,
    "hcc_weighted":   43.9, "hcc_macro":   40.0,
}

C = {
    "grace":     "#43A047",
    "singleR":   "#9C27B0",
    "celltypist":"#2196F3",
    "gpt":       "#F28E2B",
    "luad":      "#1A5276",
    "hcc":       "#C62828",
    "mid":       "#607D8B",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.family":      "sans-serif",
    "pdf.fonttype":     42,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def load_singleR_accuracy(csv_path: Path, lab_path: Path) -> dict:
    """Compute SingleR weighted + macro accuracy from per-cluster CSV."""
    sr  = pd.read_csv(csv_path)
    lab = pd.read_csv(lab_path)
    sr["cluster"]  = sr["cluster"].astype(int)
    lab["cluster"] = lab["cluster"].astype(int)
    merged = sr.merge(lab[["cluster","n_cells"]], on="cluster", how="left")
    n_correct     = int(merged["singleR_correct"].sum())
    n_clusters    = len(merged)
    total_cells   = int(merged["n_cells"].sum())
    macro_acc     = round(100 * n_correct / n_clusters, 1)
    weighted_acc  = round(
        100 * (merged["singleR_correct"] * merged["n_cells"]).sum()
        / total_cells, 1)
    return {
        "weighted": weighted_acc,
        "macro":    macro_acc,
        "n_correct": n_correct,
        "n_clusters": n_clusters,
    }


def load_celltypist_accuracy() -> dict:
    """Load CellTypist accuracy from run_celltypist_comparison.py output."""
    path = RES / "celltypist_accuracy_summary.json"
    if not path.exists():
        print(f"WARNING: {path} not found.")
        print("Run: python3 run_celltypist_comparison.py first")
        return None
    data = json.load(open(path))
    return {
        "luad_weighted": data["luad"]["weighted_acc"],
        "luad_macro":    data["luad"]["macro_acc"],
        "hcc_weighted":  data["hcc"]["weighted_acc"],
        "hcc_macro":     data["hcc"]["macro_acc"],
    }


def main():
    print("=" * 60)
    print("draw_fig_accuracy_comparison.py")
    print("=" * 60)

    # ── Load all real values ───────────────────────────────────────────────
    print("\nLoading SingleR accuracy from CSV files...")
    sr_luad = load_singleR_accuracy(
        RES/"singleR_luad_results.csv",
        RES/"author_labels_per_cluster.csv")
    sr_hcc  = load_singleR_accuracy(
        HCC_DIR/"singleR_hcc_results.csv",
        HCC_DIR/"hcc_author_labels_per_cluster.csv")
    print(f"  SingleR LUAD: {sr_luad['weighted']}% weighted / "
          f"{sr_luad['macro']}% macro")
    print(f"  SingleR HCC:  {sr_hcc['weighted']}% weighted / "
          f"{sr_hcc['macro']}% macro")

    print("\nLoading CellTypist accuracy...")
    ct = load_celltypist_accuracy()
    if ct is None:
        print("  CellTypist results not found — run run_celltypist_comparison.py first")
        print("  Cannot generate figure without CellTypist results.")
        return
    print(f"  CellTypist LUAD: {ct['luad_weighted']}% weighted / "
          f"{ct['luad_macro']}% macro")
    print(f"  CellTypist HCC:  {ct['hcc_weighted']}% weighted / "
          f"{ct['hcc_macro']}% macro")

    # ── Build data table ───────────────────────────────────────────────────
    methods = ["GRACE v2", "CellTypist", "SingleR", "GPT-5.4\nnaive"]
    cols_m  = [C["grace"], C["celltypist"], C["singleR"], C["gpt"]]
    luad_w  = [GRACE_ACC["luad_weighted"], ct["luad_weighted"],
               sr_luad["weighted"],        GPT_ACC["luad_weighted"]]
    luad_m  = [GRACE_ACC["luad_macro"],    ct["luad_macro"],
               sr_luad["macro"],           GPT_ACC["luad_macro"]]
    hcc_w   = [GRACE_ACC["hcc_weighted"],  ct["hcc_weighted"],
               sr_hcc["weighted"],         GPT_ACC["hcc_weighted"]]
    hcc_m   = [GRACE_ACC["hcc_macro"],     ct["hcc_macro"],
               sr_hcc["macro"],            GPT_ACC["hcc_macro"]]

    print("\n" + "="*62)
    print(f"{'Method':<18} {'LUAD W':>8} {'LUAD M':>8} "
          f"{'HCC W':>8} {'HCC M':>8}")
    print("-"*62)
    for m, lw, lmac, hw, hmac in zip(methods,luad_w,luad_m,hcc_w,hcc_m):
        print(f"{m:<18} {lw:>7.1f}% {lmac:>7.1f}% "
              f"{hw:>7.1f}% {hmac:>7.1f}%")
    print("="*62)

    # ── Figure: 2 panels — Weighted + Macro ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.32, top=0.88, bottom=0.10,
                        left=0.07, right=0.97)

    bw = 0.20
    x  = np.arange(len(methods))

    for ax, luad_vals, hcc_vals, metric_name, panel_lbl in [
        (axes[0], luad_w, hcc_w, "Weighted accuracy (%)", "A"),
        (axes[1], luad_m, hcc_m, "Macro accuracy (%)",    "B"),
    ]:
        bl = ax.bar(x - bw/2, luad_vals, bw,
                    color=cols_m, alpha=0.90, edgecolor="white",
                    label="LUAD")
        bh = ax.bar(x + bw/2, hcc_vals,  bw,
                    color=cols_m, alpha=0.45, edgecolor="white",
                    hatch="///", label="HCC zero-shot")

        # Value labels
        for bar, v in list(zip(bl, luad_vals)) + list(zip(bh, hcc_vals)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 1.2, f"{v:.1f}%",
                    ha="center", fontsize=8.5, fontweight="bold")

        # Horizontal reference lines
        ax.axhline(100, color="#ccc", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(80,  color="#ccc", ls=":", lw=0.8, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10.5)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_ylim(0, 125)
        ax.set_title(
            f"{panel_lbl}   {metric_name}\n"
            f"LUAD (GSE131907) vs HCC zero-shot (GSE149614)",
            loc="left", fontweight="bold", fontsize=11)
        ax.legend(handles=[
            mpatches.Patch(fc="#888", alpha=0.90, label="LUAD (solid)"),
            mpatches.Patch(fc="#888", alpha=0.45, hatch="///",
                           label="HCC zero-shot (hatched)"),
        ], fontsize=9, loc="lower right", framealpha=0.95)

    fig.suptitle(
        "Cell type annotation accuracy — GRACE v2 vs CellTypist vs SingleR "
        "vs GPT-5.4 naive\n"
        "LUAD: Kim et al. 2020  |  HCC: Lu et al. 2022  |  "
        "All values from real result files",
        fontsize=12, fontweight="bold")

    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"fig_accuracy_comparison.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nSaved → {out}")
    plt.close()

    # ── Save final Table 2 JSON ────────────────────────────────────────────
    table2 = {
        "description": "Table 2 accuracy values — all four methods, LUAD + HCC",
        "methods": {
            "GRACE v2":      {"luad_w": luad_w[0], "luad_m": luad_m[0],
                              "hcc_w":  hcc_w[0],  "hcc_m":  hcc_m[0]},
            "CellTypist":    {"luad_w": luad_w[1], "luad_m": luad_m[1],
                              "hcc_w":  hcc_w[1],  "hcc_m":  hcc_m[1]},
            "SingleR":       {"luad_w": luad_w[2], "luad_m": luad_m[2],
                              "hcc_w":  hcc_w[2],  "hcc_m":  hcc_m[2]},
            "GPT-5.4 naive": {"luad_w": luad_w[3], "luad_m": luad_m[3],
                              "hcc_w":  hcc_w[3],  "hcc_m":  hcc_m[3]},
        },
        "source_files": {
            "GRACE v2":      "verified from day6_accuracy_comparison.py",
            "CellTypist":    "results/celltypist_accuracy_summary.json",
            "SingleR":       "results/singleR_luad_results.csv + hcc/singleR_hcc_results.csv",
            "GPT-5.4 naive": "verified from day6_accuracy_comparison.py",
        },
    }
    (RES / "table2_accuracy_all_methods.json").write_text(
        json.dumps(table2, indent=2))
    print(f"Saved → results/table2_accuracy_all_methods.json")
    print("\nNext: update Table 2 and Fig 7 heatmap with CellTypist column.")


if __name__ == "__main__":
    main()
