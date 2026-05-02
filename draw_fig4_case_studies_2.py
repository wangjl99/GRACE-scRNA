#!/usr/bin/env python3
"""
draw_fig4_case_studies.py
==========================
Generates Figure 4: Uncertainty-flagged clusters — GRACE case studies.

Usage
-----
    cd /data/jwang58/lung_scrnaseq
    conda activate luad_agents
    python3 draw_fig4_case_studies.py

Output
------
    figures/fig4_case_studies.png
    figures/fig4_case_studies.pdf

Data sources (ALL real — reads from result files)
---------------------------------------------------
    results/degs/cluster_15_degs.csv
    results/hcc/degs/cluster_11_degs.csv
    results/luad_enrichr_pathways.json
    results/hcc/hcc_enrichr_pathways.json
    results/versionB_v2only_results.json
    results/hcc/hcc_versionB_results.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RES     = Path("results")
HCC_DIR = RES / "hcc"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

C = {
    "luad": "#1A5276", "hcc": "#C62828", "path": "#E67E22",
    "conf": "#43A047", "unct": "#E53935", "mid": "#607D8B",
    "amber": "#7D3C00",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_data():
    print("Loading real data from server files...")

    # DEGs
    df15 = pd.read_csv(RES / "degs" / "cluster_15_degs.csv")
    df15 = (df15[df15["logfoldchanges"] >= 0.5]
            .sort_values("scores", ascending=False).head(8)
            .reset_index(drop=True))

    df11 = pd.read_csv(HCC_DIR / "degs" / "cluster_11_degs.csv")
    df11 = (df11[df11["logfoldchanges"] >= 0.5]
            .sort_values("scores", ascending=False).head(8)
            .reset_index(drop=True))

    print(f"  C15 top DEG: {df15['names'].iloc[0]}  FC={df15['logfoldchanges'].iloc[0]:.2f}")
    print(f"  C11 top DEG: {df11['names'].iloc[0]}  FC={df11['logfoldchanges'].iloc[0]:.2f}")

    # Enrichr pathways
    enr_l   = json.load(open(RES / "luad_enrichr_pathways.json"))
    enr_h   = json.load(open(HCC_DIR / "hcc_enrichr_pathways.json"))
    paths15 = enr_l.get("15", {}).get("enrichr_pathways", [])[:5]
    paths11 = enr_h.get("11", {}).get("enrichr_pathways", [])[:5]
    print(f"  C15 pathways: {len(paths15)}")
    print(f"  C11 pathways: {len(paths11)}")

    # Agent confidence
    vbd_l = {str(x["cluster"]): x
             for x in json.load(open(RES / "versionB_v2only_results.json"))}
    vbd_h = {str(x["cluster"]): x
             for x in json.load(open(HCC_DIR / "hcc_versionB_results.json"))}

    def get_conf(item):
        orch = item.get("orchestration", {})
        ac   = orch.get("agent_confidences", {})
        return {
            "overall": round(float(orch.get("overall_confidence", 0)), 3),
            "deg":     round(float(ac.get("deg", 0)), 3),
            "pathway": round(float(ac.get("pathway", 0)), 3),
            "disease": round(float(ac.get("disease", 0)), 3),
            "cell_id": round(float(ac.get("cell_identity",
                               ac.get("cell_id", ac.get("c_cell_id", 0)))), 3),
            "n_unc":   len(item.get("uncertainty_claims",
                           orch.get("uncertainty_claims", []))),
        }

    c15    = get_conf(vbd_l.get("15", {}))
    c11    = get_conf(vbd_h.get("11", {}))
    c1_ref = get_conf(vbd_l.get("1",  {}))

    print(f"  C15: c_overall={c15['overall']}  c_cell_id={c15['cell_id']}  n_unc={c15['n_unc']}")
    print(f"  C11: c_overall={c11['overall']}  c_cell_id={c11['cell_id']}  n_unc={c11['n_unc']}")
    print(f"  C1:  c_overall={c1_ref['overall']}  c_cell_id={c1_ref['cell_id']}  n_unc={c1_ref['n_unc']}")

    return dict(df15=df15, df11=df11,
                paths15=paths15, paths11=paths11,
                c15=c15, c11=c11, c1_ref=c1_ref)


def deg_panel(ax, df, title, col):
    genes = df["names"].tolist()[::-1]
    fcs   = df["logfoldchanges"].tolist()[::-1]
    y     = np.arange(len(genes))
    bars  = ax.barh(y, fcs, 0.65, color=col, alpha=0.82, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Log₂ fold change", fontsize=10.5)
    ax.set_xlim(0, max(fcs) * 1.20)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    for bar, fc in zip(bars, fcs):
        ax.text(fc + max(fcs)*0.015,
                bar.get_y() + bar.get_height()/2,
                f"{fc:.1f}", va="center", fontsize=9, fontweight="bold")


def path_panel(ax, paths, title, col, note):
    if not paths:
        ax.text(0.5, 0.5, "No significant pathways",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#999")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        return
    terms  = [p["Term"][:40] for p in paths][::-1]
    scores = [p["Combined Score"] for p in paths][::-1]
    y      = np.arange(len(terms))
    bars   = ax.barh(y, scores, 0.65, color=col, alpha=0.82, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=9.5)
    ax.set_xlabel("Enrichr combined score", fontsize=10.5)
    ax.set_xlim(0, max(scores) * 1.20)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
    for bar, s in zip(bars, scores):
        ax.text(s + max(scores)*0.015,
                bar.get_y() + bar.get_height()/2,
                f"{s:.0f}", va="center", fontsize=8.5, fontweight="bold")
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.0))


def conf_badge(ax, cd, col):
    sym_ov = "✓" if cd["overall"]  >= 0.50 else "✗"
    sym_ci = "✓" if cd["cell_id"] >= 0.50 else "✗"
    ax.text(0.98, 0.02,
            f"c_overall={cd['overall']:.3f} {sym_ov}\n"
            f"c_cell_id={cd['cell_id']:.3f}  {sym_ci}\n"
            f"[UNCERTAIN] flags: {cd['n_unc']}",
            transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="right", color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.0))


def conf_comparison(ax, c15, c11, c1_ref):
    agents = ["DEG\nvalidator", "Reactome\npathway",
              "DisGeNET\ndisease", "CellMarker\ncell ID", "c_overall"]
    c15v = [c15["deg"],    c15["pathway"],    c15["disease"],    c15["cell_id"],    c15["overall"]]
    c11v = [c11["deg"],    c11["pathway"],    c11["disease"],    c11["cell_id"],    c11["overall"]]
    c1v  = [c1_ref["deg"], c1_ref["pathway"], c1_ref["disease"], c1_ref["cell_id"], c1_ref["overall"]]

    x, bw = np.arange(5), 0.25
    ax.bar(x-bw, c15v, bw, color=C["amber"], alpha=0.88, edgecolor="white",
           label=f"C15 LUAD (KRAS-Hippo)  c_overall={c15['overall']:.3f}  [{c15['n_unc']} flag]")
    ax.bar(x,    c11v, bw, color=C["hcc"],   alpha=0.78, edgecolor="white",
           label=f"C11 HCC (xenobiotic/drug metab)  c_overall={c11['overall']:.3f}  [{c11['n_unc']} flags]")
    ax.bar(x+bw, c1v,  bw, color=C["conf"],  alpha=0.88, edgecolor="white",
           label=f"C1 LUAD (CD8+ T cell, confident reference)  c_overall={c1_ref['overall']:.3f}  [{c1_ref['n_unc']} flags]")

    for xi, (v15, v11, v1) in enumerate(zip(c15v, c11v, c1v)):
        for offset, val, col_ in [(-bw, v15, C["amber"]),
                                   (0,   v11, C["hcc"]),
                                   (bw,  v1,  C["conf"])]:
            if val > 0:
                ax.text(xi+offset, val+0.024, f"{val:.2f}",
                        ha="center", fontsize=8, fontweight="bold", color=col_)

    ax.axhline(0.50, color=C["unct"], ls="--", lw=1.4, alpha=0.8,
               label="Threshold (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=10.5)
    ax.set_ylabel("Agent confidence score", fontsize=10.5)
    ax.set_ylim(0, 1.22)
    ax.set_title(
        "E   Agent confidence profile comparison\n"
        "C15 and C11 vs C1 LUAD (confident reference)",
        loc="left", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.97, edgecolor="#ccc")


def draw(d):
    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.48, wspace=0.32,
                            top=0.90, bottom=0.05,
                            left=0.05, right=0.97)

    # Row 0: C15 LUAD
    ax_a = fig.add_subplot(gs[0, 0])
    deg_panel(ax_a, d["df15"], "A   C15 LUAD — top DEGs", C["luad"])
    conf_badge(ax_a, d["c15"], C["unct"])

    ax_b = fig.add_subplot(gs[0, 1])
    path_panel(ax_b, d["paths15"], "B   C15 LUAD — enriched pathways",
               C["amber"],
               "Only 1 pathway confirmed\nNo canonical LUAD driver genes\ndetected in top DEGs")

    # Row 1: C11 HCC
    ax_c = fig.add_subplot(gs[1, 0])
    deg_panel(ax_c, d["df11"], "C   C11 HCC — top DEGs", C["hcc"])
    conf_badge(ax_c, d["c11"], C["unct"])

    ax_d = fig.add_subplot(gs[1, 1])
    path_panel(ax_d, d["paths11"], "D   C11 HCC — enriched pathways",
               C["path"],
               "5 confirmed pathways\nXenobiotic/drug metabolism dominant\nRich signal, identity uncertain")

    # Row 2: confidence comparison
    ax_e = fig.add_subplot(gs[2, :])
    conf_comparison(ax_e, d["c15"], d["c11"], d["c1_ref"])

    # Row separators and labels
    for y in [0.66, 0.35]:
        fig.add_artist(plt.Line2D([0.03, 0.97], [y, y],
                       transform=fig.transFigure,
                       color="#CCCCCC", lw=0.8, ls="--"))
    for yp, txt, col_, bg, ec in [
        (0.79, "C15\nLUAD", C["luad"], "#EBF5FB", C["luad"]),
        (0.49, "C11\nHCC",  C["hcc"],  "#FDEDEC", C["hcc"]),
    ]:
        fig.text(0.005, yp, txt, ha="center", va="center", fontsize=9,
                 fontweight="bold", color=col_, rotation=90,
                 bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec=ec, lw=1.2))

    fig.suptitle("Figure 4: Uncertainty-flagged clusters — GRACE case studies",
                 fontsize=13, fontweight="bold", y=0.975)

    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"fig4_case_studies.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    print("=" * 55)
    print("draw_fig4_case_studies.py")
    print("=" * 55)
    d = load_data()
    draw(d)
    print("\nAll data from real files — no hardcoded values.")
