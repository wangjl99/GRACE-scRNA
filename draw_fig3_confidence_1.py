#!/usr/bin/env python3
"""
draw_fig3_confidence.py
=======================
Generates Figure 3: Cell Identity Agent — confidence comparison
LUAD (panels A, D) and HCC zero-shot (panel B), plus
distribution comparison (panel C).

Usage
-----
    cd /data/jwang58/lung_scrnaseq
    python3 draw_fig3_confidence.py

Output
------
    figures/fig3_confidence_scores.png
    figures/fig3_confidence_scores.pdf

Panels
------
A  LUAD per-cluster: GRACE v1 vs v2 vs GPT (full width)
B  HCC per-cluster:  GRACE v2 vs GPT (full width, zero-shot)
C  Violin distribution: v1 LUAD / v2 LUAD / v2 HCC / GPT
D  Stacked agent contribution to GRACE v2 c_overall — LUAD

Data sources
------------
LUAD : results/versionB_v3_backup.json   (agent_confidences per cluster)
HCC  : results/hcc/hcc_versionB_results.json  (overall_confidence per cluster)
GPT  : No uncertainty mechanism — c_overall = 0.0 for all clusters

Verified results (April 2026)
    LUAD v1: mean=0.407,  2/20 above threshold (0.50)
    LUAD v2: mean=0.546, 16/20 above threshold
    HCC  v2: mean=0.425,  9/25 above threshold
    GPT:     mean=0.000,  0 clusters (no mechanism)
"""

import json
import numpy as np
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

THRESHOLD = 0.50

C = {
    "v1":    "#9E9E9E",
    "v2":    "#43A047",
    "gpt":   "#F28E2B",
    "unct":  "#E53935",
    "mid":   "#607D8B",
    "luad":  "#1A5276",
    "hcc":   "#C62828",
    "deg":   "#CFD8DC",
    "path":  "#4FC3F7",
    "dis":   "#FFB74D",
    "cid":   "#81C784",
    "amber": "#7D3C00",
    "unc_h": "#FF8F00",
}

AUTHOR_L = ["T lymph","T lymph","Myeloid","B lymph","T lymph","T lymph",
            "Epithel","Epithel","Mast",   "Myeloid","Myeloid","Fibrobla",
            "B lymph","Epithel","T lymph","Epithel","Myeloid","Endothe",
            "T lymph","Epithel"]
AUTHOR_H = ["Hepato","Myeloid","T/NK","Myeloid","T/NK","Hepato",
            "Hepato","Endothe","B","Fibrobla","Hepato","Hepato",
            "Hepato","Hepato","Myeloid","Hepato","T/NK","B",
            "T/NK","Hepato","Hepato","T/NK","Myeloid","Myeloid","Myeloid"]

UNCERTAIN_H = {5, 10, 11, 12}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.family":      "sans-serif",
    "pdf.fonttype":     42,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def load_data():
    # ── LUAD ──────────────────────────────────────────────────────────────
    FALLBACK_L = [
        (0,1.0,0.30,0.60),(0,1.0,0.30,0.90),(0,1.0,0.30,0.90),(0,1.0,0.30,0.90),
        (0,1.0,0.30,0.35),(0,1.0,0.30,0.90),(0,0.0,0.30,0.35),(0,1.0,0.30,0.90),
        (0,1.0,0.30,0.59),(0,1.0,0.30,0.63),(0,1.0,0.30,0.90),(0,1.0,0.30,0.75),
        (0,0.0,0.30,0.88),(0,1.0,0.30,0.60),(0,1.0,0.30,0.90),(0,1.0,0.63,0.29),
        (0,0.8,0.30,0.89),(0,1.0,0.63,0.59),(0,1.0,0.30,0.90),(0,1.0,0.30,0.45),
    ]
    try:
        vb_l  = json.load(open(RES / "versionB_v3_backup.json"))
        vbd_l = {str(x["cluster"]): x for x in vb_l}
        agents_l = []
        for cl in range(20):
            ac = vbd_l.get(str(cl), {}).get(
                 "orchestration", {}).get("agent_confidences", {})
            agents_l.append((
                float(ac.get("deg",          0)),
                float(ac.get("pathway",       0)),
                float(ac.get("disease",       0)),
                float(ac.get("cell_identity",
                     ac.get("cell_id", ac.get("c_cell_id", 0)))),
            ))
        print("LUAD: loaded from versionB_v3_backup.json")
    except FileNotFoundError:
        print("WARNING: versionB_v3_backup.json not found — using verified defaults")
        agents_l = FALLBACK_L

    v1_l = [round((d+p+dis)/3,                     3) for d,p,dis,cid in agents_l]
    v2_l = [round(0.20*d+0.30*p+0.20*dis+0.30*cid, 3) for d,p,dis,cid in agents_l]

    # ── HCC ───────────────────────────────────────────────────────────────
    FALLBACK_H = [0.499,0.566,0.597,0.542,0.198,0.404,0.588,0.507,
                  0.532,0.504,0.187,0.413,0.399,0.170,0.207,0.409,
                  0.427,0.455,0.321,0.620,0.399,0.582,0.287,0.342,0.475]
    try:
        vb_h  = json.load(open(HCC_DIR / "hcc_versionB_results.json"))
        vbd_h = {str(x["cluster"]): x for x in vb_h}
        v2_h  = [round(float(vbd_h.get(str(cl), {}).get(
                 "orchestration", {}).get("overall_confidence", 0)), 3)
                 for cl in range(25)]
        print("HCC:  loaded from hcc_versionB_results.json")
    except FileNotFoundError:
        print("WARNING: hcc_versionB_results.json not found — using verified defaults")
        v2_h = FALLBACK_H

    v1_mean   = round(float(np.mean(v1_l)), 3)
    v2l_mean  = round(float(np.mean(v2_l)), 3)
    v2h_mean  = round(float(np.mean(v2_h)), 3)
    v1_above  = sum(1 for c in v1_l if c >= THRESHOLD)
    v2l_above = sum(1 for c in v2_l if c >= THRESHOLD)
    v2h_above = sum(1 for c in v2_h if c >= THRESHOLD)

    print(f"LUAD v1: mean={v1_mean}   above={v1_above}/20")
    print(f"LUAD v2: mean={v2l_mean}  above={v2l_above}/20")
    print(f"HCC  v2: mean={v2h_mean}  above={v2h_above}/25")

    return dict(
        agents_l=agents_l,
        v1_l=v1_l, v2_l=v2_l, v2_h=v2_h,
        v1_mean=v1_mean, v2l_mean=v2l_mean, v2h_mean=v2h_mean,
        v1_above=v1_above, v2l_above=v2l_above, v2h_above=v2h_above,
    )


def panel_luad_bars(ax, d):
    x, bw = np.arange(20), 0.26
    ax.bar(x-bw,  d["v1_l"], bw, color=C["v1"], alpha=0.82,
           label="GRACE v1 (3 agents)", edgecolor="white")
    ax.bar(x,     d["v2_l"], bw, color=C["v2"], alpha=0.88,
           label="GRACE v2 (4 agents)", edgecolor="white")
    ax.bar(x+bw,  [0]*20,    bw, color=C["gpt"],alpha=0.70,
           label="GPT-5.4 naive (no uncertainty mechanism)", edgecolor="white")
    ax.axhline(THRESHOLD,       color=C["unct"],ls="--",lw=1.4,alpha=0.8,
               label="Threshold (0.50)")
    ax.axhline(d["v1_mean"],  color=C["v1"], ls=":",lw=1.2,alpha=0.7,
               label=f"v1 mean={d['v1_mean']:.3f}")
    ax.axhline(d["v2l_mean"], color=C["v2"], ls=":",lw=1.2,alpha=0.7,
               label=f"v2 mean={d['v2l_mean']:.3f}")
    ax.add_patch(plt.Rectangle(
        (14.35+bw, -0.01), 0.70*bw+0.04, d["v2_l"][15]+0.04,
        fill=False, ec=C["amber"], lw=2, clip_on=False))
    ax.text(15, d["v2_l"][15]+0.05, "★C15",
            ha="center", fontsize=7, color=C["amber"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}\n{AUTHOR_L[i][:6]}" for i in range(20)],fontsize=8)
    ax.set_ylabel("c_overall", fontsize=10.5)
    ax.set_ylim(0, 0.90)
    ax.set_title(
        f"A   LUAD (GSE131907, 20 clusters)  —  "
        f"v1: mean={d['v1_mean']:.3f}, {d['v1_above']}/20 above threshold  |  "
        f"v2: mean={d['v2l_mean']:.3f}, {d['v2l_above']}/20 above threshold  |  "
        "GPT: c=0 all clusters",
        loc="left", fontweight="bold", fontsize=11, color=C["luad"])
    ax.legend(fontsize=9, loc="upper right", framealpha=0.97,
              ncol=3, edgecolor="#ccc")


def panel_hcc_bars(ax, d):
    x, bw = np.arange(25), 0.30
    bc_h = [C["unc_h"] if i in UNCERTAIN_H
            else C["v2"] if v >= THRESHOLD
            else C["unct"]
            for i, v in enumerate(d["v2_h"])]
    ax.bar(x-bw/2, d["v2_h"], bw, color=bc_h, alpha=0.85,
           edgecolor="white", label="GRACE v2 (4 agents)")
    ax.bar(x+bw/2, [0]*25,    bw, color=C["gpt"], alpha=0.70,
           edgecolor="white",
           label="GPT-5.4 naive (no uncertainty mechanism)")
    ax.axhline(THRESHOLD,       color=C["unct"],ls="--",lw=1.4,alpha=0.8,
               label="Threshold (0.50)")
    ax.axhline(d["v2h_mean"], color=C["hcc"], ls=":",lw=1.2,alpha=0.7,
               label=f"HCC mean={d['v2h_mean']:.3f}")
    ax.axhline(d["v2l_mean"], color=C["luad"],ls=":",lw=1.0,alpha=0.5,
               label=f"LUAD mean={d['v2l_mean']:.3f} (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{i}\n{AUTHOR_H[i][:5]}" for i in range(25)], fontsize=7.5)
    ax.set_ylabel("c_overall", fontsize=10.5)
    ax.set_ylim(0, 0.90)
    ax.set_title(
        f"B   HCC (GSE149614, zero-shot, 25 clusters)  —  "
        f"v2: mean={d['v2h_mean']:.3f}, {d['v2h_above']}/25 above threshold  |  "
        "Orange = high-uncertainty hepatocyte clusters (C5, C10, C11, C12)",
        loc="left", fontweight="bold", fontsize=11, color=C["hcc"])
    ax.legend(fontsize=9, loc="upper right", framealpha=0.97,
              ncol=3, edgecolor="#ccc")


def panel_violin(ax, d):
    rng = np.random.default_rng(42)
    for pos, dat, col in [
        (1, d["v1_l"], C["v1"]),
        (2, d["v2_l"], C["v2"]),
        (3, d["v2_h"], C["hcc"]),
    ]:
        pv = ax.violinplot([dat], positions=[pos], widths=0.5,
                            showmeans=True, showmedians=False)
        for pc in pv["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.75)
        pv["cmeans"].set_color("black"); pv["cmeans"].set_lw(2)
        ax.scatter(pos + rng.uniform(-0.10, 0.10, len(dat)), dat,
                   color=col, s=35, alpha=0.80,
                   edgecolors="white", lw=0.5, zorder=3)
    ax.plot([3.7, 4.3], [0, 0], color=C["gpt"], lw=3,
            alpha=0.85, solid_capstyle="round")
    ax.text(4, 0.03, "GPT: c=0\n(no mechanism)",
            ha="center", fontsize=8, color=C["gpt"], fontweight="bold")
    ax.axhline(THRESHOLD, color=C["unct"], ls="--", lw=1.4, alpha=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([
        f"GRACE v1\nLUAD\nmean={d['v1_mean']:.3f}",
        f"GRACE v2\nLUAD\nmean={d['v2l_mean']:.3f}",
        f"GRACE v2\nHCC (zero-shot)\nmean={d['v2h_mean']:.3f}",
        "GPT-5.4\nnaive",
    ], fontsize=9.5)
    ax.set_ylabel("c_overall", fontsize=10.5)
    ax.set_ylim(-0.05, 0.90)
    ax.set_title("C   Confidence distribution comparison",
                 loc="left", fontweight="bold", fontsize=11)
    ax.text(0.02, 0.97,
            f"LUAD v1→v2: +{d['v2l_above']-d['v1_above']} clusters above threshold\n"
            f"LUAD v2: {d['v2l_above']}/20  vs  HCC v2: {d['v2h_above']}/25\n"
            "Lower HCC confidence → lower calibration (Figure 6D–F)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C["mid"], lw=1.0))


def panel_stacked(ax, d):
    agents = d["agents_l"]; v2_l = d["v2_l"]
    x = np.arange(20)
    w_deg  = [0.20*a[0] for a in agents]
    w_path = [0.30*a[1] for a in agents]
    w_dis  = [0.20*a[2] for a in agents]
    w_cid  = [0.30*a[3] for a in agents]
    ax.bar(x, [0.012]*20, 0.65, color=C["deg"],  alpha=0.90,
           edgecolor="white",
           label="Agent 1: DEG validator (×0.20) — c=0\n"
                 "(LUAD TME markers absent from UniProt Swiss-Prot)")
    ax.bar(x, w_path, 0.65, color=C["path"], alpha=0.90,
           edgecolor="white", label="Agent 2: Reactome pathway (×0.30)",
           bottom=[0.012]*20)
    ax.bar(x, w_dis,  0.65, color=C["dis"],  alpha=0.90,
           edgecolor="white", label="Agent 3: Disease / DisGeNET (×0.20)",
           bottom=[a+b for a,b in zip([0.012]*20, w_path)])
    ax.bar(x, w_cid,  0.65, color=C["cid"],  alpha=0.90,
           edgecolor="white", label="Agent 4: CellMarker 2.0 (×0.30) ★",
           bottom=[a+b+c for a,b,c in zip([0.012]*20, w_path, w_dis)])
    ax.axhline(THRESHOLD, color=C["unct"], ls="--", lw=1.4, alpha=0.8,
               label="Threshold (0.50)")
    ax.add_patch(plt.Rectangle(
        (14.35, 0), 0.65, v2_l[15]+0.03,
        fill=False, ec=C["amber"], lw=2.0, clip_on=False, zorder=5))
    ax.text(15, v2_l[15]+0.05, "★C15",
            ha="center", fontsize=7, color=C["amber"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(20)], fontsize=8.5)
    ax.set_ylabel("Weighted agent contribution to c_overall", fontsize=10.5)
    ax.set_ylim(0, 0.90)
    ax.set_title("D   Agent contribution to GRACE v2 c_overall — LUAD",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right",
              framealpha=0.97, edgecolor="#ccc")


def draw(d):
    fig = plt.figure(figsize=(26, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.42, wspace=0.28,
                            top=0.93, bottom=0.05,
                            left=0.06, right=0.97)
    panel_luad_bars(fig.add_subplot(gs[0, 0:2]), d)
    panel_hcc_bars( fig.add_subplot(gs[1, 0:2]), d)
    panel_violin(   fig.add_subplot(gs[2, 0]),   d)
    panel_stacked(  fig.add_subplot(gs[2, 1]),   d)

    for y in [0.67, 0.34]:
        fig.add_artist(plt.Line2D(
            [0.03, 0.97], [y, y], transform=fig.transFigure,
            color="#CCCCCC", lw=0.8, ls="--"))

    for yp, txt, col, bg, ec in [
        (0.80, "LUAD",             C["luad"], "#EBF5FB", C["luad"]),
        (0.50, "HCC\n(zero-shot)", C["hcc"],  "#FDEDEC", C["hcc"]),
    ]:
        fig.text(0.01, yp, txt, ha="center", va="center",
                 fontsize=10, fontweight="bold", color=col, rotation=90,
                 bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec=ec, lw=1.2))

    fig.suptitle("Figure 3: Cell Identity Agent",
                 fontsize=14, fontweight="bold", y=0.975)

    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"fig3_confidence_scores.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    print("=" * 55)
    print("draw_fig3_confidence.py — Figure 3 generation")
    print("=" * 55)
    d = load_data()
    draw(d)
    print("\nData sources:")
    print("  LUAD: results/versionB_v3_backup.json")
    print("  HCC:  results/hcc/hcc_versionB_results.json")
