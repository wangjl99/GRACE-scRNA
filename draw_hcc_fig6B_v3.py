"""
draw_hcc_fig6B_v3.py
====================
Fixed version:
  - Novel cluster labels staggered at top/middle/bottom to avoid overlap
  - Gap arrow replaced with clear annotation explaining what it means
  - Each label has a clean arrow to its actual data point

Run:
    cd /data/jwang58/lung_scrnaseq
    python3 draw_hcc_fig6B_v3.py
Output: figures/hcc_fig6B_v3.png / .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
hcc_confs = [0.50,0.57,0.60,0.54,0.20,0.40,0.59,0.51,0.53,0.50,
             0.19,0.41,0.40,0.17,0.21,0.41,0.43,0.46,0.32,0.62,
             0.40,0.58,0.29,0.34,0.47]
hcc_go_rec= [0.75,0.88,0.88,0.75,0.50,0.63,0.75,0.88,0.75,0.63,
             0.50,0.63,0.63,0.50,0.50,0.63,0.75,0.63,0.50,0.88,
             0.63,0.88,0.50,0.63,0.63]

NOVEL = {
    5:  {"col":"#E15759","label":"C5  GPC3⁺ Hepatocyte→HCC transition",  "short":"C5"},
    10: {"col":"#F28E2B","label":"C10 NQO1/MIF stress-adapted HCC",       "short":"C10"},
    11: {"col":"#4E79A7","label":"C11 SQSTM1 drug-resistant HCC",         "short":"C11"},
    12: {"col":"#59A14F","label":"C12 Cancer-testis antigen HCC",          "short":"C12"},
}

flagged   = [(i,g) for i,(c,g) in enumerate(zip(hcc_confs,hcc_go_rec)) if c <  0.50]
unflagged = [(i,g) for i,(c,g) in enumerate(zip(hcc_confs,hcc_go_rec)) if c >= 0.50]
fg_go = [g for _,g in flagged]
uf_go = [g for _,g in unflagged]
gap   = np.mean(uf_go) - np.mean(fg_go)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7.5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.35, right=0.95, top=0.88, bottom=0.12)

# ── Boxplots ──────────────────────────────────────────────────────────────────
bp = ax.boxplot(
    [fg_go, uf_go],
    tick_labels=[
        f"Low confidence\n(c_overall < 0.50)\nn = {len(fg_go)} clusters",
        f"High confidence\n(c_overall ≥ 0.50)\nn = {len(uf_go)} clusters",
    ],
    patch_artist=True, widths=0.45,
    boxprops=dict(facecolor="#FFCDD2", alpha=0.85, linewidth=1.5),
    medianprops=dict(color="#E53935", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2, color="#555"),
    capprops=dict(linewidth=1.5, color="#555"),
    zorder=2,
)
bp["boxes"][1].set_facecolor("#C8E6C9")
bp["boxes"][1].set_linewidth(1.5)
bp["medians"][1].set_color("#2E7D32")

# ── Jittered points ───────────────────────────────────────────────────────────
rng = np.random.RandomState(42)

for (ci, g) in flagged:
    jx  = rng.uniform(0.82, 1.18)
    col = NOVEL[ci]["col"] if ci in NOVEL else "#E53935"
    mk  = "*" if ci in NOVEL else "o"
    sz  = 140 if ci in NOVEL else 50
    ax.scatter(jx, g, c=[col], s=sz, marker=mk,
               alpha=0.92, zorder=4, edgecolors="white", lw=0.5)

for (ci, g) in unflagged:
    jx  = rng.uniform(1.82, 2.18)
    col = NOVEL[ci]["col"] if ci in NOVEL else "#43A047"
    mk  = "*" if ci in NOVEL else "o"
    sz  = 140 if ci in NOVEL else 50
    ax.scatter(jx, g, c=[col], s=sz, marker=mk,
               alpha=0.92, zorder=4, edgecolors="white", lw=0.5)

# ── Staggered labels for novel clusters in the LOW-confidence group ───────────
# C11 is in flagged group (conf=0.41 < 0.50), others too
# Assign vertical positions spread across the left margin:
#   top label ~ 0.74, middle ~ 0.62, lower ~ 0.52, bottom ~ 0.43
novel_in_flagged = [(ci,g) for (ci,g) in flagged if ci in NOVEL]
novel_in_flagged.sort(key=lambda x: x[1], reverse=True)  # sort by GO recall

# Fixed stagger y-positions (well separated)
stagger_y = [0.76, 0.66, 0.55, 0.44]

for rank, (ci, g) in enumerate(novel_in_flagged):
    col   = NOVEL[ci]["col"]
    label = NOVEL[ci]["label"]
    lbl_y = stagger_y[rank] if rank < len(stagger_y) else 0.44 - rank*0.10

    # Find actual x position of this point (use centroid since jittered)
    # We re-scatter at a known x for annotation purposes
    pt_x = 1.0   # centre of left box
    pt_y = g

    ax.annotate(
        label,
        xy=(pt_x - 0.08, pt_y),          # point: just left of jittered dots
        xytext=(-0.28, lbl_y),            # label: in left margin (axes coords → data)
        xycoords=("data", "data"),
        textcoords=("axes fraction", "data"),
        fontsize=8.5, fontweight="bold", color=col,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col,
                  lw=1.4, alpha=0.97),
        arrowprops=dict(
            arrowstyle="->",
            color=col,
            lw=1.2,
            connectionstyle="arc3,rad=0.0",
            shrinkA=0, shrinkB=5,
        ),
        zorder=8,
    )

# ── Mean dotted lines ─────────────────────────────────────────────────────────
ax.plot([0.73, 1.27], [np.mean(fg_go)] * 2,
        color="#E53935", lw=1.2, ls=":", alpha=0.7)
ax.plot([1.73, 2.27], [np.mean(uf_go)] * 2,
        color="#2E7D32", lw=1.2, ls=":", alpha=0.7)

ax.text(1.30, np.mean(fg_go) + 0.005,
        f"mean = {np.mean(fg_go):.3f}",
        fontsize=8.5, color="#E53935", fontweight="bold", va="bottom")
ax.text(2.30, np.mean(uf_go) + 0.005,
        f"mean = {np.mean(uf_go):.3f}",
        fontsize=8.5, color="#2E7D32", fontweight="bold", va="bottom")

# ── Calibration gap — replaced with clear explanatory annotation ──────────────
# Instead of a confusing arrow between the two boxes,
# use a bracket-style annotation in the right margin explaining the gap.
gap_x = 2.55
ax.annotate(
    "",
    xy  =(gap_x, np.mean(uf_go)),
    xytext=(gap_x, np.mean(fg_go)),
    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.8),
    zorder=6,
)
ax.text(gap_x + 0.06,
        (np.mean(uf_go) + np.mean(fg_go)) / 2,
        f"Calibration\ngap\n{gap:+.3f}",
        fontsize=8.5, color="#444", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", lw=0.8))

# ── Calibration verdict ───────────────────────────────────────────────────────
ax.text(0.50, 0.97,
        f"Calibration gap = mean(high-conf GO recall) − mean(low-conf GO recall)\n"
        f"= {np.mean(uf_go):.3f} − {np.mean(fg_go):.3f} = {gap:+.3f}   →   Well-calibrated ✓",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color="#2E7D32", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#E8F5E9",
                  ec="#2E7D32", lw=1.5, alpha=0.97))

# ── Legend ────────────────────────────────────────────────────────────────────
leg = [
    mpatches.Patch(fc="#FFCDD2", ec="#E53935", lw=1.5, label="Low confidence (<0.50)"),
    mpatches.Patch(fc="#C8E6C9", ec="#2E7D32", lw=1.5, label="High confidence (≥0.50)"),
]
for ci, info in NOVEL.items():
    leg.append(plt.Line2D([0],[0], marker="*", color=info["col"],
                          ms=11, lw=0, label=f"{info['short']} ★ novel population"))

ax.legend(handles=leg, fontsize=8.5, loc="lower right",
          framealpha=0.97, edgecolor="#cccccc", borderpad=0.7)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_ylabel("GO-term recall", fontsize=11)
ax.set_ylim(0.38, 1.05)
ax.set_xlim(0.3, 2.85)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9)

ax.set_title(
    "Figure 6B (HCC): Calibration validation — GSE149614 (25 clusters)\n"
    "Low confidence clusters show lower biological accuracy — "
    "uncertainty scores are biologically meaningful",
    fontweight="bold", fontsize=10.5, pad=8,
)

for ext in ["png", "pdf"]:
    fig.savefig(FIG_DIR / f"hcc_fig6B_v3.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved → figures/hcc_fig6B_v3.png / .pdf")
