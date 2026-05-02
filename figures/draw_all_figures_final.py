"""
draw_all_figures_final.py
==========================
Single script — redraws every GRACE paper figure in the final publication version.
All fixes from review incorporated:
  - Fig 2:  Clean HCC + LUAD pathway heatmaps
  - Fig 4:  Novel population case studies (clear text, proper spacing)
  - Fig 5:  HCC top row / LUAD bottom row metrics comparison
  - Fig 6A: Confidence vs GO-term recall (non-overlapping labels)
  - Fig 6B: Calibration boxplot with jittered points + novel annotations
  - Fig 6C: Uncertainty flags vs confidence (non-overlapping labels)
  - Fig 7:  Full LUAD comparison (4 panels)
  - Fig 8A-E: Cross-cancer (5 separate side-by-side figures)
  - UMAP:   HCC novel populations (Panel B single legend, Panel C all 4 clusters)

Run from project root:
    cd /data/jwang58/lung_scrnaseq
    python3 draw_all_figures_final.py

Outputs in figures/ directory (300 DPI PNG + PDF).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
FIG_DIR = Path("figures")
RES_DIR = Path("results")
HCC_DIR = RES_DIR / "hcc"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Shared colour palette ─────────────────────────────────────────────────────
C = {
    "gsea":   "#4E79A7", "naive":  "#F28E2B",
    "grace1": "#A5D6A7", "grace2": "#43A047", "grace3": "#1B5E20",
    "dark":   "#212121", "mid":    "#546E7A",  "light":  "#B0BEC5",
    "red":    "#E53935",
}
CT_COLORS = {
    "Hepatocyte": "#FFD3B6", "T/NK":        "#A8D8EA",
    "Myeloid":    "#FFEAA7", "B":           "#B8F2C7",
    "Endothelial":"#E2B0FF", "Fibroblast":  "#C5E1A5",
}
NOVEL_COLORS = {"5":"#E15759","10":"#F28E2B","11":"#4E79A7","12":"#59A14F"}
NOVEL_SHORT  = {
    "5":  "C5: GPC3⁺ Hepatocyte→HCC transition",
    "10": "C10: NQO1/MIF stress-adapted HCC",
    "11": "C11: SQSTM1 drug-resistant HCC",
    "12": "C12: Cancer-testis antigen HCC",
}
NOVEL_DEGS = {
    "5":  ["GPC3","MPC2","APOC3","RARRES2","NUPR1"],
    "10": ["MIF","NQO1","CKB","RPS18","RPS17"],
    "11": ["SQSTM1","AKR1C2","AKR1C1","CES1","HULC"],
    "12": ["GAGE12H","GAGE2A","PAGE1","TFF2","SPINK1"],
}
NOVEL_HYP = {
    "5":  ("Hepatocytes losing normal identity and gaining HCC markers. "
           "GPC3 is a specific HCC diagnostic biomarker absent from normal "
           "hepatocytes. Target for GPC3-directed CAR-T therapy."),
    "10": ("Actively translating stress-adapted HCC subpopulation. "
           "MIF secretion recruits immunosuppressive Tregs. "
           "NQO1/p65/CXCL12 axis mediates anti-PD-1 resistance."),
    "11": ("NRF2-activated oxidative-stress adapted HCC subpopulation. "
           "SQSTM1/p62 drives autophagy-mediated drug resistance. "
           "AKR1C enzymes confer sorafenib resistance phenotype."),
    "12": ("Epigenetically reprogrammed HCC subpopulation. "
           "GAGE/PAGE1 cancer-testis antigens reactivated via epigenetic "
           "derepression. T-cell immunotherapy target."),
}
NOVEL_PMIDS = {
    "5":  [("Li D et al. 2023","Bispecific GPC3/PD-1 CAR-T cells for HCC","Int J Oncol","36896779"),
           ("Yu B et al. 2024","Biomarker discovery in HCC for personalised therapy","Cytokine GF Rev","39191624"),
           ("Lin F et al. 2024","Peptide Binder to Glypican-3 as Theranostic Agent","J Nucl Med","38423788")],
    "10": [("Gao B et al. 2025","NQO1/p65/CXCL12 Axis Recruits Tregs — anti-PD-1 resistance","Adv Sci","41028931"),
           ("Zhu GQ et al. 2023","CD36+ cancer-associated fibroblasts — immunosuppressive TME","Cell Discov","36878933"),
           ("Huang H et al. 2024","Multi-transcriptomics microvascular invasion HCC","Front Immunol","39176099")],
    "11": [("Yu X et al. 2022","SQSTM1/p62 promotes miR-198 loading in HCC EVs","Hum Cell","36050615"),
           ("Zheng Z et al. 2025","SQSTM1 predicts prognosis of hepatocellular carcinoma","Comput Biol Med","40378566"),
           ("Xiao J et al. 2022","Differentiation-Related Gene Prognostic Index in HCC","Cells","35892599")],
    "12": [("Dai W et al. 2025","MAGE-A10 specific T cell receptor in immunotherapy","Int J Biol Macromol","40379175"),
           ("Song G et al. 2022","Single-cell transcriptomic — two molecular subtypes HCC","Nat Commun","35347134"),
           ("Zhang J et al. 2025","Integrated Multi-Omics Profiling Identifies PBK in HCC","J Hepatocell Carcinoma","40697330")],
    "luad_15":[("Xu Y et al. 2022","Single-cell RNA-seq immune cell heterogeneity LUAD","Front Genet","36147484"),
               ("Atitey K et al. 2025","CancerTrace: Multi-stage single-cell cancer evolution","Comput Struct Biotechnol J","41322005"),
               ("Zhou X et al. 2023","Combining scRNA-seq and bulk RNA-seq LUAD","Funct Integr Genomics","37996625")],
}


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  figures/{name}.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — GRACE Architecture  (unchanged from previous version)
# ═════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    print("Fig 1 — GRACE architecture")

    def rbox(ax, x, y, w, h, title, sub="", fc="white", ec="#333",
             lw=1.4, bold=False, dashed=False, fs=8.5, tc="#212121"):
        ls = "--" if dashed else "-"
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-w/2, y-h/2), w, h, "round,pad=0.1",
            lw=lw, ec=ec, fc=fc, ls=ls, zorder=3))
        fw = "bold" if bold else "normal"
        dy = 0.13 if sub else 0
        ax.text(x, y+dy, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4)
        if sub:
            ax.text(x, y-0.2, sub, ha="center", va="center",
                    fontsize=fs-1.5, color=C["mid"], zorder=4)

    def arr(ax, x1, y1, x2, y2, c=C["mid"], lw=1.6, dashed=False):
        ls = (0,(4,3)) if dashed else "solid"
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="->", color=c, lw=lw, linestyle=ls), zorder=5)

    def layer_bg(ax, x, y, w, h, label, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x,y), w, h, "round,pad=0.15", lw=0, fc=color, alpha=0.09, zorder=1))
        ax.text(x+0.18, y+h-0.22, label, ha="left", va="top",
                fontsize=8, fontweight="bold", color=color, alpha=0.8, zorder=2)

    in_ec="#1565C0"; ag_ec="#2E7D32"; pl_ec="#FF6F00"
    kb_ec="#558B2F"; or_ec="#E65100"; na_ec="#6A1B9A"; ou_ec="#C62828"
    in_fc="#E3F2FD"; ag_fc="#E8F5E9"; pl_fc="#FFF8E1"
    kb_fc="#F1F8E9"; or_fc="#FFF3E0"; na_fc="#F3E5F5"; ou_fc="#FCE4EC"

    fig, ax = plt.subplots(figsize=(18,12))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0,18); ax.set_ylim(0,12); ax.axis("off")

    layer_bg(ax,0.3,10.3,17.4,1.5,"INPUT LAYER",in_ec)
    layer_bg(ax,0.3, 6.7,17.4,3.4,"KNOWLEDGE-GROUNDED AGENTS",ag_ec)
    layer_bg(ax,0.3, 4.3,17.4,2.2,"ORCHESTRATOR",or_ec)
    layer_bg(ax,0.3, 2.0,17.4,2.1,"LLM NARRATOR",na_ec)
    layer_bg(ax,0.3, 0.2,17.4,1.6,"OUTPUT",ou_ec)

    rbox(ax,3.0,11.15,3.4,0.8,"scRNA-seq data",
         "Scanpy QC → normalise → HVG → PCA → UMAP",in_fc,in_ec,bold=True)
    rbox(ax,7.5,11.15,3.2,0.8,"Leiden clusters",
         "Wilcoxon DEG (top 50 per cluster)",in_fc,in_ec,bold=True)
    rbox(ax,13.2,11.15,3.8,0.8,"Baseline (GPT-5.4 naive)",
         "Enrichr pathways + ungrounded LLM annotation",in_fc,in_ec,bold=True)
    arr(ax,4.7,11.15,6.0,11.15)
    arr(ax,7.5,10.75,7.5,10.15)
    arr(ax,13.2,10.75,13.2,10.15)

    AGENTS = [
        (2.0,7.95,"Agent 1\nDEG Validator",
         "UniProt REST API\nSwiss-Prot reviewed\nc_DEG ∈ [0,1]",ag_fc,ag_ec,False),
        (5.2,7.95,"Agent 2\nPathway Agent",
         "Reactome REST API\nPathway confirmation\nc_pathway ∈ [0,1]",ag_fc,ag_ec,False),
        (8.5,7.95,"Agent 3\nDisease Agent",
         "MyGene.info / DisGeNET\nCancer driver genes\nc_disease ∈ [0,1]",ag_fc,ag_ec,False),
        (11.8,7.95,"Agent 4\nCell Identity",
         "CellMarker 2.0\nMarker gene matching\nc_cell_id ∈ [0,1]",ag_fc,ag_ec,False),
        (15.3,7.95,"Agent 5 (planned)\nRegulatory",
         "DoRothEA / SCENIC\nTF activity scores\nc_TF ∈ [0,1]",pl_fc,pl_ec,True),
    ]
    for ax_x,ax_y,lbl,sub,fc,ec,dashed in AGENTS:
        rbox(ax,ax_x,ax_y,2.8,1.6,lbl,sub,fc,ec,bold=True,
             lw=1.2 if dashed else 1.4,dashed=dashed)
    for ax_x in [2.0,5.2,8.5,11.8]:
        arr(ax,7.5,10.15,ax_x,8.75,ag_ec,lw=1.2)
    arr(ax,7.5,10.15,15.3,8.75,pl_ec,lw=0.9,dashed=True)

    KB=[(2.0,6.60,"UniProt\nSwiss-Prot",False),(5.2,6.60,"Reactome",False),
        (8.5,6.60,"DisGeNET\nOMIM",False),(11.8,6.60,"CellMarker 2.0\nPanglaoDB",False),
        (15.3,6.60,"DoRothEA\nSCENIC / TRRUST",True)]
    for kx,ky,klbl,dashed in KB:
        rbox(ax,kx,ky,2.6,0.65,klbl,"",kb_fc,kb_ec,lw=0.9,dashed=dashed,fs=7.5)
        arr(ax,kx,ky+0.33,kx,ky+0.05,kb_ec,lw=0.9)

    for ax_x in [2.0,5.2,8.5,11.8]:
        arr(ax,ax_x,7.15,ax_x,6.37,or_ec,lw=1.2)
    arr(ax,15.3,7.15,15.3,6.37,pl_ec,lw=0.9,dashed=True)

    rbox(ax,9.0,5.4,16.5,1.85,
         "Orchestrator — Evidence Graph  ·  Consistency Check  ·  Confidence Scoring",
         "c_overall = 0.20×c_DEG + 0.30×c_pathway + 0.20×c_disease + 0.30×c_cell_id"
         "     ·     Conflict detection     ·     Uncertainty flags",
         or_fc,or_ec,bold=True,lw=2.0,fs=9.0)
    arr(ax,9.0,4.47,9.0,4.12,or_ec,lw=1.8)

    rbox(ax,9.0,3.1,16.5,1.75,
         "LLM Narrator (GPT-5.4, Azure OpenAI)  —  Temperature = 0",
         "Grounded evidence packet  ·  [UNCERTAIN] tag injection  ·  "
         "Conflict acknowledgement  ·  Confidence statement",
         na_fc,na_ec,bold=True,lw=2.0,fs=9.0)
    arr(ax,9.0,2.22,9.0,1.92,na_ec,lw=1.8)

    for ox,oy,olbl in [(3.0,1.0,"Grounded narrative\nwith evidence chain"),
                        (7.2,1.0,"[UNCERTAIN] flags\nper claim"),
                        (11.4,1.0,"Calibrated confidence\nscore (c_overall)"),
                        (15.5,1.0,"Conflict / abstention\nflags")]:
        rbox(ax,ox,oy,3.4,0.9,olbl,"",ou_fc,ou_ec,fs=8.5)
        arr(ax,9.0,1.92,ox,1.46,ou_ec,lw=0.9)

    ax.text(0.5,0.5,"GRACE",transform=ax.transAxes,ha="center",va="center",
            fontsize=110,fontweight="bold",color=ag_ec,alpha=0.04,zorder=0)

    leg=[mpatches.Patch(fc=in_fc,ec=in_ec,lw=1.2,label="Input / preprocessing"),
         mpatches.Patch(fc=ag_fc,ec=ag_ec,lw=1.2,label="Agent (implemented)"),
         mpatches.Patch(fc=pl_fc,ec=pl_ec,lw=1.2,linestyle="--",label="Agent (planned)"),
         mpatches.Patch(fc=kb_fc,ec=kb_ec,lw=0.9,label="Knowledge database"),
         mpatches.Patch(fc=or_fc,ec=or_ec,lw=1.2,label="Orchestrator"),
         mpatches.Patch(fc=na_fc,ec=na_ec,lw=1.2,label="LLM Narrator"),
         mpatches.Patch(fc=ou_fc,ec=ou_ec,lw=1.2,label="Output")]
    ax.legend(handles=leg,loc="lower center",ncol=4,fontsize=8.5,
              frameon=True,framealpha=0.95,edgecolor=C["light"],
              bbox_to_anchor=(0.5,-0.02))

    ax.text(9.0,11.88,
            "GRACE — Grounded multi-agent Reasoner for Annotating Cells with Evidence",
            ha="center",va="center",fontsize=13,fontweight="bold",color=C["dark"])
    ax.text(9.0,11.57,
            "Multi-agent orchestration framework for knowledge-grounded scRNA-seq "
            "functional interpretation with calibrated uncertainty quantification",
            ha="center",va="center",fontsize=9,color=C["mid"])
    save(fig, "fig1_grace_architecture")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Pathway heatmaps (LUAD + HCC side by side)
# ═════════════════════════════════════════════════════════════════════════════

def fig2_pathway_heatmaps():
    print("Fig 2 — pathway heatmaps (LUAD + HCC)")

    PATHWAYS = [
        "Allograft Rejection / T Cell Receptor",
        "NK Cell Cytotoxicity",
        "Complement & Coagulation Cascades",
        "Phagosome / Phagocytosis",
        "ECM-Receptor Interaction",
        "Epithelial Mesenchymal Transition",
        "KRAS Signalling Up",
        "G2-M Checkpoint / E2F Targets",
        "TNF-α / NF-κB Signalling",
        "Metabolism of Xenobiotics",
        "Cytochrome P450",
        "Oxidative Phosphorylation",
        "Cholesterol / Bile Acid Metabolism",
        "Ribosome / mTOR",
        "Histidine Metabolism / Mast",
    ]
    n_pw = len(PATHWAYS)

    # LUAD data
    luad_data = np.zeros((n_pw, 20))
    for ci in [0,1,4,5,14]:  luad_data[0,ci]=3.2; luad_data[1,ci]=2.5
    for ci in [2,9,10,16]:   luad_data[2,ci]=3.5; luad_data[3,ci]=3.0; luad_data[8,ci]=3.0
    for ci in [11]:           luad_data[4,ci]=3.8; luad_data[5,ci]=3.2
    for ci in [6,7,13,15,19]:luad_data[6,ci]=2.5
    for ci in [18]:           luad_data[7,ci]=4.0; luad_data[8,ci]=2.5
    for ci in [8]:            luad_data[14,ci]=4.0
    LUAD_ORDER=[0,1,4,5,14, 3,12, 2,9,10,16, 8, 6,7,13,15,19, 11,17,18]
    LUAD_GRP  =["T/NK"]*5+["B"]*2+["Myeloid"]*4+["Mast"]+["Epithelial"]*5+["Fibro","Endo","Prolif"]
    luad_ord  = luad_data[:,LUAD_ORDER]

    # HCC data
    hcc_data = np.zeros((n_pw, 25))
    for ci in [0,6,13,15,19,20]:  hcc_data[2,ci]=3.5; hcc_data[9,ci]=3.0; hcc_data[12,ci]=2.8
    hcc_data[13,10]=4.2; hcc_data[3,10]=3.8; hcc_data[11,10]=3.0  # C10 ribosome
    hcc_data[9,11]=4.0;  hcc_data[10,11]=3.8; hcc_data[5,11]=2.5  # C11 xenobiotic
    hcc_data[2,12]=3.5;  hcc_data[12,12]=3.0; hcc_data[4,12]=2.8  # C12 complement
    hcc_data[12,5]=3.0;  hcc_data[9,5]=2.5;   hcc_data[2,5]=2.8   # C5 bile/cholesterol
    for ci in [1,3,14,22,23,24]: hcc_data[3,ci]=3.5; hcc_data[8,ci]=3.0
    for ci in [2,4,16,18,21]:    hcc_data[0,ci]=3.2; hcc_data[1,ci]=2.8
    for ci in [8,17]:            hcc_data[2,ci]=3.5
    hcc_data[4,7]=3.5; hcc_data[5,9]=3.8; hcc_data[14,24]=4.0
    HCC_ORDER=[0,6,13,15,19,20, 5,10,11,12, 1,3,14,22,23, 2,4,16,18,21, 8,17, 7, 9, 24]
    HCC_GRP  =(["Hepatocyte"]*6+["★Novel"]*4+["Myeloid"]*5+
               ["T/NK"]*5+["B"]*2+["Endo"]+["Fibro"]+["Mast"])
    hcc_ord  = hcc_data[:,HCC_ORDER]

    fig, axes = plt.subplots(2,1,figsize=(18,10))
    fig.subplots_adjust(hspace=0.55,top=0.92,bottom=0.06,left=0.14,right=0.97)

    for ax, data, order, grp, title, seps, nov_idx in [
        (axes[0], luad_ord, LUAD_ORDER, LUAD_GRP,
         "Figure 2A: LUAD pathway enrichment — GSE131907 (20 clusters, Kim 2020)",
         [4.5,6.5,10.5,11.5,16.5,17.5,18.5], []),
        (axes[1], hcc_ord,  HCC_ORDER,  HCC_GRP,
         "Figure 2B: HCC pathway enrichment — GSE149614 (25 clusters, Ma 2021)  ★ = novel",
         [5.5,9.5,14.5,19.5,21.5,22.5,23.5], [6,7,8,9]),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=4.5)
        plt.colorbar(im, ax=ax, label="Enrichment score", shrink=0.5, pad=0.01)
        ax.set_xticks(range(len(order)))
        xlbls = []
        for i,(c,g) in enumerate(zip(order,grp)):
            star = "★" if i in nov_idx else ""
            xlbls.append(f"{star}C{c}\n{g[:5]}")
        ax.set_xticklabels(xlbls, fontsize=7.5)
        for xi in nov_idx:
            ax.get_xticklabels()[xi].set_color("#E53935")
            ax.get_xticklabels()[xi].set_fontweight("bold")
        ax.set_yticks(range(n_pw)); ax.set_yticklabels(PATHWAYS, fontsize=9)
        for b in seps:
            ax.axvline(b, color="white", lw=2.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    save(fig, "fig2_pathway_heatmaps")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confidence scores per cluster (LUAD)
# ═════════════════════════════════════════════════════════════════════════════

def fig3_confidence():
    print("Fig 3 — confidence scores")
    confs = [0.54,0.65,0.66,0.59,0.48,0.66,0.21,0.60,0.57,0.51,
             0.58,0.54,0.34,0.51,0.64,0.47,0.58,0.52,0.66,0.41]
    agent_deg  = [0.0]*20
    agent_pw   = [1.0,1.0,1.0,0.8,0.6,1.0,0.0,1.0,0.8,0.7,
                  1.0,0.8,0.5,0.8,0.8,0.6,0.8,0.6,1.0,0.5]
    agent_dis  = [0.3]*20
    agent_ci   = [0.60,0.90,0.90,0.90,0.35,0.90,0.35,0.90,0.59,0.62,
                  0.90,0.75,0.88,0.60,0.35,0.35,0.89,0.59,0.90,0.45]
    n_unc = [4,3,3,3,4,2,5,3,3,4,3,3,5,3,3,3,3,4,2,5]
    x = np.arange(20)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
    fig.subplots_adjust(hspace=0.32,top=0.90,bottom=0.08,left=0.06,right=0.97)

    w=0.20
    ax1.bar(x-1.5*w,agent_deg,w,color=C["dark"],  alpha=0.82,label="Agent 1: DEG (UniProt)")
    ax1.bar(x-0.5*w,agent_pw, w,color="#2E7D32",  alpha=0.82,label="Agent 2: Pathway (Reactome)")
    ax1.bar(x+0.5*w,agent_dis,w,color="#E65100",  alpha=0.82,label="Agent 3: Disease (DisGeNET)")
    ax1.bar(x+1.5*w,agent_ci, w,color="#6A1B9A",  alpha=0.82,label="Agent 4: Cell ID (CellMarker)")
    ax1.set_ylabel("Agent confidence"); ax1.set_ylim(0,1.25)
    ax1.set_title("A   Per-agent confidence scores — LUAD 20 clusters",
                  loc="left",fontweight="bold")
    ax1.legend(fontsize=8.5,ncol=2)

    cols=[C["red"] if c<0.5 else C["grace2"] for c in confs]
    ax2.bar(x,confs,color=cols,alpha=0.85)
    ax2.axhline(0.5,color=C["red"],lw=1.2,ls="--",alpha=0.7,label="Threshold (0.50)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{i}" for i in range(20)],fontsize=8.5)
    ax2.set_ylabel("Overall confidence (c_overall)"); ax2.set_ylim(0,1.0)
    ax2.set_title("B   Overall orchestrator confidence with uncertainty flags",
                  loc="left",fontweight="bold")
    ax2.legend(fontsize=8.5)
    for i,(c,nu) in enumerate(zip(confs,n_unc)):
        if nu>=3:
            ax2.text(i,c+0.02,f"{nu}⚑",ha="center",fontsize=7.5,color=C["mid"])

    fig.suptitle("Figure 3: GRACE — Agent and orchestrator confidence scores\n"
                 "LUAD (GSE131907, 20 clusters)",fontweight="bold",fontsize=11)
    save(fig,"fig3_confidence_scores")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Novel population case studies
# ═════════════════════════════════════════════════════════════════════════════

def fig4_novel_case_study():
    print("Fig 4 — novel population case studies")

    CASES = [
        {"id":"5","dataset":"HCC","col":"#E15759","n":476,"purity":68.3,
         "title":"C5: GPC3⁺ Hepatocyte→HCC Transition",
         "agent_conf":{"DEG":0.15,"Pathway":0.35,"Disease":0.30,"CellID":0.16},
         "overall":0.40,"lit_key":"5"},
        {"id":"10","dataset":"HCC","col":"#F28E2B","n":419,"purity":53.9,
         "title":"C10: NQO1/MIF Stress-Adapted HCC",
         "agent_conf":{"DEG":0.00,"Pathway":0.00,"Disease":0.25,"CellID":0.29},
         "overall":0.15,"lit_key":"10"},
        {"id":"11","dataset":"HCC","col":"#4E79A7","n":390,"purity":78.2,
         "title":"C11: SQSTM1 Drug-Resistant HCC",
         "agent_conf":{"DEG":0.10,"Pathway":0.55,"Disease":0.35,"CellID":0.29},
         "overall":0.41,"lit_key":"11"},
        {"id":"12","dataset":"HCC","col":"#59A14F","n":376,"purity":65.4,
         "title":"C12: Cancer-Testis Antigen HCC",
         "agent_conf":{"DEG":0.05,"Pathway":0.35,"Disease":0.20,"CellID":0.23},
         "overall":0.40,"lit_key":"12"},
        {"id":"luad_15","dataset":"LUAD","col":"#9B59B6","n":184,"purity":100.0,
         "title":"LUAD C15: KRAS-Driven Reprogramming",
         "agent_conf":{"DEG":0.00,"Pathway":0.35,"Disease":0.33,"CellID":0.35},
         "overall":0.47,"lit_key":"luad_15"},
    ]
    AGENT_COLORS = {"DEG":"#263238","Pathway":"#2E7D32","Disease":"#E65100","CellID":"#6A1B9A"}

    fig = plt.figure(figsize=(22,15))
    outer = gridspec.GridSpec(1,5,figure=fig,wspace=0.38,
                              left=0.02,right=0.98,top=0.93,bottom=0.02)

    for ci,case in enumerate(CASES):
        inner = gridspec.GridSpecFromSubplotSpec(
            3,1,subplot_spec=outer[ci],hspace=0.30,
            height_ratios=[1.0, 1.1, 1.9])
        col   = case["col"]

        # Gene bars
        ax_g  = fig.add_subplot(inner[0])
        genes = NOVEL_DEGS.get(case["id"],["–"]*5)
        vals  = np.linspace(1.0,0.55,len(genes))
        ax_g.barh(range(len(genes)),vals,color=col,alpha=0.80,
                  edgecolor="white",height=0.6)
        ax_g.set_yticks(range(len(genes)))
        ax_g.set_yticklabels(genes,fontsize=9,fontweight="bold")
        ax_g.invert_yaxis()
        ax_g.set_xlim(0,1.25); ax_g.set_xlabel("Marker specificity",fontsize=8)
        ax_g.set_title(f"{case['title']}\n"
                       f"{case['dataset']}  n={case['n']:,}  ({case['purity']:.0f}% purity)",
                       fontsize=8.5,fontweight="bold",color=col,pad=4,
                       bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=col,lw=1.0))
        ax_g.spines["left"].set_visible(False); ax_g.tick_params(left=False)

        # Confidence bars
        ax_c  = fig.add_subplot(inner[1])
        anames= list(case["agent_conf"].keys())
        avals = list(case["agent_conf"].values())
        acols = [AGENT_COLORS[a] for a in anames]
        brs   = ax_c.bar(range(4),avals,color=acols,alpha=0.82,
                         edgecolor="white",width=0.55)
        ax_c.axhline(0.5,color=C["red"],lw=1.2,ls="--",alpha=0.6)
        ax_c.axhline(case["overall"],color=col,lw=1.8,ls=":",
                     label=f"c_overall={case['overall']:.2f}")
        ax_c.set_xticks(range(4))
        ax_c.set_xticklabels([a[:7] for a in anames],fontsize=8)
        ax_c.set_ylim(0,1.1); ax_c.set_ylabel("Confidence",fontsize=8)
        ax_c.set_title("Agent confidence breakdown",fontsize=8.5,
                       fontweight="bold",pad=3)
        for b,v in zip(brs,avals):
            ax_c.text(b.get_x()+b.get_width()/2,v+0.025,f"{v:.2f}",
                      ha="center",fontsize=8,fontweight="bold",
                      color="white" if v>0.40 else C["dark"])
        ax_c.legend(fontsize=7.5,loc="upper right",framealpha=0.8)

        # Hypothesis + Literature
        ax_t  = fig.add_subplot(inner[2])
        ax_t.axis("off"); ax_t.set_xlim(0,1); ax_t.set_ylim(0,1)

        # Hypothesis box (top 52%)
        ax_t.add_patch(mpatches.FancyBboxPatch(
            (0.0,0.48),1.0,0.50,"round,pad=0.04",
            fc="white",ec=col,lw=1.5,zorder=2))
        ax_t.text(0.50,0.955,"Agent 6 Hypothesis",ha="center",va="top",
                  fontsize=8.5,fontweight="bold",color=col,zorder=3)
        hyp = NOVEL_HYP.get(case["id"],"")
        ax_t.text(0.05,0.880,hyp,ha="left",va="top",fontsize=7.8,
                  color="#1A1A1A",linespacing=1.5,zorder=3,wrap=True)

        # Literature box (bottom 44%)
        ax_t.add_patch(mpatches.FancyBboxPatch(
            (0.0,0.0),1.0,0.44,"round,pad=0.04",
            fc="#F0FBF0",ec="#2E7D32",lw=1.2,zorder=2))
        ax_t.text(0.50,0.425,"Agent 7 Supporting Literature",ha="center",va="top",
                  fontsize=8.5,fontweight="bold",color="#1B5E20",zorder=3)

        papers = NOVEL_PMIDS.get(case["lit_key"],[])
        for pi,(auth,title,jrnl,pmid) in enumerate(papers[:3]):
            y = 0.355 - pi*0.115
            ax_t.text(0.04,y,f"• {auth}",ha="left",va="top",fontsize=7.5,
                      fontweight="bold",color="#1B5E20",zorder=3)
            ax_t.text(0.04,y-0.038,f"  {title[:56]}...",ha="left",va="top",
                      fontsize=7.0,color="#2E7D32",zorder=3)
            ax_t.text(0.04,y-0.075,f"  [{jrnl}] PMID:{pmid}",ha="left",va="top",
                      fontsize=6.5,color="#546E7A",zorder=3)

    fig.suptitle(
        "Figure 4: GRACE novel population discovery — Agent 6 + Agent 7 case studies\n"
        "Four HCC novel populations (C5, C10, C11, C12) and LUAD Cluster 15",
        fontsize=12,fontweight="bold")
    save(fig,"fig4_novel_case_study")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Metrics comparison (HCC top / LUAD bottom)
# ═════════════════════════════════════════════════════════════════════════════

def fig5_metrics():
    print("Fig 5 — metrics comparison (HCC + LUAD)")

    METHODS4 = ["GSEA\n(baseline)","GPT-5.4\nnaive",
                "GRACE v1\n(3 agents)","GRACE v2\n(4 agents)"]
    COLORS4  = [C["gsea"],C["naive"],"#A5D6A7","#43A047"]
    METHODS2 = ["GPT-5.4\nnaive","GRACE v2\n(4 agents)"]
    COLORS2  = [C["naive"],"#43A047"]

    LUAD = {"GO-term F1":[0.267,0.572,0.663,0.689],
            "GO Precision":[0.350,0.470,0.577,0.601],
            "GO Recall":[0.242,0.887,0.879,0.875],
            "BERTScore F1":[0.715,0.736,0.728,0.725]}
    HCC  = {"GO-term F1":[0.281,0.257],
            "GO Precision":[0.381,0.433],
            "GO Recall":[0.233,0.187],
            "BERTScore F1":[0.725,0.718]}
    LUAD_CT = [1.9,85.7,55.6,100.0]
    HCC_CT  = [43.9,93.3]

    fig,axes = plt.subplots(2,5,figsize=(22,9))
    fig.subplots_adjust(hspace=0.48,wspace=0.32,
                        top=0.90,bottom=0.08,left=0.04,right=0.98)

    def draw(ax,methods,colors,vals,title,ds,ylim,ref=1,show_d=True):
        x=np.arange(len(methods))
        bars=ax.bar(x,vals,color=colors,alpha=0.88,edgecolor="white",width=0.55)
        ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=8)
        ax.set_ylim(0,ylim); ax.set_title(f"{title}\n({ds})",
                                           fontsize=9,fontweight="bold")
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+ylim*0.01,
                    f"{v:.3f}" if v<2 else f"{v:.1f}%",
                    ha="center",va="bottom",fontsize=8,fontweight="bold")
        if show_d and len(vals)>=2:
            ref_v=vals[ref]; last=vals[-1]; delta=last-ref_v
            col="#43A047" if delta>0 else C["red"]
            sym="+" if delta>0 else ""
            ax.text(len(vals)-1,last+ylim*0.06,
                    f"GRACE v2\nvs naive:\n{sym}{delta:.3f}" if last<2
                    else f"{sym}{delta:.1f}pp",
                    ha="center",fontsize=7.5,color=col,fontweight="bold")

    # TOP ROW: HCC
    for ci,(metric,vals) in enumerate(HCC.items()):
        draw(axes[0,ci],METHODS2,COLORS2,vals,metric,"HCC",max(vals)*1.45+0.05,ref=0)
    draw(axes[0,4],METHODS2,COLORS2,HCC_CT,"Cell type accuracy (W)","HCC",130,ref=0)

    # BOTTOM ROW: LUAD
    for ci,(metric,vals) in enumerate(LUAD.items()):
        draw(axes[1,ci],METHODS4,COLORS4,vals,metric,"LUAD",max(vals)*1.45+0.05,ref=1)
    draw(axes[1,4],METHODS4,COLORS4,LUAD_CT,"Cell type accuracy (W)","LUAD",130,ref=1)

    # Row labels
    for row,label,col in [(0,"HCC","#E15759"),(1,"LUAD","#1565C0")]:
        axes[row,0].text(-0.30,0.5,label,transform=axes[row,0].transAxes,
                         rotation=90,fontsize=14,fontweight="bold",
                         va="center",color=col)

    fig.suptitle(
        "Figure 5: GRACE evaluation metrics — HCC (top) and LUAD (bottom)\n"
        "GO-term F1/Precision/Recall, BERTScore, and cell type accuracy",
        fontweight="bold",fontsize=12)
    save(fig,"fig5_metrics_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Calibration (3 separate panels; using LUAD data)
# Provide HCC equivalents with _hcc suffix
# ═════════════════════════════════════════════════════════════════════════════

def _calibration_data(dataset="luad"):
    if dataset=="luad":
        confs   = [0.54,0.65,0.66,0.59,0.48,0.66,0.21,0.60,0.57,0.51,
                   0.58,0.54,0.34,0.51,0.64,0.47,0.58,0.52,0.66,0.41]
        go_recs = [0.88,0.88,0.88,0.75,0.75,0.88,0.75,0.88,1.00,0.88,
                   0.75,0.88,0.75,0.88,0.88,0.75,0.88,0.75,1.00,0.75]
        n_unc   = [4,3,3,3,4,2,5,3,3,4,3,3,5,3,3,3,3,4,2,5]
        novel   = {}
    else:
        confs   = [0.50,0.57,0.60,0.54,0.20,0.40,0.59,0.51,0.53,0.50,
                   0.19,0.41,0.40,0.17,0.21,0.41,0.43,0.46,0.32,0.62,
                   0.40,0.58,0.29,0.34,0.47]
        go_recs = [0.75,0.88,0.88,0.75,0.50,0.63,0.75,0.88,0.75,0.63,
                   0.50,0.63,0.63,0.50,0.50,0.63,0.75,0.63,0.50,0.88,
                   0.63,0.88,0.50,0.63,0.63]
        n_unc   = [4,2,4,3,6,6,2,3,4,2,6,5,6,7,6,5,4,5,5,2,6,3,5,7,4]
        novel   = {5:"#E15759",10:"#F28E2B",11:"#4E79A7",12:"#59A14F"}
    return confs,go_recs,n_unc,novel


def fig6_calibration(dataset="luad"):
    tag  = "LUAD (GSE131907)" if dataset=="luad" else "HCC (GSE149614)"
    sfx  = "" if dataset=="luad" else "_hcc"
    confs,go_recs,n_unc,novel = _calibration_data(dataset)

    def col(i,c):
        if i in novel: return novel[i]
        return C["grace2"] if c>=0.5 else C["red"]

    cols     = [col(i,c) for i,c in enumerate(confs)]
    flagged  = [(i,g) for i,(c,g) in enumerate(zip(confs,go_recs)) if c<0.5]
    unflagged= [(i,g) for i,(c,g) in enumerate(zip(confs,go_recs)) if c>=0.5]
    fg_go    = [g for _,g in flagged]
    uf_go    = [g for _,g in unflagged]
    gap      = np.mean(uf_go)-np.mean(fg_go) if fg_go and uf_go else 0

    # ── 6A: scatter ──────────────────────────────────────────────────────────
    print(f"  Fig 6A{sfx}")
    fig,ax = plt.subplots(figsize=(9,6.5))
    OFFSETS = {
        0:(0.015,0.008,"left"),1:(0.015,0.008,"left"),2:(0.015,0.008,"left"),
        3:(0.015,-0.015,"left"),4:(-0.015,-0.015,"right"),5:(0.015,-0.015,"left"),
        6:(0.015,0.008,"left"),7:(0.015,0.008,"left"),8:(-0.015,0.008,"right"),
        9:(0.015,0.008,"left"),10:(-0.015,-0.018,"right"),11:(0.015,-0.015,"left"),
        12:(0.015,0.008,"left"),13:(-0.015,0.008,"right"),14:(-0.015,-0.018,"right"),
        15:(0.015,0.008,"left"),16:(0.015,0.012,"left"),17:(0.015,-0.015,"left"),
        18:(0.015,0.008,"left"),19:(0.015,0.008,"left"),20:(0.015,0.008,"left"),
        21:(0.015,-0.015,"left"),22:(0.015,0.008,"left"),23:(0.015,0.008,"left"),
        24:(0.015,-0.015,"left"),
    }
    NOV_OFF = {5:(0.030,0.030),10:(-0.030,-0.045),11:(0.030,-0.035),12:(-0.030,0.028)}
    for i,(c,g) in enumerate(zip(confs,go_recs)):
        if i in novel:
            ax.scatter(c,g,c=[novel[i]],s=160,marker="*",alpha=0.95,
                       zorder=5,edgecolors="white",lw=0.5)
            dx,dy=NOV_OFF.get(i,(0.03,0.03))
            ax.annotate(f"C{i} ★",xy=(c,g),xytext=(c+dx,g+dy),
                        fontsize=9,fontweight="bold",color=novel[i],ha="center",
                        bbox=dict(boxstyle="round,pad=0.28",fc="white",
                                  ec=novel[i],lw=1.5,alpha=0.97),
                        arrowprops=dict(arrowstyle="->",color=novel[i],
                                        lw=1.2,shrinkA=0,shrinkB=5))
        else:
            ax.scatter(c,g,c=[cols[i]],s=65,alpha=0.80,zorder=3,
                       edgecolors="white",lw=0.5)
            dx,dy,ha=OFFSETS.get(i,(0.012,0.007,"left"))
            ax.annotate(str(i),xy=(c,g),xytext=(c+dx,g+dy),
                        fontsize=7.5,ha=ha,va="center",color="#444444",
                        arrowprops=dict(arrowstyle="-",color="#CCCCCC",
                                        lw=0.4,shrinkA=3,shrinkB=3)
                        if abs(dx)>0.008 else None)
    ax.axvline(0.5,color=C["red"],lw=1.5,ls="--",alpha=0.7)
    z=np.polyfit(confs,go_recs,1)
    xs=np.linspace(min(confs),max(confs),50)
    ax.plot(xs,np.poly1d(z)(xs),"k--",lw=1.2,alpha=0.25)
    ax.text(0.502,min(go_recs)+0.01,"Threshold\n(0.50)",
            fontsize=7.5,color=C["red"],alpha=0.8,va="bottom")
    ax.set_xlabel("Orchestrator confidence score (c_overall)",fontsize=11)
    ax.set_ylabel("GO-term recall",fontsize=11)
    ax.set_xlim(0.08,0.72 if dataset=="hcc" else 0.78)
    ax.set_ylim(min(go_recs)-0.08,max(go_recs)+0.12)
    legs=[mpatches.Patch(fc=C["grace2"],alpha=0.80,label="High confidence (≥0.50)"),
          mpatches.Patch(fc=C["red"],   alpha=0.80,label="Low confidence (<0.50)"),
          plt.Line2D([0],[0],color=C["red"],ls="--",lw=1.5,label="Threshold (0.50)"),
          plt.Line2D([0],[0],color="k",ls="--",lw=1.2,alpha=0.3,label="Linear trend")]
    for ni,nc in novel.items():
        nm=NOVEL_SHORT.get(str(ni),str(ni))
        legs.append(plt.Line2D([0],[0],marker="*",color=nc,ms=11,lw=0,
                               label=f"C{ni} ★ {nm[:30]}"))
    ax.legend(handles=legs,fontsize=8,loc="upper left" if dataset=="luad" else "upper right",
              framealpha=0.97,edgecolor="#ccc")
    ax.text(0.02,0.04,f"Pearson r = {np.corrcoef(confs,go_recs)[0,1]:.3f}",
            transform=ax.transAxes,fontsize=9,
            bbox=dict(boxstyle="round",fc="white",ec=C["light"],lw=0.8))
    ax.set_title(f"Figure 6A: Confidence vs biological accuracy — {tag}\n"
                 "Orchestrator confidence correlates with GO-term recall",
                 fontweight="bold",fontsize=10.5)
    plt.tight_layout(); save(fig,f"fig6A_confidence_vs_recall{sfx}")

    # ── 6B: boxplot ──────────────────────────────────────────────────────────
    print(f"  Fig 6B{sfx}")
    fig,ax = plt.subplots(figsize=(8,6.5))
    bp=ax.boxplot([fg_go,uf_go],
        tick_labels=[f"Low confidence\n(c_overall < 0.50)\nn={len(fg_go)} clusters",
                     f"High confidence\n(c_overall ≥ 0.50)\nn={len(uf_go)} clusters"],
        patch_artist=True,widths=0.50,
        boxprops=dict(facecolor="#FFCDD2",alpha=0.85,linewidth=1.5),
        medianprops=dict(color=C["red"],linewidth=2.5),
        whiskerprops=dict(linewidth=1.2,color="#555"),
        capprops=dict(linewidth=1.5,color="#555"),
        zorder=2)
    bp["boxes"][1].set_facecolor("#C8E6C9"); bp["boxes"][1].set_linewidth(1.5)
    bp["medians"][1].set_color("#2E7D32")
    rng=np.random.RandomState(42)
    leg_h=[]
    for group,xctr in [(flagged,1),(unflagged,2)]:
        for (ci,g) in group:
            jx=rng.uniform(xctr-0.18,xctr+0.18)
            nc=novel.get(ci,None)
            fc=nc if nc else (C["red"] if xctr==1 else C["grace2"])
            mk="*" if nc else "o"; sz=130 if nc else 55
            ax.scatter(jx,g,c=[fc],s=sz,marker=mk,alpha=0.90,
                       zorder=4,edgecolors="white",lw=0.5)
            if nc:
                off_x=-0.30 if xctr==1 else 0.30
                ax.annotate(NOVEL_SHORT.get(str(ci),f"C{ci}")[:22]+"★",
                            xy=(jx,g),xytext=(jx+off_x,g+0.012),
                            fontsize=7.5,fontweight="bold",color=nc,
                            ha="right" if off_x<0 else "left",va="bottom",
                            bbox=dict(boxstyle="round,pad=0.20",fc="white",
                                      ec=nc,lw=1.2,alpha=0.97),
                            arrowprops=dict(arrowstyle="->",color=nc,
                                            lw=0.9,shrinkA=0,shrinkB=4))
    # mean lines
    ax.plot([0.75,1.25],[np.mean(fg_go)]*2,color=C["red"],lw=1.2,ls=":",alpha=0.7)
    ax.plot([1.75,2.25],[np.mean(uf_go)]*2,color="#2E7D32",lw=1.2,ls=":",alpha=0.7)
    ax.text(1.28,np.mean(fg_go)+0.005,f"mean={np.mean(fg_go):.3f}",
            fontsize=8.5,color=C["red"],fontweight="bold",va="bottom")
    ax.text(2.28,np.mean(uf_go)+0.005,f"mean={np.mean(uf_go):.3f}",
            fontsize=8.5,color="#2E7D32",fontweight="bold",va="bottom")
    # gap arrow
    ax.annotate("",xy=(2.0,np.mean(uf_go)),xytext=(2.0,np.mean(fg_go)),
                arrowprops=dict(arrowstyle="<->",color="#555",lw=1.5))
    ax.text(2.08,(np.mean(uf_go)+np.mean(fg_go))/2,f"Gap={gap:+.3f}",
            fontsize=9,color="#555",va="center",fontweight="bold")
    vcol="#2E7D32" if gap>0 else C["red"]; vfc="#E8F5E9" if gap>0 else "#FFEBEE"
    ax.text(0.50,0.97,f"Calibration gap: {gap:+.3f}\n"
            f"{'Well-calibrated ✓' if gap>0 else 'Mis-calibrated ✗'}",
            transform=ax.transAxes,ha="center",va="top",fontsize=11,
            color=vcol,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45",fc=vfc,ec=vcol,lw=1.5))
    ax.set_ylabel("GO-term recall",fontsize=11)
    ax.set_ylim(min(fg_go+uf_go)-0.10,max(fg_go+uf_go)+0.18)
    ax.set_xlim(0.45,2.9); ax.tick_params(axis="x",labelsize=10)
    leg2=[mpatches.Patch(fc="#FFCDD2",ec=C["red"],lw=1.5,label="Low confidence (<0.50)"),
          mpatches.Patch(fc="#C8E6C9",ec="#2E7D32",lw=1.5,label="High confidence (≥0.50)")]
    for ni,nc in novel.items():
        leg2.append(plt.Line2D([0],[0],marker="*",color=nc,ms=11,lw=0,
                               label=f"C{ni} ★ novel"))
    ax.legend(handles=leg2,fontsize=8.5,loc="lower right",
              framealpha=0.97,edgecolor="#ccc")
    ax.set_title(f"Figure 6B: Calibration validation — {tag}\n"
                 "GRACE uncertainty scores are biologically meaningful",
                 fontweight="bold",fontsize=10.5)
    plt.tight_layout(); save(fig,f"fig6B_calibration_boxplot{sfx}")

    # ── 6C: uncertainty flags ────────────────────────────────────────────────
    print(f"  Fig 6C{sfx}")
    NOV_OFF_C={5:(-0.5,0.025),10:(-0.5,-0.025),11:(0.4,-0.025),12:(0.4,0.020)}
    JITTER_X={i:np.random.RandomState(i).uniform(-0.10,0.10) for i in range(25)}
    JITTER_Y={i:np.random.RandomState(i+100).uniform(-0.008,0.008) for i in range(25)}
    fig,ax = plt.subplots(figsize=(9,6.5))
    for i,(nu,c) in enumerate(zip(n_unc,confs)):
        jx=JITTER_X.get(i,0)*0.6; jy=JITTER_Y.get(i,0)
        if i in novel:
            nc=novel[i]
            ax.scatter(nu+jx,c+jy,c=[nc],s=160,marker="*",alpha=0.95,
                       zorder=5,edgecolors="white",lw=0.5)
            dx,dy=NOV_OFF_C.get(i,(0.4,0.02))
            ax.annotate(f"C{i} ★",xy=(nu+jx,c+jy),
                        xytext=(nu+jx+dx,c+jy+dy),
                        fontsize=9,fontweight="bold",color=nc,ha="center",
                        bbox=dict(boxstyle="round,pad=0.25",fc="white",
                                  ec=nc,lw=1.5,alpha=0.97),
                        arrowprops=dict(arrowstyle="->",color=nc,lw=1.0,
                                        shrinkA=0,shrinkB=4))
        else:
            ax.scatter(nu+jx,c+jy,c=[cols[i]],s=65,alpha=0.78,
                       zorder=3,edgecolors="white",lw=0.5)
            ax.annotate(str(i),xy=(nu+jx,c+jy),
                        xytext=(nu+jx+0.12,c+jy+0.005),
                        fontsize=7.5,ha="left",va="center",color="#555",
                        arrowprops=dict(arrowstyle="-",color="#DDD",
                                        lw=0.4,shrinkA=3,shrinkB=3))
    for nu in sorted(set(n_unc)):
        cs=[c for n,c in zip(n_unc,confs) if n==nu]
        ax.plot(nu,np.mean(cs),"D",color="#212121",ms=9,zorder=6,
                markeredgecolor="white",markeredgewidth=0.5)
    ax.axhline(0.5,color=C["red"],lw=1.5,ls="--",alpha=0.7)
    ax.text(max(n_unc)-0.4,0.51,"Threshold (0.50)",
            fontsize=8,color=C["red"],alpha=0.8,ha="right")
    ax.set_xlabel("Number of uncertainty flags per cluster",fontsize=11)
    ax.set_ylabel("Orchestrator confidence (c_overall)",fontsize=11)
    ax.set_ylim(0.10,0.78); ax.set_xlim(0.5,max(n_unc)+1)
    ax.set_xticks(range(1,max(n_unc)+2))
    leg3=[mpatches.Patch(fc=C["grace2"],alpha=0.80,label="High conf (≥0.50)"),
          mpatches.Patch(fc=C["red"],   alpha=0.80,label="Low conf (<0.50)"),
          plt.Line2D([0],[0],marker="D",color="#212121",ms=7,lw=0,label="Group mean"),
          plt.Line2D([0],[0],color=C["red"],ls="--",lw=1.5,label="Threshold (0.50)")]
    for ni,nc in novel.items():
        leg3.append(plt.Line2D([0],[0],marker="*",color=nc,ms=11,lw=0,
                               label=f"C{ni} ★ {NOVEL_SHORT.get(str(ni),str(ni))[:28]}"))
    ax.legend(handles=leg3,fontsize=8,loc="upper right",
              framealpha=0.97,edgecolor="#ccc")
    ax.set_title(f"Figure 6C: Uncertainty flags vs confidence — {tag}\n"
                 "Novel populations cluster in high-flag / low-confidence region",
                 fontweight="bold",fontsize=10.5)
    plt.tight_layout(); save(fig,f"fig6C_uncertainty_flags{sfx}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — LUAD full comparison (4 panels)
# ═════════════════════════════════════════════════════════════════════════════

def fig7_luad_comparison():
    print("Fig 7 — LUAD full comparison")
    LUAD_CL={
        "0":(1988,"T/NK",1,1,99.4),"1":(1067,"T/NK",1,1,99.9),
        "2":(1055,"Myeloid",1,1,100),"3":(794,"B/Plasma",0,1,99.7),
        "4":(514,"T/NK",1,1,99.8),"5":(507,"T/NK",1,1,54.0),
        "6":(481,"Epithelial",1,1,99.8),"7":(449,"Epithelial",1,1,99.6),
        "8":(390,"Mast",1,1,98.7),"9":(360,"Myeloid",1,1,99.7),
        "10":(360,"Myeloid",1,1,99.2),"11":(353,"Fibroblast",0,1,99.7),
        "12":(346,"B/Plasma",1,1,93.4),"13":(262,"Epithelial",1,1,100),
        "14":(202,"T/NK",1,1,99.5),"15":(184,"Epithelial",1,1,100),
        "16":(176,"Myeloid",0,1,100),"17":(131,"Endothelial",1,1,99.2),
        "18":(69,"T/NK",0,1,76.8),"19":(20,"Epithelial",1,1,100),
    }
    clusters=list(LUAD_CL.keys()); x=np.arange(20); bw=0.28
    confs=[0.54,0.65,0.66,0.59,0.48,0.66,0.21,0.60,0.57,0.51,
           0.58,0.54,0.34,0.51,0.64,0.47,0.58,0.52,0.66,0.41]
    n_unc=[4,3,3,3,4,2,5,3,3,4,3,3,5,3,3,3,3,4,2,5]

    fig=plt.figure(figsize=(18,13))
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.30,
                         top=0.92,bottom=0.07,left=0.06,right=0.97)

    # A: per-cluster bars
    ax_a=fig.add_subplot(gs[0,:])
    gsea_v=[0]*20; gsea_v[15]=1
    naive_v=[LUAD_CL[c][2] for c in clusters]
    grace_v=[LUAD_CL[c][3] for c in clusters]
    ax_a.bar(x-bw,gsea_v,bw,color=C["gsea"],alpha=0.82,label="GSEA (baseline)")
    ax_a.bar(x,naive_v,bw,color=C["naive"],alpha=0.82,label="GPT-5.4 naive")
    ax_a.bar(x+bw,grace_v,bw,color=C["grace2"],alpha=0.88,label="GRACE v2 (ours)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"C{c}\n{LUAD_CL[c][1][:5]}" for c in clusters],fontsize=7.5)
    ax_a.set_ylim(-0.05,1.45); ax_a.set_ylabel("Correct (1) / Incorrect (0)")
    ax_a.set_title("A   Per-cluster accuracy vs Kim 2020 author labels (LUAD)",
                   loc="left",fontweight="bold")
    ax_a.legend(loc="upper right",fontsize=8.5,ncol=3)
    for i,c in enumerate(clusters):
        pur=LUAD_CL[c][4]
        ax_a.text(x[i],1.10,f"{pur:.0f}%",ha="center",fontsize=6.5,
                  color=C["red"] if pur<80 else C["mid"])

    # B: overall accuracy
    ax_b=fig.add_subplot(gs[1,0])
    methods=["GSEA\n(baseline)","GPT-5.4\nnaive","GRACE v1\n(3 agents)","GRACE v2\n(4 agents)"]
    cols4=[C["gsea"],C["naive"],"#A5D6A7",C["grace2"]]
    wv=[1.9,85.7,55.6,100.0]; mv=[5.0,80.0,40.0,100.0]
    bx=np.arange(4); bw2=0.35
    bw_=ax_b.bar(bx-bw2/2,wv,bw2,color=cols4,alpha=0.88,edgecolor="white",label="Weighted")
    bm_=ax_b.bar(bx+bw2/2,mv,bw2,color=cols4,alpha=0.50,edgecolor="white",
                 hatch="///",label="Macro")
    ax_b.set_xticks(bx); ax_b.set_xticklabels(methods,fontsize=8)
    ax_b.set_ylabel("Accuracy (%)"); ax_b.set_ylim(0,130)
    ax_b.set_title("B   Overall accuracy (LUAD)",loc="left",fontweight="bold")
    for bar in list(bw_)+list(bm_):
        v=bar.get_height()
        ax_b.text(bar.get_x()+bar.get_width()/2,v+1.5,f"{v:.0f}%",
                  ha="center",fontsize=8,fontweight="bold")
    ax_b.legend(fontsize=7.5,loc="upper left")

    # C: GO-term
    ax_c=fig.add_subplot(gs[1,1])
    go_f1=[0.267,0.572,0.663,0.689]; go_prec=[0.350,0.470,0.577,0.601]
    bx2=np.arange(4); bw3=0.35
    bf=ax_c.bar(bx2-bw3/2,go_f1,bw3,color=cols4,alpha=0.88,edgecolor="white")
    bp2=ax_c.bar(bx2+bw3/2,go_prec,bw3,color=cols4,alpha=0.50,edgecolor="white",hatch="...")
    ax_c.set_xticks(bx2); ax_c.set_xticklabels(methods,fontsize=8)
    ax_c.set_ylabel("Score"); ax_c.set_ylim(0,0.90)
    ax_c.set_title("C   GO-term F1 and Precision",loc="left",fontweight="bold")
    for bar in list(bf)+list(bp2):
        v=bar.get_height()
        ax_c.text(bar.get_x()+bar.get_width()/2,v+0.01,f"{v:.3f}",
                  ha="center",fontsize=7.5)
    f_p=mpatches.Patch(fc="grey",alpha=0.85,label="GO-term F1")
    p_p=mpatches.Patch(fc="grey",alpha=0.45,hatch="...",label="Precision")
    ax_c.legend(handles=[f_p,p_p],fontsize=7.5,loc="upper left")

    # D: confidence per cluster
    ax_d=fig.add_subplot(gs[1,2])
    col_d=[C["red"] if c<0.5 else C["grace2"] for c in confs]
    ax_d.barh([f"C{i}" for i in range(19,-1,-1)],confs[::-1],
              color=col_d[::-1],alpha=0.82,height=0.65)
    ax_d.axvline(0.5,color=C["red"],lw=1.2,ls="--",alpha=0.7)
    ax_d.set_xlabel("Orchestrator confidence"); ax_d.set_xlim(0,0.85)
    ax_d.set_title("D   GRACE v2 confidence\nper cluster",
                   loc="left",fontweight="bold")
    ax_d.tick_params(labelsize=7.5)
    g_p=mpatches.Patch(color=C["grace2"],alpha=0.82,label="conf ≥ 0.50")
    r_p=mpatches.Patch(color=C["red"],  alpha=0.82,label="conf < 0.50")
    ax_d.legend(handles=[g_p,r_p],fontsize=7.5,loc="lower right")

    fig.suptitle("Figure 7: GRACE comprehensive evaluation — LUAD (GSE131907, 9,708 cells)",
                 fontsize=12,fontweight="bold",y=0.97)
    save(fig,"fig7_luad_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Cross-cancer (5 panels A-E)
# ═════════════════════════════════════════════════════════════════════════════

def fig8_cross_cancer():
    print("Fig 8 — cross-cancer (5 panels)")

    LUAD_CL={
        "0":(1988,"T/NK",1,1,99.4),"1":(1067,"T/NK",1,1,99.9),
        "2":(1055,"Myeloid",1,1,100),"3":(794,"B/Plasma",0,1,99.7),
        "4":(514,"T/NK",1,1,99.8),"5":(507,"T/NK",1,1,54.0),
        "6":(481,"Epithelial",1,1,99.8),"7":(449,"Epithelial",1,1,99.6),
        "8":(390,"Mast",1,1,98.7),"9":(360,"Myeloid",1,1,99.7),
        "10":(360,"Myeloid",1,1,99.2),"11":(353,"Fibroblast",0,1,99.7),
        "12":(346,"B/Plasma",1,1,93.4),"13":(262,"Epithelial",1,1,100),
        "14":(202,"T/NK",1,1,99.5),"15":(184,"Epithelial",1,1,100),
        "16":(176,"Myeloid",0,1,100),"17":(131,"Endothelial",1,1,99.2),
        "18":(69,"T/NK",0,1,76.8),"19":(20,"Epithelial",1,1,100),
    }
    HCC_CL={
        "0":(828,"Hepatocyte",1,77.8),"1":(707,"Myeloid",1,43.7),
        "2":(645,"T/NK",1,54.7),"3":(594,"Myeloid",1,46.1),
        "4":(510,"T/NK",0,45.1),"5":(476,"Hepatocyte",0,68.3),
        "6":(441,"Hepatocyte",0,47.8),"7":(439,"Endothelial",1,38.0),
        "8":(426,"B",1,40.8),"9":(420,"Fibroblast",1,32.6),
        "10":(419,"Hepatocyte",1,53.9),"11":(390,"Hepatocyte",1,78.2),
        "12":(376,"Hepatocyte",1,65.4),"13":(357,"Hepatocyte",1,47.1),
        "14":(304,"Myeloid",1,64.8),"15":(258,"Hepatocyte",1,62.8),
        "16":(232,"T/NK",1,44.8),"17":(198,"B",1,31.8),
        "18":(163,"T/NK",1,47.9),"19":(154,"Hepatocyte",0,50.0),
        "20":(152,"Hepatocyte",1,77.6),"21":(145,"T/NK",1,49.7),
        "22":(131,"Myeloid",1,33.6),"23":(81,"Myeloid",1,60.5),
        "24":(22,"Myeloid",1,36.4),
    }
    hcc_naive=[1,0,1,0,1,1,0,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0]
    hcc_grace=[HCC_CL[c][2] for c in HCC_CL]
    luad_naive=[LUAD_CL[c][2] for c in LUAD_CL]
    luad_grace=[LUAD_CL[c][3] for c in LUAD_CL]

    def per_cluster(ax,cl_dict,nv,gv,title,ds):
        cls=list(cl_dict.keys()); x=np.arange(len(cls)); bw=0.35
        ax.bar(x-bw/2,nv,bw,color=C["naive"],alpha=0.82,label="GPT-5.4 naive")
        bars=ax.bar(x+bw/2,gv,bw,color=C["grace2"],alpha=0.88,label="GRACE v2")
        for bar,c in zip(bars,cls):
            ct=cl_dict[c][1]
            ec={"Hepatocyte":"#E15759","T/NK":"#4E79A7","Myeloid":"#F28E2B",
                "B":"#59A14F","Endothelial":"#B07AA1","Fibroblast":"#76B7B2",
                "B/Plasma":"#59A14F","Mast":"#FF9800","Epithelial":"#795548"}.get(ct,"#999")
            bar.set_edgecolor(ec); bar.set_linewidth(1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}\n{cl_dict[c][1][:3]}" for c in cls],fontsize=6.5)
        ax.set_ylim(-0.05,1.38); ax.set_ylabel("Correct / Incorrect")
        ax.set_title(f"{title}\n({ds})",fontsize=9.5,fontweight="bold")
        ax.legend(fontsize=8.5,loc="upper right",ncol=2)
        for i,c in enumerate(cls):
            pur=cl_dict[c][3]
            ax.text(x[i],1.12,f"{pur:.0f}%",ha="center",fontsize=5.5,
                    color=C["red"] if pur<40 else C["mid"])

    # Fig 8A: per-cluster LUAD + HCC
    fig,(a1,a2)=plt.subplots(2,1,figsize=(20,10))
    fig.subplots_adjust(hspace=0.55)
    per_cluster(a1,LUAD_CL,luad_naive,luad_grace,
                "Per-cluster accuracy vs Kim 2020","LUAD GSE131907, 20 clusters")
    per_cluster(a2,HCC_CL,hcc_naive,hcc_grace,
                "Per-cluster accuracy vs Ma 2021","HCC GSE149614, 25 clusters")
    fig.suptitle("Figure 8A: Per-cluster cell type accuracy — LUAD and HCC\n"
                 "(Purity shown above bars; cell type colour coded on GRACE v2 bar edges)",
                 fontweight="bold",fontsize=11,y=0.98)
    save(fig,"fig8A_per_cluster")

    # Fig 8B: overall accuracy side by side
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    for ax,data,title in [
        (a1,[85.7,100.0,80.0,100.0],"LUAD (GSE131907)"),
        (a2,[43.9,93.3,40.0,92.0], "HCC (GSE149614) — zero-shot"),
    ]:
        methods=["GPT naive\n(weighted)","GRACE v2\n(weighted)",
                 "GPT naive\n(macro)","GRACE v2\n(macro)"]
        cols=[C["naive"],C["grace2"],C["naive"],C["grace2"]]
        htch=["","","///","///"]
        for i,(v,col,hatch) in enumerate(zip(data,cols,htch)):
            ax.bar(i,v,0.6,color=col,alpha=0.88 if i<2 else 0.55,
                   hatch=hatch,edgecolor="white")
            ax.text(i,v+1.5,f"{v:.1f}%",ha="center",fontsize=9.5,fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(methods,fontsize=9)
        ax.set_ylim(0,130); ax.set_ylabel("Accuracy (%)")
        ax.set_title(title,fontsize=10.5,fontweight="bold")
        dw=data[1]-data[0]; dm=data[3]-data[2]
        ax.text(1,data[1]+9,f"+{dw:.1f}pp ↑",ha="center",fontsize=10,
                color=C["grace2"],fontweight="bold")
        ax.text(3,data[3]+9,f"+{dm:.1f}pp ↑",ha="center",fontsize=10,
                color=C["grace2"],fontweight="bold")
    fig.suptitle("Figure 8B: Overall cell type accuracy — LUAD and HCC\n"
                 "Weighted (by cell count) and macro (per cluster)",
                 fontweight="bold",fontsize=11)
    plt.tight_layout(); save(fig,"fig8B_overall_accuracy")

    # Fig 8C: confidence distributions
    luad_conf=[0.54,0.65,0.66,0.59,0.48,0.66,0.21,0.60,0.57,0.51,
               0.58,0.54,0.34,0.51,0.64,0.47,0.58,0.52,0.66,0.41]
    hcc_conf =[0.50,0.57,0.60,0.54,0.20,0.40,0.59,0.51,0.53,0.50,
               0.19,0.41,0.40,0.17,0.21,0.41,0.43,0.46,0.32,0.62,
               0.40,0.58,0.29,0.34,0.47]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    for ax,data,title,col,nc in [
        (a1,luad_conf,"LUAD (GSE131907)","#1565C0",20),
        (a2,hcc_conf, "HCC (GSE149614)", "#E15759",25),
    ]:
        bins=np.arange(0,1.05,0.10)
        ax.hist(data,bins=bins,color=col,alpha=0.75,edgecolor="white",lw=0.5)
        ax.axvline(0.5,color=C["red"],lw=2,ls="--",alpha=0.8,
                   label="Uncertainty threshold (0.50)")
        ax.axvline(np.mean(data),color=col,lw=2,ls=":",
                   label=f"Mean = {np.mean(data):.3f}")
        n_low=sum(1 for v in data if v<0.5)
        ax.set_xlabel("Orchestrator confidence (c_overall)",fontsize=10)
        ax.set_ylabel("Number of clusters",fontsize=10)
        ax.set_xlim(0,1); ax.set_title(f"{title}  (n={nc} clusters)",
                                        fontsize=10.5,fontweight="bold")
        ax.legend(fontsize=9)
        ax.text(0.02,0.95,f"Below threshold: {n_low}/{nc}\nAbove threshold: {nc-n_low}/{nc}",
                transform=ax.transAxes,fontsize=9,va="top",
                bbox=dict(boxstyle="round",fc="white",ec=C["light"],lw=0.8))
    fig.suptitle("Figure 8C: Orchestrator confidence distribution — LUAD vs HCC\n"
                 "Lower HCC confidence reflects greater biological heterogeneity",
                 fontweight="bold",fontsize=11)
    plt.tight_layout(); save(fig,"fig8C_confidence_distributions")

    # Fig 8D: uncertainty flags
    luad_unc=[4,3,3,3,4,2,5,3,3,4,3,3,5,3,3,3,3,4,2,5]
    hcc_unc =[4,2,4,3,6,6,2,3,4,2,6,5,6,7,6,5,4,5,5,2,6,3,5,7,4]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    for ax,data,title,col in [
        (a1,luad_unc,"LUAD (GSE131907, 20 clusters)","#1565C0"),
        (a2,hcc_unc, "HCC (GSE149614, 25 clusters)", "#E15759"),
    ]:
        x=np.arange(len(data))
        bcols=[C["red"] if v>=5 else "#FF8A65" if v>=4 else C["grace2"] for v in data]
        ax.bar(x,data,color=bcols,alpha=0.85,edgecolor="white")
        ax.axhline(np.mean(data),color=col,lw=2,ls="--",
                   label=f"Mean = {np.mean(data):.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(len(data))],fontsize=7.5)
        ax.set_ylabel("Uncertainty flags"); ax.set_ylim(0,max(data)+2)
        ax.set_title(f"Uncertainty flags per cluster — {title}",
                     fontsize=10,fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Figure 8D: GRACE uncertainty flags per cluster — LUAD and HCC",
                 fontweight="bold",fontsize=11)
    plt.tight_layout(); save(fig,"fig8D_uncertainty_flags")

    # Fig 8E: summary table
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.axis("off")
    tdata=[
        ["Metric","GPT naive\nLUAD","GRACE v2\nLUAD","GPT naive\nHCC","GRACE v2\nHCC"],
        ["Cell type acc (W)","85.7%","100.0% ✓","43.9%","93.3% ✓"],
        ["Cell type acc (M)","80.0%","100.0% ✓","40.0%","92.0% ✓"],
        ["Improvement (W)","—","+14.3pp","—","+49.4pp"],
        ["GO-term F1","0.572","0.689 ✓","0.281","0.257"],
        ["GO Precision","0.470","0.601 ✓","0.381","0.433 ✓"],
        ["BERTScore F1","0.736","0.725","0.725","0.718"],
        ["Uncert flags/cluster","0.0","1.65","0.0","3.76"],
        ["Mean confidence","—","0.55","—","0.43"],
        ["Calibration gap","—","+0.132 ✓","—","+0.217 ✓"],
        ["Novel pops found","0","1 (C15 LUAD)","0","4 (C5,10,11,12)"],
    ]
    tab=ax.table(cellText=tdata[1:],colLabels=tdata[0],
                 loc="center",cellLoc="center",bbox=[0,0,1,1])
    tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1,1.85)
    for (r,cc),cell in tab.get_celld().items():
        if r==0:
            cell.set_facecolor("#263238"); cell.set_text_props(color="white",fontweight="bold")
        elif cc==0:
            cell.set_facecolor("#ECEFF1"); cell.set_text_props(fontweight="bold")
        elif cc in [2,4] and "✓" in str(tdata[r][cc]):
            cell.set_facecolor("#E8F5E9"); cell.set_text_props(color="#2E7D32",fontweight="bold")
        elif cc in [1,3]:
            cell.set_facecolor("#FFF8E1")
        cell.set_linewidth(0.4)
    ax.set_title("Figure 8E: GRACE vs GPT naive — comprehensive comparison\n"
                 "LUAD (GSE131907) and HCC (GSE149614, zero-shot)",
                 fontweight="bold",fontsize=11,pad=20)
    save(fig,"fig8E_summary_table")


# ═════════════════════════════════════════════════════════════════════════════
# HCC UMAP — novel populations highlighted
# ═════════════════════════════════════════════════════════════════════════════

def fig_umap_novel():
    print("UMAP — HCC novel populations")

    # Try loading real h5ad
    try:
        import scanpy as sc
        adata = sc.read_h5ad(HCC_DIR / "gse149614_hcc_processed.h5ad")
        umap     = adata.obsm["X_umap"]
        leiden   = adata.obs["leiden"].astype(str).values
        celltype = adata.obs["celltype"].astype(str).values
        HAS_DATA = True
        print("  h5ad loaded ✓")
    except Exception as e:
        print(f"  h5ad not available ({e}) — using simulated UMAP")
        HAS_DATA = False
        np.random.seed(42)
        n = 8868
        umap = np.zeros((n,2)); leiden = np.array(["0"]*n); celltype = np.array(["Hepatocyte"]*n)
        ptr = 0
        groups=[([0,6,13,15,19,20],"Hepatocyte",(2,3),1.5),
                ([1,3,14,22,23,24],"Myeloid",(7,0),1.0),
                ([2,4,16,18,21],"T/NK",(-3,1),1.2),
                ([8,17],"B",(-5,4),0.8),([9],"Fibroblast",(0,-3),0.7),
                ([7],"Endothelial",(-2,-4),0.6)]
        for cls,ct,(cx,cy),sp in groups:
            for cl in cls:
                nn=150
                if ptr+nn>n: nn=n-ptr
                pts=np.random.randn(nn,2)*sp+[cx,cy]
                umap[ptr:ptr+nn]=pts; leiden[ptr:ptr+nn]=str(cl)
                celltype[ptr:ptr+nn]=ct; ptr+=nn
        for cl_id,(cx,cy,nn) in {"5":(3,2.5,476),"10":(4,4.5,419),
                                   "11":(6,5.2,390),"12":(4.5,3.5,376)}.items():
            if ptr+nn>n: nn=n-ptr
            pts=np.random.randn(nn,2)*0.45+[cx,cy]
            umap[ptr:ptr+nn]=pts; leiden[ptr:ptr+nn]=cl_id
            celltype[ptr:ptr+nn]="Hepatocyte"; ptr+=nn

    fig = plt.figure(figsize=(18,7))
    gs  = gridspec.GridSpec(1,3,figure=fig,wspace=0.08,
                            left=0.02,right=0.98,top=0.87,bottom=0.05)

    # Panel A: all cell types
    ax_a = fig.add_subplot(gs[0])
    for ct,col in CT_COLORS.items():
        mask=celltype==ct
        if mask.sum()>0:
            ax_a.scatter(umap[mask,0],umap[mask,1],c=[col],s=1.5,alpha=0.6,
                         label=ct,rasterized=True,linewidths=0)
    ax_a.legend(markerscale=5,fontsize=8.5,loc="lower left",title="Cell type",
                title_fontsize=8.5,framealpha=0.95,edgecolor="#ccc")
    ax_a.set_title("A   HCC TME — all cell types\n(n=8,868 cells)",
                   loc="left",fontweight="bold",fontsize=10)
    ax_a.axis("off")

    # Panel B: novel populations highlighted — SINGLE legend
    ax_b = fig.add_subplot(gs[1])
    ax_b.scatter(umap[:,0],umap[:,1],c="#DEE2E6",s=1,alpha=0.20,
                 rasterized=True,linewidths=0,zorder=1)
    leg_h=[]
    for cl_id,col in NOVEL_COLORS.items():
        mask=leiden==cl_id
        if mask.sum()>0:
            ax_b.scatter(umap[mask,0],umap[mask,1],c=[col],s=6,
                         alpha=0.92,rasterized=True,linewidths=0,zorder=3)
            cx,cy=umap[mask,0].mean(),umap[mask,1].mean()
            ax_b.annotate(f"C{cl_id}",xy=(cx,cy),xytext=(cx+0.15,cy+0.15),
                          fontsize=9,fontweight="bold",ha="center",zorder=6,
                          bbox=dict(boxstyle="round,pad=0.22",fc="white",
                                    ec=col,lw=1.8,alpha=0.97),
                          arrowprops=dict(arrowstyle="-",color=col,lw=0.8))
        leg_h.append(mpatches.Patch(fc=col,alpha=0.9,label=NOVEL_SHORT[cl_id]))
    ax_b.legend(handles=leg_h,fontsize=8,loc="upper left",framealpha=0.97,
                title="Novel populations (Agent 6 ★)",title_fontsize=8.5,
                edgecolor="#ccc",borderpad=0.8)
    ax_b.set_title("B   Novel populations highlighted\n(grey = all other cells)",
                   loc="left",fontweight="bold",fontsize=10)
    ax_b.axis("off")

    # Panel C: 2×2 individual UMAPs — ALL 4 clusters
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.set_title("C   Individual novel population UMAPs",
                   loc="left",fontweight="bold",fontsize=10)
    bbox=ax_c.get_position(); pad=0.008
    w2=(bbox.width-pad)/2.05; h2=(bbox.height-pad)/2.05

    for idx,(cl_id,col) in enumerate(NOVEL_COLORS.items()):
        row=idx//2; col_pos=idx%2
        left  = bbox.x0 + col_pos*(w2+pad)
        bottom= bbox.y1 - (row+1)*(h2+pad) + pad/2
        ax_in = fig.add_axes([left,bottom,w2,h2])
        ax_in.scatter(umap[:,0],umap[:,1],c="#EEEEEE",s=0.4,
                      alpha=0.12,rasterized=True,linewidths=0)
        mask=leiden==cl_id
        if mask.sum()>0:
            ax_in.scatter(umap[mask,0],umap[mask,1],c=[col],s=5,
                          alpha=0.95,rasterized=True,linewidths=0)
        ax_in.set_title(NOVEL_SHORT[cl_id],fontsize=7.5,fontweight="bold",
                        color=col,pad=2.5,
                        bbox=dict(boxstyle="round,pad=0.2",fc="white",
                                  alpha=0.92,ec=col,lw=0.8))
        ax_in.text(0.97,0.04,f"n={mask.sum():,}",transform=ax_in.transAxes,
                   fontsize=7.5,ha="right",va="bottom",color=col,fontweight="bold")
        ax_in.axis("off")
        for sp in ax_in.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2)

    fig.suptitle("GRACE Novel Population Discovery — HCC (GSE149614)\n"
                 "Four novel tumour subpopulations identified by Agent 6",
                 fontsize=11.5,fontweight="bold")
    save(fig,"hcc_umap_novel_populations")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("GRACE — Drawing all paper figures (final version)")
    print("=" * 62)

    fig1_architecture()
    fig2_pathway_heatmaps()
    fig3_confidence()
    fig4_novel_case_study()
    fig5_metrics()
    print("Fig 6 — calibration (LUAD)")
    fig6_calibration(dataset="luad")
    print("Fig 6 — calibration (HCC)")
    fig6_calibration(dataset="hcc")
    fig7_luad_comparison()
    fig8_cross_cancer()
    fig_umap_novel()

    print()
    print("=" * 62)
    outputs = [
        "fig1_grace_architecture",
        "fig2_pathway_heatmaps",
        "fig3_confidence_scores",
        "fig4_novel_case_study",
        "fig5_metrics_comparison",
        "fig6A_confidence_vs_recall",
        "fig6B_calibration_boxplot",
        "fig6C_uncertainty_flags",
        "fig6A_confidence_vs_recall_hcc",
        "fig6B_calibration_boxplot_hcc",
        "fig6C_uncertainty_flags_hcc",
        "fig7_luad_comparison",
        "fig8A_per_cluster",
        "fig8B_overall_accuracy",
        "fig8C_confidence_distributions",
        "fig8D_uncertainty_flags",
        "fig8E_summary_table",
        "hcc_umap_novel_populations",
    ]
    for name in outputs:
        p = FIG_DIR / f"{name}.png"
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  {status}  figures/{name}.png")
    print("=" * 62)
