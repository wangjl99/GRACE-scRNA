#!/usr/bin/env python3
"""
draw_fig2_cluster_composition.py
Generates Figure 2: UMAP + bar charts for LUAD and HCC.
ALL data from real h5ad and CSV files — no hardcoded values.

Usage:
    cd /data/jwang58/lung_scrnaseq
    conda activate luad_agents
    python3 draw_fig2_cluster_composition.py
"""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("WARNING: scanpy not installed. Run: pip install scanpy --break-system-packages")

RES     = Path("results")
HCC_DIR = RES/"hcc"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

H5AD_L = RES/"gse131907_tlung_processed.h5ad"
H5AD_H = HCC_DIR/"gse149614_hcc_processed.h5ad"
CSV_L  = RES/"author_labels_per_cluster.csv"
CSV_H  = HCC_DIR/"hcc_author_labels_per_cluster.csv"

LINEAGE_MAP_L = [
    ("T lymph","T lymphocyte"),("NK","T/NK"),("Myeloid","Myeloid"),
    ("B lymph","B lymphocyte"),("Plasma","B/Plasma"),
    ("Epithelial","Malignant epithelial"),("epithelial","Malignant epithelial"),
    ("Malignant","Malignant epithelial"),("Fibroblast","Fibroblast"),
    ("Endothelial","Endothelial"),("Mast","Mast"),("Proliferating","Proliferating"),
]
LINEAGE_MAP_H = [
    ("Hepato","Hepatocyte"),("Myeloid","Myeloid"),("T/NK","T/NK"),
    ("T lymph","T/NK"),("NK","T/NK"),("Endothel","Endothelial"),
    ("B cell","B cell"),("B lymph","B cell"),("Plasma","B cell/Plasma"),
    ("Fibro","Fibroblast"),
]
def assign_lineage(label, rules):
    for sub,lin in rules:
        if sub.lower() in str(label).lower(): return lin
    return str(label)

LUAD_COL = {
    "T lymphocyte":"#4CAF50","T/NK":"#66BB6A","Myeloid":"#FF9800",
    "B lymphocyte":"#9C27B0","B/Plasma":"#CE93D8",
    "Malignant epithelial":"#F44336","Fibroblast":"#795548",
    "Endothelial":"#2196F3","Mast":"#FF5722","Proliferating":"#607D8B",
}
HCC_COL = {
    "Hepatocyte":"#F44336","Myeloid":"#FF9800","T/NK":"#4CAF50",
    "Endothelial":"#2196F3","B cell":"#9C27B0","B cell/Plasma":"#CE93D8",
    "Fibroblast":"#795548",
}
DEFAULT = "#9E9E9E"
LUAD_NOVEL    = {15}
HCC_UNCERTAIN = {5,10,11,12}

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "font.family":"sans-serif","pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False,
})

def load_csv(path, lmap):
    df = pd.read_csv(path)
    df["cluster"] = df["cluster"].astype(int)
    df = df.sort_values("cluster").reset_index(drop=True)
    df["lineage"] = df["author_label"].apply(lambda x: assign_lineage(x,lmap))
    return df

def load_umap(h5ad_path, csv_df):
    if not HAS_SCANPY or not h5ad_path.exists():
        print(f"  Skipping UMAP: {'no scanpy' if not HAS_SCANPY else h5ad_path.name+' not found'}")
        return None, None, None
    print(f"  Reading {h5ad_path.name}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}  obsm: {list(adata.obsm.keys())}")
    if "X_umap" not in adata.obsm:
        print("  Computing UMAP...")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    # Find cluster column
    cluster_col = next((c for c in ["leiden","louvain","cluster","clusters"]
                        if c in adata.obs.columns), None)
    if cluster_col is None:
        print("  WARNING: no cluster column found")
        return umap, np.zeros(adata.n_obs,dtype=int), np.full(adata.n_obs,"Unknown")
    clusters = adata.obs[cluster_col].astype(int).values
    cl2lin   = {int(r["cluster"]):r["lineage"] for _,r in csv_df.iterrows()}
    lineages = np.array([cl2lin.get(c,DEFAULT) for c in clusters])
    return umap, clusters, lineages

def draw_umap(ax, umap, clusters, lineages, csv_df, col_map,
              special_set, sp_edge, sp_label, title):
    if umap is None:
        ax.text(0.5,0.5,"UMAP not available",ha="center",va="center",
                transform=ax.transAxes,fontsize=12,color="#999")
        ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
        ax.axis("off"); return
    for lin in sorted(set(lineages)):
        mask = lineages==lin
        ax.scatter(umap[mask,0],umap[mask,1],
                   c=col_map.get(lin,DEFAULT),s=1.5,alpha=0.5,
                   edgecolors="none",rasterized=True,
                   label=f"{lin} (n={mask.sum():,})")
    # Highlight special clusters
    cl2lin = {int(r["cluster"]):r["lineage"] for _,r in csv_df.iterrows()}
    for sp in special_set:
        m = clusters==sp
        if m.sum()==0: continue
        ax.scatter(umap[m,0],umap[m,1],
                   c=col_map.get(cl2lin.get(sp,""),DEFAULT),
                   s=8,alpha=0.9,edgecolors=sp_edge,
                   linewidths=0.5,rasterized=True,zorder=4)
    # Cluster centroids labels
    for _,row in csv_df.iterrows():
        cl = int(row["cluster"]); m = clusters==cl
        if m.sum()==0: continue
        cx,cy = umap[m,0].mean(),umap[m,1].mean()
        ax.text(cx,cy,str(cl),fontsize=5.5,ha="center",va="center",
                fontweight="bold",color="white",
                bbox=dict(boxstyle="round,pad=0.08",
                          fc=col_map.get(cl2lin.get(cl,""),DEFAULT),
                          alpha=0.75,ec="none"))
    ax.set_xlabel("UMAP 1",fontsize=9)
    ax.set_ylabel("UMAP 2",fontsize=9)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
    handles = [mpatches.Patch(fc=col_map.get(l,DEFAULT),alpha=0.85,label=l)
               for l in sorted(set(lineages))]
    if special_set:
        handles.append(mpatches.Patch(fc=DEFAULT,alpha=0.7,
                       ec=sp_edge,lw=1.5,label=sp_label))
    ax.legend(handles=handles,fontsize=7.5,loc="lower left",
              framealpha=0.95,edgecolor="#ccc",
              ncol=1 if len(handles)<=6 else 2)

def draw_bars(ax, csv_df, col_map, special_set, sp_edge, sp_sym, title):
    x    = np.arange(len(csv_df))
    cols = [col_map.get(r["lineage"],DEFAULT) for _,r in csv_df.iterrows()]
    bars = ax.bar(x,csv_df["n_cells"],0.70,color=cols,alpha=0.85,edgecolor="white")
    for i,(_,row) in enumerate(csv_df.iterrows()):
        if int(row["cluster"]) in special_set:
            bars[i].set_edgecolor(sp_edge); bars[i].set_linewidth(2.5)
    maxn = csv_df["n_cells"].max()
    for i,(_,row) in enumerate(csv_df.iterrows()):
        pur = float(row["purity_pct"])
        col = "#C62828" if pur<80 else "#E65100" if pur<92 else "#444"
        ax.text(i,row["n_cells"]+maxn*0.022,f"{pur:.0f}%",
                ha="center",fontsize=6,color=col,fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{int(r['cluster'])}{sp_sym if int(r['cluster']) in special_set else ''}"
         f"\n{str(r['author_label'])[:7]}"
         for _,r in csv_df.iterrows()],
        fontsize=6.5 if len(csv_df)>20 else 7.5)
    ax.set_ylabel("Number of cells",fontsize=10.5)
    ax.set_ylim(0,maxn*1.30)
    total = int(csv_df["n_cells"].sum())
    n_cl  = len(csv_df)
    n99   = sum(1 for p in csv_df["purity_pct"] if float(p)>=99)
    ax.set_title(title+f"\nTotal: {total:,} cells | {n_cl} clusters | "
                 f"Purity ≥99%: {n99}/{n_cl} | purity (%) above bars",
                 loc="left",fontweight="bold",fontsize=11)
    seen = {}
    for _,r in csv_df.iterrows():
        if r["lineage"] not in seen:
            seen[r["lineage"]] = col_map.get(r["lineage"],DEFAULT)
    leg = [mpatches.Patch(fc=c,alpha=0.85,label=k) for k,c in seen.items()]
    if special_set:
        lbl = ("★ C15: novel KRAS subpopulation" if sp_sym=="★"
               else "† High-uncertainty hepatocyte (C5,C10,C11,C12)")
        leg.append(mpatches.Patch(fc=DEFAULT,alpha=0.7,ec=sp_edge,lw=2.5,label=lbl))
    ax.legend(handles=leg,fontsize=7.5,loc="upper right",
              framealpha=0.97,ncol=2,edgecolor="#ccc")

def main():
    print("="*60)
    print("draw_fig2_cluster_composition.py")
    print("="*60)

    print("\nLoading CSV files...")
    df_l = load_csv(CSV_L, LINEAGE_MAP_L)
    df_h = load_csv(CSV_H, LINEAGE_MAP_H)
    print(f"LUAD: {len(df_l)} clusters  {df_l['n_cells'].sum():,} cells")
    print(df_l[["cluster","author_label","n_cells","purity_pct","lineage"]].to_string(index=False))
    print(f"\nHCC:  {len(df_h)} clusters  {df_h['n_cells'].sum():,} cells")
    print(df_h[["cluster","author_label","n_cells","purity_pct","lineage"]].to_string(index=False))

    # Save verified JSON
    json.dump({
        "luad":df_l[["cluster","author_label","n_cells","purity_pct","lineage"]].to_dict(orient="records"),
        "hcc": df_h[["cluster","author_label","n_cells","purity_pct","lineage"]].to_dict(orient="records"),
    }, open(RES/"cluster_composition_real.json","w"), indent=2)
    print("\nSaved → results/cluster_composition_real.json")

    print("\nLoading UMAP from h5ad files...")
    umap_l,cl_l,lin_l = load_umap(H5AD_L, df_l)
    umap_h,cl_h,lin_h = load_umap(H5AD_H, df_h)

    fig = plt.figure(figsize=(26,18))
    gs  = gridspec.GridSpec(2,2,figure=fig,
                            hspace=0.42,wspace=0.28,
                            top=0.91,bottom=0.05,
                            left=0.04,right=0.97)

    draw_umap(fig.add_subplot(gs[0,0]),
              umap_l,cl_l,lin_l,df_l,LUAD_COL,LUAD_NOVEL,
              "#7D3C00","★ C15: novel KRAS subpopulation",
              "A   UMAP — LUAD (GSE131907, Kim et al. 2020)")

    draw_umap(fig.add_subplot(gs[0,1]),
              umap_h,cl_h,lin_h,df_h,HCC_COL,HCC_UNCERTAIN,
              "#FF8F00","† High-uncertainty hepatocyte (C5,C10,C11,C12)",
              "B   UMAP — HCC (GSE149614, Lu et al. 2022, zero-shot)")

    draw_bars(fig.add_subplot(gs[1,0]),
              df_l,LUAD_COL,LUAD_NOVEL,"#7D3C00","★",
              "C   Cell counts per cluster — LUAD")

    draw_bars(fig.add_subplot(gs[1,1]),
              df_h,HCC_COL,HCC_UNCERTAIN,"#FF8F00","†",
              "D   Cell counts per cluster — HCC (zero-shot)")

    fig.add_artist(plt.Line2D([0.03,0.97],[0.50,0.50],
                   transform=fig.transFigure,
                   color="#CCCCCC",lw=0.8,ls="--"))
    for yp,txt in [(0.75,"UMAP"),(0.25,"Cell\ncounts")]:
        fig.text(0.005,yp,txt,ha="center",va="center",fontsize=9,
                 fontweight="bold",color="#1A5276",rotation=90,
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="#EBF5FB",ec="#1A5276",lw=1.0))

    fig.suptitle(
        "Figure 2: Cluster composition — LUAD (A, C) and HCC zero-shot (B, D)\n"
        "UMAP from h5ad files  |  Cell counts and purity from author_labels CSVs",
        fontsize=12,fontweight="bold",y=0.975)

    for ext in ["png","pdf"]:
        out = FIG_DIR/f"fig2_cluster_composition.{ext}"
        fig.savefig(out,dpi=300,bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close()

if __name__=="__main__":
    main()
