"""
HCC Preprocessing — GSE149614 (Ma et al. 2021, Cancer Cell)
=============================================================
Equivalent of day1_download_preprocess.py for the HCC dataset.
Filters to primary tumour cells, runs Scanpy QC, clustering, DEG analysis.

Run:
    python3 hcc_preprocess.py
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HCC_DIR     = Path("data/hcc")
COUNTS_FILE = HCC_DIR / "GSE149614_HCC_counts.txt.gz"
META_FILE   = HCC_DIR / "GSE149614_HCC_metadata.txt.gz"

HCC_RESULTS = Path("results/hcc")
HCC_FIGS    = Path("figures/hcc")
HCC_RESULTS.mkdir(parents=True, exist_ok=True)
HCC_FIGS.mkdir(parents=True, exist_ok=True)

# ── Parameters (mirroring LUAD day1 settings) ─────────────────────────────────
TISSUE_FILTER   = "Tumor"          # site column value for primary tumour
N_SUBSAMPLE     = 10000            # subsample to match LUAD analysis
MIN_GENES       = 200
MAX_GENES       = 5000
MAX_MITO_PCT    = 15
N_HVG           = 2000
N_PCS           = 30
N_NEIGHBORS     = 15
LEIDEN_RES      = 0.5
TOP_N_DEGS      = 50
RANDOM_STATE    = 42

sc.settings.verbosity = 1
sc.settings.figdir    = str(HCC_FIGS)


def main():
    print("=" * 60)
    print("HCC Preprocessing — GSE149614")
    print("=" * 60)

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata...")
    meta = pd.read_csv(META_FILE, sep="\t", index_col="Cell")
    print(f"  Total cells in metadata: {len(meta):,}")
    print(f"  Site distribution:\n{meta['site'].value_counts().to_string()}")
    print(f"\n  Cell type distribution:\n{meta['celltype'].value_counts().to_string()}")

    # Filter to primary tumour
    tumour_cells = meta[meta["site"] == TISSUE_FILTER].index.tolist()
    print(f"\n  Primary tumour cells (site=={TISSUE_FILTER!r}): {len(tumour_cells):,}")

    # Subsample if needed
    np.random.seed(RANDOM_STATE)
    if len(tumour_cells) > N_SUBSAMPLE:
        tumour_cells = np.random.choice(
            tumour_cells, N_SUBSAMPLE, replace=False
        ).tolist()
        print(f"  Subsampled to: {len(tumour_cells):,} cells")

    # ── 2. Load count matrix ──────────────────────────────────────────────────
    print("\n[2/5] Loading count matrix (genes × cells)...")
    print("  Reading compressed matrix — this may take 2-3 minutes...")

    # Read in chunks to handle large file
    chunks = []
    chunk_size = 5000

    # First get all column names
    import gzip
    with gzip.open(COUNTS_FILE, "rt") as f:
        header = f.readline().strip().split("\t")

    all_cells = header[1:]  # first col is gene name
    keep_idx  = [i + 1 for i, c in enumerate(all_cells) if c in set(tumour_cells)]
    keep_cells = [all_cells[i - 1] for i in keep_idx]

    print(f"  Cells to extract: {len(keep_cells):,}")
    print(f"  Reading selected columns from matrix...")

    # Use pandas read_csv with usecols for efficiency
    use_cols = [0] + keep_idx   # col 0 = gene names
    df_counts = pd.read_csv(
        COUNTS_FILE,
        sep="\t",
        index_col=0,
        usecols=use_cols,
        compression="gzip",
    )
    # Sync keep_cells to actual columns loaded (handles any off-by-one)
    keep_cells = df_counts.columns.tolist()
    # Sync keep_cells to actual columns loaded (handles any off-by-one)
    keep_cells = df_counts.columns.tolist()
    # Sync keep_cells to actual columns loaded (handles any off-by-one)
    keep_cells = df_counts.columns.tolist()
    # Sync keep_cells to actual columns loaded (handles any off-by-one)
    keep_cells = df_counts.columns.tolist()
    # Rename columns to match cell barcodes
    # Column names are set automatically from header row by pandas
    print(f"  Matrix shape: {df_counts.shape[0]:,} genes × {df_counts.shape[1]:,} cells")

    # ── 3. Create AnnData and QC ──────────────────────────────────────────────
    print("\n[3/5] Quality control...")

    # AnnData expects cells × genes
    adata = sc.AnnData(X=df_counts.T.values.astype(np.float32))
    adata.obs_names = df_counts.columns.tolist()
    adata.var_names = df_counts.index.tolist()

    # Attach metadata
    meta_sub = meta.loc[adata.obs_names, :]
    for col in ["sample", "site", "patient", "stage", "celltype"]:
        if col in meta_sub.columns:
            adata.obs[col] = meta_sub[col].values

    print(f"  AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Mito genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    print(f"  Before QC: {adata.n_obs:,} cells")
    print(f"  Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"  Median mito%: {adata.obs['pct_counts_mt'].median():.1f}%")

    # QC filters
    before = adata.n_obs
    adata = adata[adata.obs["n_genes_by_counts"] >= MIN_GENES].copy()
    adata = adata[adata.obs["n_genes_by_counts"] <= MAX_GENES].copy()
    adata = adata[adata.obs["pct_counts_mt"] <= MAX_MITO_PCT].copy()
    print(f"  After QC: {adata.n_obs:,} cells (removed {before - adata.n_obs:,})")

    # ── 4. Normalise, HVG, PCA, clustering ───────────────────────────────────
    print("\n[4/5] Preprocessing and clustering...")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat_v3", span=0.3
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  HVGs selected: {adata.n_vars}")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, random_state=RANDOM_STATE)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS,
                    random_state=RANDOM_STATE)
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    sc.tl.leiden(adata, resolution=LEIDEN_RES, random_state=RANDOM_STATE)

    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Leiden clusters (res={LEIDEN_RES}): {n_clusters}")
    print(f"  Cluster sizes:\n{adata.obs['leiden'].value_counts().sort_index().to_string()}")

    # ── 5. DEG analysis ───────────────────────────────────────────────────────
    print("\n[5/5] DEG analysis (Wilcoxon one-vs-rest)...")

    # Restore raw counts for DEG
    adata_raw = adata.raw.to_adata()
    adata_raw.obs["leiden"] = adata.obs["leiden"].values

    sc.tl.rank_genes_groups(
        adata_raw, "leiden", method="wilcoxon",
        n_genes=TOP_N_DEGS, pts=True
    )

    # Save DEGs per cluster
    degs_dir = HCC_RESULTS / "degs"
    degs_dir.mkdir(exist_ok=True)
    all_rows = []

    for cl in sorted(adata_raw.obs["leiden"].unique(), key=int):
        result = sc.get.rank_genes_groups_df(adata_raw, group=cl)
        result = result.head(TOP_N_DEGS)
        result["cluster"] = cl
        result.to_csv(degs_dir / f"cluster_{cl}_degs.csv", index=False)
        all_rows.append(result)
        top5 = result["names"].head(5).tolist()
        print(f"  Cluster {cl:>2}: top DEGs = {', '.join(top5)}")

    pd.concat(all_rows).to_csv(HCC_RESULTS / "hcc_all_degs.csv", index=False)

    # Author label purity per cluster
    purity = (
        adata.obs.groupby("leiden")["celltype"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    purity.columns = ["cluster", "author_label"]
    purity_pct = (
        adata.obs.groupby("leiden")["celltype"]
        .agg(lambda x: round(x.value_counts().iloc[0] / len(x) * 100, 1))
        .reset_index()
    )
    purity_pct.columns = ["cluster", "purity_pct"]
    purity = purity.merge(purity_pct, on="cluster")
    purity["n_cells"] = adata.obs["leiden"].value_counts().sort_index().values
    purity.to_csv(HCC_RESULTS / "hcc_author_labels_per_cluster.csv", index=False)

    print("\n  Author label per cluster:")
    print(purity.to_string(index=False))

    # ── Save processed AnnData ────────────────────────────────────────────────
    out_path = HCC_RESULTS / "gse149614_hcc_processed.h5ad"
    adata.write_h5ad(out_path)
    print(f"\n  Saved → {out_path}")

    # ── UMAP figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    umap = adata.obsm["X_umap"]
    leiden = adata.obs["leiden"].astype(str)
    clusters = sorted(leiden.unique(), key=int)
    cmap = plt.cm.get_cmap("tab20", len(clusters))

    for i, cl in enumerate(clusters):
        mask = leiden == cl
        axes[0].scatter(umap[mask, 0], umap[mask, 1],
                        c=[cmap(i)], s=2, alpha=0.7,
                        rasterized=True, linewidths=0)
        cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
        axes[0].text(cx, cy, cl, fontsize=7, ha="center", va="center",
                     fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.12",
                               facecolor="black", alpha=0.5, linewidth=0))

    CELLTYPE_COLORS = {
        "T/NK":         "#4E79A7",
        "Hepatocyte":   "#E15759",
        "Myeloid":      "#F28E2B",
        "B":            "#59A14F",
        "Endothelial":  "#B07AA1",
        "Fibroblast":   "#76B7B2",
    }
    celltypes = adata.obs["celltype"].astype(str)
    for ct, color in CELLTYPE_COLORS.items():
        mask = celltypes == ct
        if mask.any():
            axes[1].scatter(umap[mask, 0], umap[mask, 1],
                            c=[color], s=2, alpha=0.75, label=ct,
                            rasterized=True, linewidths=0)

    for ax in axes:
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    axes[0].set_title("A   Leiden clusters", loc="left", fontweight="bold")
    axes[1].set_title("B   Author cell type annotations", loc="left", fontweight="bold")
    axes[1].legend(markerscale=4, fontsize=8, loc="lower left",
                   title="Cell type", title_fontsize=8)

    fig.suptitle(f"GSE149614 HCC — primary tumour (Tumor, n = {adata.n_obs:,} cells, "
                 f"{n_clusters} clusters)", fontsize=11, y=1.01)
    plt.tight_layout(w_pad=3)
    fig.savefig(HCC_FIGS / "hcc_umap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → figures/hcc/hcc_umap.png")

    print("\n" + "=" * 60)
    print("HCC Preprocessing DONE")
    print("=" * 60)
    print(f"  Cells processed: {adata.n_obs:,}")
    print(f"  Leiden clusters: {n_clusters}")
    print(f"  results/hcc/gse149614_hcc_processed.h5ad")
    print(f"  results/hcc/degs/cluster_*.csv")
    print(f"  results/hcc/hcc_author_labels_per_cluster.csv")
    print(f"\nNext: python3 cell_identity_agent.py")


if __name__ == "__main__":
    main()
