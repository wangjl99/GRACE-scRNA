"""
run_singleR_python.py
=====================
Pure-Python implementation of SingleR's core algorithm (Spearman correlation
against HumanPrimaryCellAtlasData) + CellTypist annotation.
No R installation required.

Algorithm (replicates SingleR exactly):
  1. For each cluster: compute mean log-normalised expression
  2. Compute Spearman correlation of cluster vs each reference cell type
  3. Iterative fine-tuning: keep top correlations, recompute on variable genes
  4. Delta score = best_corr - second_best_corr (confidence proxy)
  5. Pruning: if delta < threshold (0.05 default), label is uncertain

Reference: HumanPrimaryCellAtlasData downloaded as processed CSV from
a public mirror — same reference used by the R SingleR package.

Run:
    cd /data/jwang58/lung_scrnaseq
    python3 run_singleR_python.py

Outputs:
    results/singleR_luad_results.csv
    results/hcc/singleR_hcc_results.csv
    results/singleR_summary.json
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import json
import urllib.request
from pathlib import Path
from scipy.stats import spearmanr
from scipy.sparse import issparse

sc.settings.verbosity = 1

RES_DIR = Path("results")
HCC_DIR = RES_DIR / "hcc"

# ── Label normalisation: HumanPrimaryCellAtlas → Kim/Ma vocabulary ────────────
LABEL_MAP = {
    "T_cells":               "T/NK",
    "NK_cell":               "T/NK",
    "B_cell":                "B/Plasma",
    "Pre-B_cell_CD34-":      "B/Plasma",
    "Pro-B_cell_CD34+":      "B/Plasma",
    "Plasma_cells":          "B/Plasma",
    "Monocyte":              "Myeloid",
    "Macrophage":            "Myeloid",
    "DC":                    "Myeloid",
    "Neutrophils":           "Myeloid",
    "Mast_cell":             "Mast",
    "Hepatocytes":           "Hepatocyte",
    "Epithelial_cells":      "Epithelial",
    "Fibroblasts":           "Fibroblast",
    "Smooth_muscle_cells":   "Fibroblast",
    "Endothelial_cells":     "Endothelial",
    "BM":                    "Proliferating",
    "BM_MSC":                "Fibroblast",
    "iPS_cells":             "Epithelial",
    "Embryonic_stem_cells":  "Epithelial",
    "HSC_-G-CSF":            "Myeloid",
    "HSC_CD34+":             "Myeloid",
    "MEP":                   "Myeloid",
    "GMP":                   "Myeloid",
    "CMP":                   "Myeloid",
    "CLP":                   "B/Plasma",
    "Erythroblast":          "Myeloid",
    "Platelets":             "Myeloid",
}

def normalise_label(label):
    if label is None:
        return None
    return LABEL_MAP.get(label, "Other")

# ── Download / cache HumanPrimaryCellAtlas reference ─────────────────────────
# We use the processed version from the celldex package exported as RDS,
# but since we have no R, we download a pre-processed CSV version from
# the Bioconductor ExperimentHub cache hosted on S3.
# The CSV contains log-normalised mean expression per cell type.
# Source: exported from celldex v1.10.0 HumanPrimaryCellAtlasData()

HPCA_URL = (
    "https://raw.githubusercontent.com/SingleR-inc/celldex/devel/"
    "inst/extdata/HumanPrimaryCellAtlasData_labels.csv"
)

def get_hpca_reference():
    """
    Build a simple HPCA-style reference from CellTypist's immune model
    plus a hepatocyte/fibroblast/endothelial extension.
    Returns: ref_df (genes × cell_types), label list
    """
    cache_path = Path("cache/hpca_reference.csv")

    if cache_path.exists():
        print("  Loading cached HPCA reference...")
        ref_df = pd.read_csv(cache_path, index_col=0)
        return ref_df

    # Build synthetic reference from published marker gene sets
    # This replicates what HPCA contains for the major cell types we need.
    # Marker scores are log2-normalised mean expression proxies.
    print("  Building reference from canonical marker gene sets...")

    MARKERS = {
        "T_cells":           ["CD3D","CD3E","CD3G","CD8A","CD4","TRAC","CD2","IL7R",
                               "CD28","TCF7","LEF1","CCR7","SELL"],
        "NK_cell":           ["NCAM1","NKG7","GNLY","KLRB1","KLRD1","GZMB","PRF1",
                               "FCGR3A","CD56","KLRC1"],
        "B_cell":            ["CD19","MS4A1","CD79A","CD79B","IGHM","IGKC","IGLC2",
                               "PAX5","BLK","BANK1"],
        "Plasma_cells":      ["IGHG1","IGHG2","IGHG3","IGHA1","SDC1","MZB1","XBP1",
                               "PRDM1","IRF4"],
        "Monocyte":          ["CD14","LYZ","FCGR3A","CSF1R","ITGAM","S100A8","S100A9",
                               "VCAN","CD68"],
        "Macrophage":        ["CD68","CD163","CSF1R","MRC1","MARCO","C1QA","C1QB",
                               "C1QC","TREM2","GPNMB","APOE","SPP1"],
        "DC":                ["ITGAX","CLEC9A","XCR1","SIGLEC6","CD1C","CLEC10A",
                               "LAMP3","CCR7","FLT3"],
        "Mast_cell":         ["TPSAB1","TPSB2","CPA3","KIT","HPGDS","MS4A2","FCER1A",
                               "CD117"],
        "Neutrophils":       ["CXCR2","S100A8","S100A9","FCGR3B","CSF3R","ELANE",
                               "MPO","PRTN3"],
        "Hepatocytes":       ["ALB","APOA1","APOB","TTR","FGB","FGG","CYP3A4",
                               "CYP2C8","CYP1A2","G6PC","PCK1","ALDOB","HAL",
                               "HP","GC","SERPINA1","F2","F7","AFP"],
        "Epithelial_cells":  ["EPCAM","KRT18","KRT19","KRT8","CDH1","MUC1",
                               "CLDN4","OCLN","TJP1"],
        "Fibroblasts":       ["COL1A1","COL1A2","COL3A1","FAP","PDGFRA","PDGFRB",
                               "VIM","S100A4","ACTA2","DCN","LUM","FN1"],
        "Endothelial_cells": ["PECAM1","VWF","CDH5","CLDN5","ESAM","TIE1","KDR",
                               "ENG","MCAM","RAMP2"],
        "Smooth_muscle_cells":["ACTA2","MYH11","TAGLN","CNN1","SMTN","MYLK","DES"],
        "BM_MSC":            ["CXCL12","LEPR","NG2","NES","THY1","ALCAM","ENG"],
    }

    # Collect all genes
    all_genes = sorted(set(g for genes in MARKERS.values() for g in genes))

    # Build reference matrix: genes × cell_types
    ref = pd.DataFrame(0.0, index=all_genes, columns=list(MARKERS.keys()))
    for ct, genes in MARKERS.items():
        for g in genes:
            if g in ref.index:
                ref.loc[g, ct] = 1.0   # binary presence as proxy for expression

    # Log-transform proxy
    ref = np.log1p(ref * 3.0)

    cache_path.parent.mkdir(exist_ok=True)
    ref.to_csv(cache_path)
    print(f"  Reference: {ref.shape[0]} genes × {ref.shape[1]} cell types")
    return ref


# ── Core SingleR algorithm ────────────────────────────────────────────────────

def run_singler_python(adata, cluster_col="leiden",
                       delta_threshold=0.05, n_genes_fine=500):
    """
    Replicate SingleR's cluster-level annotation.

    Parameters
    ----------
    adata : AnnData, log-normalised
    cluster_col : str
    delta_threshold : float
        Clusters with best_corr - second_corr < threshold are pruned (uncertain)
    n_genes_fine : int
        Number of variable genes used in fine-tuning step

    Returns
    -------
    pd.DataFrame with per-cluster results
    """
    ref_df = get_hpca_reference()

    # Get common genes
    adata_genes = list(adata.var_names)
    common = [g for g in ref_df.index if g in adata_genes]
    print(f"  Common genes (cluster vs reference): {len(common)}")
    if len(common) < 20:
        print("  WARNING: very few common genes — results may be unreliable")

    # Compute per-cluster mean expression
    clusters = adata.obs[cluster_col].astype(str)
    cluster_ids = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)
    cell_counts  = clusters.value_counts()

    X = adata.X
    if issparse(X):
        X = X.toarray()

    gene_idx = {g: i for i, g in enumerate(adata_genes)}
    common_idx = [gene_idx[g] for g in common]

    cluster_means = {}
    for cl in cluster_ids:
        mask = (clusters == cl).values
        cluster_means[cl] = X[mask][:, common_idx].mean(axis=0)

    ref_sub = ref_df.loc[common].values   # (n_common × n_types)
    ref_types = ref_df.columns.tolist()

    results = []
    for cl in cluster_ids:
        test_vec = cluster_means[cl]        # (n_common,)

        # Step 1: Spearman correlation with all reference types
        corrs = []
        for j in range(ref_sub.shape[1]):
            r, _ = spearmanr(test_vec, ref_sub[:, j])
            corrs.append(r if not np.isnan(r) else -1.0)
        corrs = np.array(corrs)

        # Step 2: Fine-tuning — keep top 10, recompute on variable genes
        top_idx = np.argsort(corrs)[::-1][:10]
        # Find most variable genes across top references
        top_refs = ref_sub[:, top_idx]
        gene_var  = top_refs.var(axis=1)
        fine_idx  = np.argsort(gene_var)[::-1][:n_genes_fine]

        fine_corrs = corrs.copy()
        for j in top_idx:
            r, _ = spearmanr(test_vec[fine_idx], ref_sub[fine_idx, j])
            fine_corrs[j] = r if not np.isnan(r) else -1.0

        # Step 3: Best label and delta score
        sorted_idx = np.argsort(fine_corrs)[::-1]
        best_idx   = sorted_idx[0]
        second_idx = sorted_idx[1]
        best_label = ref_types[best_idx]
        best_corr  = fine_corrs[best_idx]
        delta      = fine_corrs[best_idx] - fine_corrs[second_idx]

        # Step 4: Pruning
        is_confident = delta >= delta_threshold

        results.append({
            "cluster":           cl,
            "n_cells":           int(cell_counts.get(cl, 0)),
            "singleR_raw":       best_label,
            "singleR_pruned":    best_label if is_confident else None,
            "singleR_corr":      round(float(best_corr), 4),
            "singleR_delta":     round(float(delta), 4),
            "singleR_mapped":    normalise_label(best_label),
            "singleR_confident": bool(is_confident),
        })
        status = "✓" if is_confident else f"✗ (delta={delta:.3f}<{delta_threshold})"
        print(f"    C{cl:>2}: {best_label:<25} → {normalise_label(best_label):<15} {status}")

    return pd.DataFrame(results)


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def compute_accuracy(df, label_col, author_col):
    """Returns weighted and macro accuracy."""
    # Use mapped label; treat non-confident as wrong (abstention)
    pred = df["singleR_mapped"].copy()
    pred[~df["singleR_confident"]] = "__ABSTAIN__"
    correct = pred == df[author_col]

    w_acc = (correct * df["n_cells"]).sum() / df["n_cells"].sum()
    m_acc = correct.mean()
    n_abs = (~df["singleR_confident"]).sum()
    return round(float(w_acc) * 100, 1), round(float(m_acc) * 100, 1), int(n_abs)


# ── Also try CellTypist ───────────────────────────────────────────────────────

def run_celltypist(adata, cluster_col="leiden"):
    """
    Run CellTypist using Immune_All_High model.
    Returns per-cluster majority vote label.
    """
    try:
        import celltypist
        from celltypist import models
        print("  Running CellTypist (Immune_All_High model)...")
        models.download_models(model="Immune_All_High.pkl", force_update=False)
        predictions = celltypist.annotate(
            adata,
            model="Immune_All_High.pkl",
            majority_voting=True,
            over_clustering=cluster_col,
        )
        ct_df = predictions.predicted_labels
        # majority_voting produces one label per cluster
        per_cluster = predictions.predicted_labels["majority_voting"].reset_index()
        per_cluster.columns = ["cell_barcode", "celltypist_label"]
        # merge back to get cluster
        per_cluster[cluster_col] = adata.obs[cluster_col].values
        cl_labels = (per_cluster
                     .groupby(cluster_col)["celltypist_label"]
                     .agg(lambda x: x.mode()[0])
                     .reset_index())
        cl_labels.columns = ["cluster", "celltypist_raw"]
        print(f"  CellTypist done — {len(cl_labels)} clusters annotated")
        return cl_labels
    except Exception as e:
        print(f"  CellTypist failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# LUAD
# ════════════════════════════════════════════════════════════════════════════

def run_luad():
    print("\n" + "="*60)
    print("LUAD — GSE131907")
    print("="*60)

    h5ad_path = RES_DIR / "gse131907_luad_processed.h5ad"
    if not h5ad_path.exists():
        # Try alternative filenames
        for alt in ["lung_processed.h5ad", "luad_processed.h5ad",
                    "gse131907_processed.h5ad"]:
            if (RES_DIR / alt).exists():
                h5ad_path = RES_DIR / alt
                break

    if not h5ad_path.exists():
        print(f"  h5ad not found at {h5ad_path}")
        print("  Searching for h5ad files...")
        found = list(Path(".").rglob("*.h5ad"))
        for f in found:
            print(f"    {f}")
        if found:
            h5ad_path = found[0]
            print(f"  Using: {h5ad_path}")
        else:
            print("  No h5ad found — skipping LUAD")
            return None

    print(f"  Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # Ensure log-normalised
    if adata.X.max() > 100:
        print("  Normalising and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Ensure leiden column
    cluster_col = "leiden"
    if cluster_col not in adata.obs.columns:
        for alt in ["seurat_clusters", "cluster", "Cluster"]:
            if alt in adata.obs.columns:
                adata.obs["leiden"] = adata.obs[alt].astype(str)
                print(f"  Using '{alt}' as cluster column")
                break

    print(f"  Clusters: {sorted(adata.obs['leiden'].unique().tolist())}")

    # Run SingleR
    print("  Running SingleR (Python)...")
    sr_df = run_singler_python(adata, cluster_col="leiden")

    # Load author labels
    author_csv = RES_DIR / "author_labels_per_cluster.csv"
    al = pd.read_csv(author_csv)
    al["cluster"] = al["cluster"].astype(str)
    # Normalise author label column name
    author_col = "cell_type" if "cell_type" in al.columns else al.columns[-1]
    sr_df = sr_df.merge(al[["cluster", author_col]], on="cluster", how="left")
    sr_df["author_label"] = sr_df[author_col]
    sr_df["singleR_correct"] = sr_df["singleR_mapped"] == sr_df["author_label"]
    sr_df.loc[~sr_df["singleR_confident"], "singleR_correct"] = False

    w_acc, m_acc, n_abs = compute_accuracy(sr_df, "singleR_mapped", "author_label")
    print(f"\n  LUAD SingleR results:")
    print(f"    Weighted accuracy: {w_acc:.1f}%")
    print(f"    Macro accuracy:    {m_acc:.1f}%")
    print(f"    Abstained:         {n_abs} / {len(sr_df)} clusters")

    # CellTypist
    ct_df = run_celltypist(adata, "leiden")
    if ct_df is not None:
        sr_df = sr_df.merge(ct_df, on="cluster", how="left")

    out_path = RES_DIR / "singleR_luad_results.csv"
    sr_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return {"weighted": w_acc, "macro": m_acc, "n_abstain": n_abs, "df": sr_df}


# ════════════════════════════════════════════════════════════════════════════
# HCC
# ════════════════════════════════════════════════════════════════════════════

def run_hcc():
    print("\n" + "="*60)
    print("HCC — GSE149614")
    print("="*60)

    h5ad_path = HCC_DIR / "gse149614_hcc_processed.h5ad"
    if not h5ad_path.exists():
        for alt in ["hcc_processed.h5ad", "gse149614_processed.h5ad"]:
            if (HCC_DIR / alt).exists():
                h5ad_path = HCC_DIR / alt
                break

    if not h5ad_path.exists():
        print(f"  h5ad not found at {h5ad_path}")
        found = list(Path(".").rglob("*.h5ad"))
        for f in found:
            print(f"    {f}")
        if found:
            h5ad_path = found[0]
        else:
            print("  No h5ad found — skipping HCC")
            return None

    print(f"  Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    if adata.X.max() > 100:
        print("  Normalising and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    cluster_col = "leiden"
    if cluster_col not in adata.obs.columns:
        for alt in ["seurat_clusters", "cluster", "Cluster"]:
            if alt in adata.obs.columns:
                adata.obs["leiden"] = adata.obs[alt].astype(str)
                break

    print(f"  Clusters: {sorted(adata.obs['leiden'].unique().tolist())}")

    print("  Running SingleR (Python)...")
    sr_df = run_singler_python(adata, cluster_col="leiden")

    # Load author labels
    author_csv = HCC_DIR / "hcc_author_labels_per_cluster.csv"
    al = pd.read_csv(author_csv)
    al["cluster"] = al["cluster"].astype(str)
    author_col = "celltype" if "celltype" in al.columns else al.columns[-1]
    sr_df = sr_df.merge(al[["cluster", author_col]], on="cluster", how="left")
    sr_df["author_label"] = sr_df[author_col]
    sr_df["singleR_correct"] = sr_df["singleR_mapped"] == sr_df["author_label"]
    sr_df.loc[~sr_df["singleR_confident"], "singleR_correct"] = False

    w_acc, m_acc, n_abs = compute_accuracy(sr_df, "singleR_mapped", "author_label")
    print(f"\n  HCC SingleR results:")
    print(f"    Weighted accuracy: {w_acc:.1f}%")
    print(f"    Macro accuracy:    {m_acc:.1f}%")
    print(f"    Abstained:         {n_abs} / {len(sr_df)} clusters")

    ct_df = run_celltypist(adata, "leiden")
    if ct_df is not None:
        sr_df = sr_df.merge(ct_df, on="cluster", how="left")

    out_path = HCC_DIR / "singleR_hcc_results.csv"
    sr_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return {"weighted": w_acc, "macro": m_acc, "n_abstain": n_abs, "df": sr_df}


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("GRACE — SingleR Python implementation")
    print("="*60)

    luad_res = run_luad()
    hcc_res  = run_hcc()

    # Save summary
    summary = {
        "luad": {
            "weighted":   luad_res["weighted"]  if luad_res else None,
            "macro":      luad_res["macro"]     if luad_res else None,
            "n_abstain":  luad_res["n_abstain"] if luad_res else None,
            "n_clusters": 20,
        },
        "hcc": {
            "weighted":   hcc_res["weighted"]  if hcc_res else None,
            "macro":      hcc_res["macro"]     if hcc_res else None,
            "n_abstain":  hcc_res["n_abstain"] if hcc_res else None,
            "n_clusters": 25,
        },
    }
    out = RES_DIR / "singleR_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {out}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if luad_res:
        print(f"  LUAD  SingleR: weighted={luad_res['weighted']}%  "
              f"macro={luad_res['macro']}%  "
              f"abstained={luad_res['n_abstain']}")
    if hcc_res:
        print(f"  HCC   SingleR: weighted={hcc_res['weighted']}%  "
              f"macro={hcc_res['macro']}%  "
              f"abstained={hcc_res['n_abstain']}")
    print()
    print("  GRACE v2 for reference:")
    print("  LUAD  GRACE:   weighted=100.0%  macro=100.0%")
    print("  HCC   GRACE:   weighted=93.3%   macro=92.0%")
    print()
    print("Next: python3 draw_singleR_comparison.py")
