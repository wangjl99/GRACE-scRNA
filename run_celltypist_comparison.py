#!/usr/bin/env python3
"""
run_celltypist_comparison.py
============================
Runs CellTypist cell type annotation on LUAD and HCC scRNA-seq datasets
and computes accuracy against published author labels.

Usage
-----
    cd /data/jwang58/lung_scrnaseq
    conda activate luad_agents
    pip install celltypist --break-system-packages   # first time only
    python3 run_celltypist_comparison.py

Output files (all in results/)
------------------------------
    results/celltypist_luad_results.csv
        Per-cluster: predicted label, author label, correct (bool), n_cells
    results/celltypist_luad_summary.json
        Weighted accuracy, macro accuracy, per-cluster details

    results/hcc/celltypist_hcc_results.csv
    results/hcc/celltypist_hcc_summary.json

    results/celltypist_accuracy_summary.json
        Combined summary for both datasets — feeds directly into
        draw_fig_accuracy_comparison.py and Table 2

GitHub file name : run_celltypist_comparison.py
Results prefix   : celltypist_*
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
RES      = Path("results")
HCC_DIR  = RES / "hcc"
H5AD_L   = RES  / "gse131907_tlung_processed.h5ad"
H5AD_H   = HCC_DIR / "gse149614_hcc_processed.h5ad"
CSV_L    = RES  / "author_labels_per_cluster.csv"
CSV_H    = HCC_DIR / "hcc_author_labels_per_cluster.csv"

# ── Vocabulary normalisation (same as used for GRACE/SingleR accuracy) ─────
# Maps CellTypist predicted labels → standard vocabulary
# Extend this dict if CellTypist returns labels not yet covered
VOCAB_MAP = {
    # T cells
    "CD4+ T cells":                          "T lymphocytes",
    "CD8+ T cells":                          "T lymphocytes",
    "CD4+ T cells, memory":                  "T lymphocytes",
    "CD8+ T cells, memory":                  "T lymphocytes",
    "Regulatory T cells":                    "T lymphocytes",
    "T cells":                               "T lymphocytes",
    "NKT cells":                             "T lymphocytes",
    "NK cells":                              "T lymphocytes",
    "Innate lymphoid cells":                 "T lymphocytes",
    "Proliferating T cells":                 "Proliferating cells",
    # Myeloid
    "Macrophages":                           "Myeloid cells",
    "Monocytes":                             "Myeloid cells",
    "Dendritic cells":                       "Myeloid cells",
    "Plasmacytoid dendritic cells":          "Myeloid cells",
    "Classical monocytes":                   "Myeloid cells",
    "Non-classical monocytes":               "Myeloid cells",
    "LAMP3+ DCs":                            "Myeloid cells",
    "Macrophages M1":                        "Myeloid cells",
    "Macrophages M2":                        "Myeloid cells",
    # B / Plasma
    "B cells":                               "B lymphocytes",
    "Plasma cells":                          "B lymphocytes",
    "Plasmablasts":                          "B lymphocytes",
    "Memory B cells":                        "B lymphocytes",
    "Naive B cells":                         "B lymphocytes",
    # Epithelial / Malignant
    "Epithelial cells":                      "Epithelial cells",
    "Alveolar cells type 1":                 "Epithelial cells",
    "Alveolar cells type 2":                 "Epithelial cells",
    "Club cells":                            "Epithelial cells",
    "Ciliated cells":                        "Epithelial cells",
    "Goblet cells":                          "Epithelial cells",
    # Fibroblast / Stromal
    "Fibroblasts":                           "Fibroblasts",
    "Smooth muscle cells":                   "Fibroblasts",
    "Pericytes":                             "Fibroblasts",
    # Endothelial
    "Endothelial cells":                     "Endothelial cells",
    # Mast
    "Mast cells":                            "Mast cells",
    # HCC specific
    "Hepatocytes":                           "Hepatocytes",
    "Hepatic stellate cells":               "Fibroblasts",
    "Kupffer cells":                         "Myeloid cells",
    "Cholangiocytes":                        "Epithelial cells",
}

# Author label normalisation — maps Kim 2020 / Lu 2022 labels to standard vocab
AUTHOR_VOCAB_L = {
    "T lymphocytes":         "T lymphocytes",
    "Myeloid cells":         "Myeloid cells",
    "B lymphocytes":         "B lymphocytes",
    "Plasma cells":          "B lymphocytes",
    "Epithelial cells":      "Epithelial cells",
    "Malignant cells":       "Epithelial cells",
    "Fibroblasts":           "Fibroblasts",
    "Endothelial cells":     "Endothelial cells",
    "Mast cells":            "Mast cells",
    "Proliferating cells":   "Proliferating cells",
}
AUTHOR_VOCAB_H = {
    "Hepatocytes":           "Hepatocytes",
    "Myeloid cells":         "Myeloid cells",
    "T cells":               "T lymphocytes",
    "NK cells":              "T lymphocytes",
    "B cells":               "B lymphocytes",
    "Endothelial cells":     "Endothelial cells",
    "Fibroblasts":           "Fibroblasts",
}

def normalise(label: str, vocab: dict) -> str:
    """Normalise a cell type label using vocabulary map."""
    if label in vocab:
        return vocab[label]
    # Partial match
    for k, v in vocab.items():
        if k.lower() in label.lower() or label.lower() in k.lower():
            return v
    return label   # keep original if no match found


def run_celltypist(h5ad_path: Path, csv_path: Path,
                   author_vocab: dict, dataset_name: str,
                   model_name: str = "Immune_All_Low.pkl"):
    """
    Run CellTypist on one dataset and compute per-cluster accuracy.

    Parameters
    ----------
    h5ad_path   : path to processed h5ad file
    csv_path    : path to author_labels CSV with columns:
                  cluster, author_label, n_cells, purity_pct
    author_vocab: vocabulary map for author labels
    dataset_name: "LUAD" or "HCC" (for printing)
    model_name  : CellTypist model to use
    """
    import celltypist
    import scanpy as sc

    print(f"\n{'='*60}")
    print(f"Running CellTypist on {dataset_name}")
    print(f"  h5ad : {h5ad_path}")
    print(f"  model: {model_name}")
    print(f"{'='*60}")

    # ── Load h5ad ──────────────────────────────────────────────────────────
    print("Loading h5ad...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # ── Normalise for CellTypist (log1p CPM required) ─────────────────────
    # Check if already normalised
    adata_ct = adata.copy()
    if adata_ct.X.max() > 100:
        print("  Normalising counts (library size → 10,000, log1p)...")
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
    else:
        print("  Data appears already log-normalised — using as-is")

    # ── Download model if needed ───────────────────────────────────────────
    print(f"  Downloading/loading model: {model_name}")
    celltypist.models.download_models(model=model_name, force_update=False)

    # ── Run CellTypist ─────────────────────────────────────────────────────
    print("  Running annotation (majority_voting=True)...")
    predictions = celltypist.annotate(
        adata_ct,
        model=model_name,
        majority_voting=True,   # assigns most common label per cluster
        over_clustering="leiden" if "leiden" in adata.obs.columns
                        else "louvain"
    )

    # ── Extract per-cell predictions ───────────────────────────────────────
    pred_labels = predictions.predicted_labels

    # Get cluster assignment per cell
    cluster_col = next(
        (c for c in ["leiden","louvain","cluster","clusters"]
         if c in adata.obs.columns), None)
    if cluster_col is None:
        raise ValueError(f"No cluster column found in {h5ad_path.name}")
    print(f"  Cluster column: '{cluster_col}'")

    adata.obs["celltypist_label"]    = pred_labels["predicted_labels"].values
    adata.obs["celltypist_majority"] = pred_labels.get(
        "majority_voting", pred_labels["predicted_labels"]).values
    adata.obs["cluster"]             = adata.obs[cluster_col].astype(int)

    # ── Load author labels ─────────────────────────────────────────────────
    lab_df = pd.read_csv(csv_path)
    lab_df["cluster"] = lab_df["cluster"].astype(int)

    # ── Per-cluster majority vote ──────────────────────────────────────────
    results = []
    for _, row in lab_df.iterrows():
        cl       = int(row["cluster"])
        n_cells  = int(row["n_cells"])
        auth_raw = str(row["author_label"])
        auth_std = normalise(auth_raw, author_vocab)

        mask  = adata.obs["cluster"] == cl
        if mask.sum() == 0:
            print(f"  WARNING: C{cl} — no cells found in h5ad")
            continue

        # Most common CellTypist label for this cluster
        ct_labels_cl = adata.obs.loc[mask, "celltypist_majority"]
        ct_majority  = Counter(ct_labels_cl).most_common(1)[0][0]
        ct_std       = normalise(ct_majority, VOCAB_MAP)
        ct_conf      = (ct_labels_cl == ct_majority).mean()

        correct = (ct_std.lower() == auth_std.lower())

        results.append({
            "cluster":           cl,
            "n_cells":           n_cells,
            "author_label":      auth_raw,
            "author_std":        auth_std,
            "celltypist_raw":    ct_majority,
            "celltypist_std":    ct_std,
            "celltypist_correct":correct,
            "celltypist_conf":   round(ct_conf, 3),
        })
        flag = "✓" if correct else "✗"
        print(f"  C{cl:>2}: {flag}  author='{auth_std}'  "
              f"celltypist='{ct_std}'  "
              f"(raw='{ct_majority}')  conf={ct_conf:.2f}")

    # ── Accuracy computation ───────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    n_correct    = res_df["celltypist_correct"].sum()
    n_total      = len(res_df)
    total_cells  = res_df["n_cells"].sum()

    # Macro accuracy = mean per-cluster
    macro_acc = round(100 * n_correct / n_total, 1)

    # Weighted accuracy = weighted by n_cells
    weighted_acc = round(
        100 * (res_df["celltypist_correct"] * res_df["n_cells"]).sum()
        / total_cells, 1)

    print(f"\n  {dataset_name} CellTypist accuracy:")
    print(f"    Correct clusters: {n_correct}/{n_total}")
    print(f"    Macro accuracy:   {macro_acc}%")
    print(f"    Weighted accuracy:{weighted_acc}%")

    return res_df, {
        "dataset":          dataset_name,
        "model":            model_name,
        "n_clusters":       n_total,
        "n_cells":          int(total_cells),
        "n_correct":        int(n_correct),
        "macro_acc":        macro_acc,
        "weighted_acc":     weighted_acc,
        "per_cluster":      results,
    }


def main():
    print("=" * 60)
    print("run_celltypist_comparison.py")
    print("CellTypist annotation: LUAD + HCC")
    print("=" * 60)

    # ── Install check ──────────────────────────────────────────────────────
    try:
        import celltypist
        import scanpy
        print(f"CellTypist version: {celltypist.__version__}")
        print(f"Scanpy   version:   {scanpy.__version__}")
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install with:")
        print("  pip install celltypist --break-system-packages")
        print("  pip install scanpy    --break-system-packages")
        return

    # ── LUAD ──────────────────────────────────────────────────────────────
    luad_df, luad_summary = run_celltypist(
        h5ad_path    = H5AD_L,
        csv_path     = CSV_L,
        author_vocab = AUTHOR_VOCAB_L,
        dataset_name = "LUAD",
        model_name   = "Immune_All_Low.pkl",
    )
    luad_df.to_csv(RES / "celltypist_luad_results.csv", index=False)
    (RES / "celltypist_luad_summary.json").write_text(
        json.dumps(luad_summary, indent=2))
    print(f"\nSaved → results/celltypist_luad_results.csv")
    print(f"Saved → results/celltypist_luad_summary.json")

    # ── HCC ───────────────────────────────────────────────────────────────
    hcc_df, hcc_summary = run_celltypist(
        h5ad_path    = H5AD_H,
        csv_path     = CSV_H,
        author_vocab = AUTHOR_VOCAB_H,
        dataset_name = "HCC",
        model_name   = "Immune_All_Low.pkl",
    )
    hcc_df.to_csv(HCC_DIR / "celltypist_hcc_results.csv", index=False)
    (HCC_DIR / "celltypist_hcc_summary.json").write_text(
        json.dumps(hcc_summary, indent=2))
    print(f"\nSaved → results/hcc/celltypist_hcc_results.csv")
    print(f"Saved → results/hcc/celltypist_hcc_summary.json")

    # ── Combined summary ──────────────────────────────────────────────────
    # This file feeds directly into draw_fig_accuracy_comparison.py
    combined = {
        "description": (
            "CellTypist accuracy on LUAD and HCC. "
            "Computed by run_celltypist_comparison.py. "
            "Feeds into draw_fig_accuracy_comparison.py and Table 2."
        ),
        "luad": {
            "weighted_acc": luad_summary["weighted_acc"],
            "macro_acc":    luad_summary["macro_acc"],
            "n_correct":    luad_summary["n_correct"],
            "n_clusters":   luad_summary["n_clusters"],
            "n_cells":      luad_summary["n_cells"],
        },
        "hcc": {
            "weighted_acc": hcc_summary["weighted_acc"],
            "macro_acc":    hcc_summary["macro_acc"],
            "n_correct":    hcc_summary["n_correct"],
            "n_clusters":   hcc_summary["n_clusters"],
            "n_cells":      hcc_summary["n_cells"],
        },
    }
    (RES / "celltypist_accuracy_summary.json").write_text(
        json.dumps(combined, indent=2))
    print(f"\nSaved → results/celltypist_accuracy_summary.json")

    # ── Print final comparison table ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'LUAD W':>8} {'LUAD M':>8} "
          f"{'HCC W':>8} {'HCC M':>8}")
    print("-" * 60)
    print(f"{'GRACE v2':<20} {'100.0%':>8} {'100.0%':>8} "
          f"{'93.3%':>8} {'92.0%':>8}")
    print(f"{'SingleR':<20} {'91.1%':>8} {'90.0%':>8} "
          f"{'80.4%':>8} {'68.0%':>8}")
    print(f"{'GPT-5.4 naive':<20} {'85.7%':>8} {'80.0%':>8} "
          f"{'43.9%':>8} {'40.0%':>8}")
    print(f"{'CellTypist':<20} "
          f"{luad_summary['weighted_acc']:>7.1f}% "
          f"{luad_summary['macro_acc']:>7.1f}% "
          f"{hcc_summary['weighted_acc']:>7.1f}% "
          f"{hcc_summary['macro_acc']:>7.1f}%")
    print("-" * 60)
    print("W = Weighted accuracy  M = Macro accuracy")
    print("HCC = zero-shot (no disease-specific configuration)")


if __name__ == "__main__":
    main()
