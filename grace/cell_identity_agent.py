"""
Cell Identity Agent — CellMarker 2.0 + PanglaoDB curated markers
=================================================================
Agent 4 of the expanded Version B framework.

Given a cluster's top DEG list:
  1. Queries CellMarker 2.0 database for matching cell types
  2. Falls back to curated PanglaoDB-equivalent marker dictionary
  3. Returns top cell type matches with confidence scores and evidence

Works for both LUAD (Lung tissue) and HCC (Liver tissue).
Results feed into the orchestrator alongside existing 3 agents.

Run standalone test:
    python3 cell_identity_agent.py
"""

import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
KB_DIR         = Path("data/knowledge_bases")
CELLMARKER_FILE= KB_DIR / "CellMarker2_Human.xlsx"
CACHE_DIR      = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Tissue aliases for CellMarker lookup ─────────────────────────────────────
TISSUE_ALIASES = {
    "lung":  ["Lung", "Lung cancer", "Undefined"],
    "liver": ["Liver", "Liver cancer", "Undefined"],
    "blood": ["Blood", "Bone marrow", "Undefined"],
    "any":   None,   # None = no tissue filter
}

# ── Curated PanglaoDB-equivalent marker dictionary ───────────────────────────
# Built from PanglaoDB, CellTypist, and HPCA canonical markers
# Covers all cell types in LUAD and HCC
CURATED_MARKERS = {
    # ── T / NK cells ──────────────────────────────────────────────────────────
    "T cell (generic)": {
        "markers": ["CD3D","CD3E","CD3G","CD247","TRAC","TRBC1","TRBC2"],
        "tissues": ["lung","liver","blood","any"],
    },
    "CD4+ T cell": {
        "markers": ["CD4","IL7R","TCF7","LEF1","CCR7","MAL","LDHB","CXCR4","IL32","MCL1","SRGN","ZFP36L2","BTG1"],
        "tissues": ["lung","liver","blood","any"],
    },
    "T cell (activated)": {
        "markers": ["CXCR4","IL32","MCL1","SRGN","ZFP36L2","BTG1","CREM","HLA-B","IL2","ICOS"],
        "tissues": ["lung","liver","blood","any"],
    },
    "CD8+ T cell (cytotoxic)": {
        "markers": ["CD8A","CD8B","GZMB","GZMK","GZMA","PRF1","NKG7","GNLY"],
        "tissues": ["lung","liver","blood","any"],
    },
    "CD8+ T cell (exhausted)": {
        "markers": ["PDCD1","TIGIT","HAVCR2","LAG3","CTLA4","TOX","ENTPD1"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Regulatory T cell (Treg)": {
        "markers": ["FOXP3","IL2RA","CTLA4","IKZF2","TIGIT","TNFRSF9"],
        "tissues": ["lung","liver","blood","any"],
    },
    "NK cell": {
        "markers": ["NCAM1","NKG7","GNLY","KLRB1","KLRC1","KLRD1","FCGR3A","XCL1","XCL2"],
        "tissues": ["lung","liver","blood","any"],
    },
    "NKT cell": {
        "markers": ["CD3D","NCAM1","NKG7","KLRB1","GNLY"],
        "tissues": ["lung","liver","blood","any"],
    },
    "γδ T cell": {
        "markers": ["TRGC1","TRGC2","TRDV1","TRDC","TRGV9"],
        "tissues": ["lung","liver","blood","any"],
    },

    # ── B cells / Plasma ──────────────────────────────────────────────────────
    "B cell (naive/mature)": {
        "markers": ["CD19","MS4A1","CD79A","CD79B","IGHM","IGHD","TCL1A","FCER2"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Plasma cell": {
        "markers": ["IGHG1","IGHG3","IGKC","MZB1","SDC1","PRDM1","XBP1","JCHAIN"],
        "tissues": ["lung","liver","blood","any"],
    },

    # ── Myeloid ───────────────────────────────────────────────────────────────
    "Tumour-associated macrophage (TAM)": {
        "markers": ["CD68","CD163","MRC1","MSR1","TREM2","GPNMB","APOE","C1QA","C1QB","C1QC"],
        "tissues": ["lung","liver","any"],
    },
    "Monocyte (classical)": {
        "markers": ["CD14","LYZ","S100A8","S100A9","VCAN","FCN1","SERPINA1"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Monocyte (non-classical)": {
        "markers": ["FCGR3A","LST1","MS4A7","LILRB2","CX3CR1","CDKN1C"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Inflammatory monocyte": {
        "markers": ["S100A8","S100A9","IL1B","CXCL8","TNF","CCL2","CCL3","CCL4"],
        "tissues": ["lung","liver","any"],
    },
    "Dendritic cell (cDC1)": {
        "markers": ["CLEC9A","XCR1","CADM1","WDFY4","IRF8","BATF3"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Dendritic cell (cDC2)": {
        "markers": ["CD1C","FCER1A","CLEC10A","SIRPA","CD1A","CD1B"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Plasmacytoid dendritic cell (pDC)": {
        "markers": ["LILRA4","CLEC4C","IL3RA","TCF4","PTCRA","IRF7"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Kupffer cell": {
        "markers": ["VSIG4","CD163","CLEC4F","TIMD4","MARCO","CD5L","HMOX1"],
        "tissues": ["liver","any"],
    },
    "Mast cell": {
        "markers": ["KIT","MS4A2","CPA3","TPSAB1","TPSB2","HDC","HPGDS","CD88"],
        "tissues": ["lung","liver","blood","any"],
    },
    "Neutrophil": {
        "markers": ["CXCR1","CXCR2","FCGR3B","S100A8","S100A9","ELANE","MPO"],
        "tissues": ["lung","liver","blood","any"],
    },

    # ── Stromal ───────────────────────────────────────────────────────────────
    "Cancer-associated fibroblast (CAF)": {
        "markers": ["FAP","ACTA2","COL1A1","COL1A2","COL3A1","PDGFRA","S100A4",
                    "POSTN","MFAP4","BGN","EMILIN1","DCN"],
        "tissues": ["lung","liver","any"],
    },
    "Endothelial cell": {
        "markers": ["PECAM1","CDH5","VWF","ENG","CLEC14A","EMCN","RAMP2","CLDN5"],
        "tissues": ["lung","liver","any"],
    },
    "Lymphatic endothelial cell": {
        "markers": ["LYVE1","PROX1","PDPN","MMRN1","CCL21"],
        "tissues": ["lung","liver","any"],
    },
    "Pericyte": {
        "markers": ["RGS5","ACTA2","PDGFRB","MCAM","NOTCH3","CSPG4"],
        "tissues": ["lung","liver","any"],
    },

    # ── Lung-specific epithelial ──────────────────────────────────────────────
    "Alveolar type I (AT1) cell": {
        "markers": ["AGER","PDPN","CAV1","CAV2","HOPX","AKAP5"],
        "tissues": ["lung","any"],
    },
    "Alveolar type II (AT2) cell": {
        "markers": ["SFTPC","SFTPB","SFTPD","SFTPA1","SFTPA2","ABCA3","NKX2-1"],
        "tissues": ["lung","any"],
    },
    "Ciliated epithelial cell": {
        "markers": ["FOXJ1","DYDC1","DYDC2","DNAI1","DNAI2","SNTN","CCDC39"],
        "tissues": ["lung","any"],
    },
    "Club cell": {
        "markers": ["SCGB1A1","SCGB3A2","CYP2B6","MSLN","SSNA1"],
        "tissues": ["lung","any"],
    },
    "LUAD tumour cell (mucinous/secretory)": {
        "markers": ["NKX2-1","EPCAM","KRAS","EGFR","KRT7","KRT17","KRT8","KRT18","MUC4","MUC16","CLDN3","CLDN4","S100A2","CRABP2","FAM83A",
                    "KRT19","MUC1","NAPSA","SFTA3","CLDN18"],
        "tissues": ["lung","any"],
    },
    "Malignant epithelial cell (LUAD, KRAS-high)": {
        "markers": ["KRAS","DUSP6","ETV4","ETV5","SPRY2","SPRED1","SPRED2"],
        "tissues": ["lung","any"],
    },

    # ── Liver-specific ────────────────────────────────────────────────────────
    "Hepatocyte": {
        "markers": ["ALB","APOB","APOE","TF","HP","FGB","FGG","CYP3A4",
                    "CYP2C9","CYP2D6","CYP1A2","HNF4A","ARG1","ASS1","OTC",
                    "GC","RBP4","APOH","APOA1","APOA2","APOC1","APOC3",
                    "TTR","FN1","FABP1","GLUL","UGT2B4","RARRES2",
                    "A1BG","CTTN","TFPI","MPC2","UQCRFS1","CFH"],
        "tissues": ["liver","any"],
    },
    "HCC tumour cell": {
        "markers": ["AFP","GPC3","EPCAM","KRT19","CD44","PROM1","ALDH1A1",
                    "MYC","CCND1","CDK4","CDK6","RB1","TP53","CTNNB1"],
        "tissues": ["liver","any"],
    },
    "Hepatic stellate cell (HSC/CAF)": {
        "markers": ["ACTA2","COL1A1","COL1A2","LRAT","VIM","DES","PDGFRB",
                    "BAMBI","TGFB1","TIMP1","TIMP2"],
        "tissues": ["liver","any"],
    },
    "Cholangiocyte (biliary cell)": {
        "markers": ["KRT7","KRT19","CFTR","AQP1","EPCAM","SOX9","HNF1B"],
        "tissues": ["liver","any"],
    },

    # ── Proliferating (any lineage) ───────────────────────────────────────────
    "Proliferating cell (cycling)": {
        "markers": ["MKI67","TOP2A","TYMS","PCNA","CDK1","CCNB1","CCNA2",
                    "BIRC5","CENPE","TPX2","BUB1","BUB1B","CDC20","PLK1"],
        "tissues": ["lung","liver","blood","any"],
    },
}


# ── CellMarker loader ─────────────────────────────────────────────────────────

def load_cellmarker(tissue: str = "lung") -> pd.DataFrame:
    """
    Load CellMarker 2.0 and filter to relevant tissue + human species.
    Returns DataFrame with cell_name and Symbol columns.
    """
    cache_path = CACHE_DIR / f"cellmarker_{tissue}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if not CELLMARKER_FILE.exists():
        print(f"  WARNING: CellMarker file not found at {CELLMARKER_FILE}")
        return pd.DataFrame(columns=["cell_name","Symbol","tissue_class","cancer_type"])

    print(f"  Loading CellMarker 2.0 for tissue={tissue!r}...")
    df = pd.read_excel(CELLMARKER_FILE)

    # Filter to human
    df = df[df["species"].str.lower() == "human"].copy()

    # Tissue filter
    aliases = TISSUE_ALIASES.get(tissue, TISSUE_ALIASES["any"])
    if aliases is not None:
        df = df[df["tissue_class"].isin(aliases)].copy()

    # Keep relevant columns
    df = df[["cell_name","Symbol","tissue_class",
             "cancer_type","cell_type","PMID"]].copy()
    df = df.dropna(subset=["Symbol","cell_name"])
    df["Symbol"] = df["Symbol"].str.strip().str.upper()
    df["cell_name"] = df["cell_name"].str.strip()

    df.to_csv(cache_path, index=False)
    print(f"    CellMarker entries for {tissue}: {len(df):,}")
    return df


# ── Core matching function ────────────────────────────────────────────────────

def query_cellmarker(
    deg_list: list,
    tissue: str = "lung",
    top_n: int = 5,
) -> list:
    """
    Query CellMarker 2.0 database for DEG list.
    Returns list of dicts with cell_name, n_markers_found, confidence, genes.
    """
    deg_set = {g.strip().upper() for g in deg_list}
    cm = load_cellmarker(tissue)

    if cm.empty:
        return []

    # For each cell_name count how many DEGs appear as markers
    matches = defaultdict(lambda: {"genes": [], "pmids": set()})
    for _, row in cm.iterrows():
        gene = str(row["Symbol"]).upper()
        if gene in deg_set:
            cell = row["cell_name"]
            matches[cell]["genes"].append(gene)
            if pd.notna(row["PMID"]):
                matches[cell]["pmids"].add(str(int(row["PMID"])))

    if not matches:
        return []

    # Compute confidence = Jaccard-like score
    # n_matched / (n_deg + n_markers_for_cell - n_matched)
    # Get total markers per cell type from the database
    cell_marker_counts = cm.groupby("cell_name")["Symbol"].nunique().to_dict()

    results = []
    for cell_name, info in matches.items():
        n_matched  = len(set(info["genes"]))
        n_markers  = cell_marker_counts.get(cell_name, 1)
        n_degs     = len(deg_set)
        jaccard    = n_matched / (n_degs + n_markers - n_matched)
        recall     = n_matched / max(n_markers, 1)
        precision  = n_matched / max(n_degs, 1)
        confidence = round((jaccard + recall + precision) / 3, 3)

        results.append({
            "cell_name":     cell_name,
            "n_matched":     n_matched,
            "confidence":    confidence,
            "matched_genes": sorted(set(info["genes"])),
            "pmids":         sorted(info["pmids"])[:3],
            "source":        "CellMarker2",
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_n]


def query_curated_markers(
    deg_list: list,
    tissue: str = "lung",
    top_n: int = 5,
) -> list:
    """
    Query curated PanglaoDB-equivalent marker dictionary.
    """
    deg_set = {g.strip().upper() for g in deg_list}
    results = []

    for cell_type, info in CURATED_MARKERS.items():
        # Check tissue relevance
        cell_tissues = info.get("tissues", ["any"])
        if tissue not in cell_tissues and "any" not in cell_tissues:
            continue

        markers = {m.upper() for m in info["markers"]}
        matched = deg_set & markers
        if not matched:
            continue

        n_matched   = len(matched)
        n_markers   = len(markers)
        n_degs      = len(deg_set)
        precision   = n_matched / max(n_degs, 1)
        recall      = n_matched / max(n_markers, 1)
        f1          = 2 * precision * recall / max(precision + recall, 1e-9)
        confidence  = round(f1, 3)

        results.append({
            "cell_name":     cell_type,
            "n_matched":     n_matched,
            "confidence":    confidence,
            "matched_genes": sorted(matched),
            "pmids":         [],
            "source":        "CuratedMarkers",
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_n]


# ── Main agent function ───────────────────────────────────────────────────────

def run_cell_identity_agent(
    cluster_id: str,
    deg_list: list,
    tissue: str = "lung",
    top_n: int = 5,
) -> dict:
    """
    Full Cell Identity Agent:
    1. Query CellMarker 2.0
    2. Query curated markers
    3. Merge and rank results
    4. Return structured output for orchestrator
    """
    cache_key = f"cell_identity_{tissue}_{cluster_id}_{'_'.join(sorted(deg_list[:10]))}"
    cache_file = CACHE_DIR / f"{abs(hash(cache_key)) % 10**8}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Query both sources
    cm_results      = query_cellmarker(deg_list, tissue, top_n=top_n * 2)
    curated_results = query_curated_markers(deg_list, tissue, top_n=top_n * 2)

    # Merge — prefer CellMarker entries, supplement with curated
    seen_types = {r["cell_name"] for r in cm_results}
    merged = cm_results.copy()
    for r in curated_results:
        if r["cell_name"] not in seen_types:
            merged.append(r)
            seen_types.add(r["cell_name"])

    merged.sort(key=lambda x: x["confidence"], reverse=True)
    top_results = merged[:top_n]

    # Compute overall agent confidence
    if top_results:
        top_conf    = top_results[0]["confidence"]
        top_matched = top_results[0]["n_matched"]
        # High confidence if top match has ≥3 genes and confidence ≥ 0.15
        if top_matched >= 3 and top_conf >= 0.15:
            agent_confidence = min(0.9, top_conf * 3)
        elif top_matched >= 2 and top_conf >= 0.08:
            agent_confidence = min(0.6, top_conf * 2.5)
        elif top_matched >= 1:
            agent_confidence = min(0.35, top_conf * 2)
        else:
            agent_confidence = 0.0
    else:
        agent_confidence = 0.0
        top_results      = []

    # Best cell type prediction
    best_cell_type = top_results[0]["cell_name"] if top_results else "Unknown"
    best_genes     = top_results[0]["matched_genes"] if top_results else []

    # Uncertainty flag if top confidence is low
    uncertain = agent_confidence < 0.30
    uncertainty_claims = []
    if uncertain:
        uncertainty_claims.append(
            f"Cell identity is uncertain (agent_confidence={agent_confidence:.2f}); "
            f"top match {best_cell_type!r} supported by only "
            f"{len(best_genes)} marker gene(s)"
        )
    if len(top_results) >= 2:
        gap = top_results[0]["confidence"] - top_results[1]["confidence"]
        if gap < 0.05:
            uncertainty_claims.append(
                f"Ambiguous: top two matches ({top_results[0]['cell_name']!r} vs "
                f"{top_results[1]['cell_name']!r}) have similar confidence scores "
                f"(gap={gap:.3f})"
            )

    output = {
        "cluster":            cluster_id,
        "tissue":             tissue,
        "deg_list":           deg_list[:20],
        "top_matches":        top_results,
        "best_cell_type":     best_cell_type,
        "best_matched_genes": best_genes,
        "agent_confidence":   round(agent_confidence, 3),
        "uncertain":          uncertain,
        "uncertainty_claims": uncertainty_claims,
        "n_degs_queried":     len(deg_list),
    }

    cache_file.write_text(json.dumps(output, indent=2))
    return output


# ── Batch runner for all clusters ─────────────────────────────────────────────

def run_all_clusters(
    degs_dir: Path,
    tissue: str = "lung",
    dataset_label: str = "luad",
) -> list:
    """
    Run Cell Identity Agent on all clusters in a DEG directory.
    """
    results = []
    deg_files = sorted(degs_dir.glob("cluster_*_degs.csv"),
                       key=lambda f: int(f.stem.split("_")[1]))

    if not deg_files:
        print(f"  No DEG files found in {degs_dir}")
        return []

    print(f"  Running Cell Identity Agent on {len(deg_files)} clusters "
          f"(tissue={tissue!r}, dataset={dataset_label!r})...")

    for f in deg_files:
        cl  = f.stem.split("_")[1]
        df  = pd.read_csv(f)
        if df.empty or "names" not in df.columns:
            continue
        degs = df["names"].head(20).tolist()

        result = run_cell_identity_agent(cl, degs, tissue=tissue)
        results.append(result)

        best  = result["best_cell_type"]
        conf  = result["agent_confidence"]
        genes = ", ".join(result["best_matched_genes"][:4])
        unc   = " [UNCERTAIN]" if result["uncertain"] else ""
        print(f"    Cluster {cl:>2}: {best[:40]:<40} conf={conf:.2f}  "
              f"genes=[{genes}]{unc}")

    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Cell Identity Agent — standalone test")
    print("=" * 60)

    # Test on LUAD clusters
    luad_degs_dir = Path("results/degs")
    hcc_degs_dir  = Path("results/hcc/degs")

    if luad_degs_dir.exists():
        print("\n── LUAD (lung tissue) ──")
        luad_results = run_all_clusters(luad_degs_dir, tissue="lung",
                                        dataset_label="luad")
        out = Path("results/cell_identity_luad.json")
        out.write_text(json.dumps(luad_results, indent=2))
        print(f"\n  Saved → {out}")
        print(f"  Mean agent confidence: "
              f"{sum(r['agent_confidence'] for r in luad_results)/max(len(luad_results),1):.3f}")
        n_uncertain = sum(1 for r in luad_results if r["uncertain"])
        print(f"  Uncertain clusters: {n_uncertain}/{len(luad_results)}")
    else:
        print("  LUAD DEG directory not found — skipping")

    if hcc_degs_dir.exists():
        print("\n── HCC (liver tissue) ──")
        hcc_results = run_all_clusters(hcc_degs_dir, tissue="liver",
                                       dataset_label="hcc")
        out = Path("results/hcc/cell_identity_hcc.json")
        out.write_text(json.dumps(hcc_results, indent=2))
        print(f"\n  Saved → {out}")
        print(f"  Mean agent confidence: "
              f"{sum(r['agent_confidence'] for r in hcc_results)/max(len(hcc_results),1):.3f}")
        n_uncertain = sum(1 for r in hcc_results if r["uncertain"])
        print(f"  Uncertain clusters: {n_uncertain}/{len(hcc_results)}")
    else:
        print("  HCC DEG directory not found — run hcc_preprocess.py first")

    # Quick demo with known markers
    print("\n── Quick demo: known TAM markers ──")
    tam_degs = ["CD68","CD163","TREM2","GPNMB","C1QA","C1QB","APOE","FCGR3A",
                "MS4A6A","MARCO","SPP1","FN1","MRC1","MSR1"]
    result = run_cell_identity_agent("demo", tam_degs, tissue="lung")
    print(f"  Input genes: {tam_degs[:6]}...")
    print(f"  Best match: {result['best_cell_type']}")
    print(f"  Confidence: {result['agent_confidence']:.3f}")
    print(f"  Matched:    {result['best_matched_genes']}")
    print()
    print("  Top 3 matches:")
    for m in result["top_matches"][:3]:
        print(f"    {m['cell_name'][:45]:<45}  "
              f"conf={m['confidence']:.3f}  "
              f"genes={m['matched_genes'][:4]}")
