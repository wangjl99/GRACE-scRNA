"""
Agent 5 — Regulatory Agent (DoRothEA / CollecTRI)
===================================================
Given a cluster's DEG list and log-fold change values, computes
transcription factor (TF) activity scores using the DoRothEA regulon
database via the decoupler-py framework.

For each cluster returns:
  - Top 3-5 active TFs with activity scores and confidence
  - Supporting target genes found in DEG list per TF
  - Regulatory layer evidence for the orchestrator
  - c_regulatory confidence score ∈ [0,1]

Works on both LUAD (lung) and HCC (liver) datasets.

Install:
    pip install decoupler --break-system-packages

Run standalone test:
    python3 regulatory_agent.py
"""

import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── DoRothEA confidence levels ────────────────────────────────────────────────
# A = highest confidence (literature-curated)
# B = high confidence
# C = medium (include for broader coverage)
# D = low (exclude by default)
DOROTHEA_LEVELS = ["A", "B", "C"]

# ── Minimum thresholds ────────────────────────────────────────────────────────
MIN_TARGET_GENES   = 3    # minimum DEG targets per TF to report
MIN_TF_ACTIVITY    = 0.15  # minimum normalised activity score
TOP_N_TFS          = 5    # number of top TFs to return

# ── Known cancer-relevant TFs for annotation ──────────────────────────────────
CANCER_TFS = {
    # LUAD-relevant
    "KRAS":   "KRAS oncogene (RAS/MAPK signalling)",
    "EGFR":   "EGFR (receptor tyrosine kinase signalling)",
    "TP53":   "TP53 tumour suppressor",
    "NKX2-1": "NKX2-1 (lung lineage TF, LUAD marker)",
    "FOXA2":  "FOXA2 (lung epithelial differentiation)",
    # Immune
    "NFKB1":  "NF-κB1 (inflammatory signalling)",
    "RELA":   "RelA/NF-κB (TNF/inflammatory response)",
    "IRF4":   "IRF4 (B cell and myeloid differentiation)",
    "IRF8":   "IRF8 (dendritic cell specification)",
    "TBX21":  "T-bet (Th1/NK cell identity)",
    "GATA3":  "GATA3 (Th2, mast cell identity)",
    "FOXP3":  "FOXP3 (regulatory T cell master TF)",
    "BCL6":   "BCL6 (B cell germinal centre)",
    "PAX5":   "PAX5 (B cell identity)",
    "SPI1":   "PU.1/SPI1 (myeloid/macrophage identity)",
    "CEBPA":  "C/EBPα (myeloid differentiation)",
    "CEBPB":  "C/EBPβ (macrophage/monocyte activation)",
    "MYC":    "MYC (cell cycle, proliferation)",
    "E2F1":   "E2F1 (G1/S transition, proliferation)",
    "E2F3":   "E2F3 (cell cycle progression)",
    # Stromal
    "TWIST1": "TWIST1 (EMT master regulator)",
    "SNAI1":  "Snail (EMT, fibroblast)",
    "ZEB1":   "ZEB1 (EMT, CAF identity)",
    "ACTA2":  "ACTA2/αSMA (myofibroblast)",
    # HCC-relevant
    "HNF4A":  "HNF4A (hepatocyte master TF)",
    "FOXA1":  "FOXA1 (hepatocyte differentiation)",
    "FOXA2":  "FOXA2 (hepatocyte identity)",
    "CTNNB1": "β-catenin/CTNNB1 (Wnt signalling, HCC)",
    "MYC":    "MYC (proliferation, HCC oncogene)",
    "STAT3":  "STAT3 (JAK-STAT, HCC progression)",
    "TP53":   "TP53 (tumour suppressor, HCC)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load DoRothEA / CollecTRI regulon
# ─────────────────────────────────────────────────────────────────────────────

def load_regulon(levels: list = DOROTHEA_LEVELS) -> pd.DataFrame:
    """
    Load DoRothEA human regulon via decoupler-py.
    Falls back to CollecTRI if DoRothEA unavailable.
    Returns DataFrame with columns: source (TF), target (gene), weight, confidence.
    """
    cache_path = CACHE_DIR / f"dorothea_regulon_{''.join(levels)}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  Loaded DoRothEA regulon from cache: {len(df):,} interactions")
        return df

    try:
        import decoupler as dc
        print("  Loading DoRothEA regulon via decoupler-py...")
        try:
            # Try DoRothEA first
            net = dc.get_dorothea(organism="human", levels=levels)
            source = "DoRothEA"
        except Exception:
            # Fallback to CollecTRI (successor to DoRothEA)
            net = dc.get_collectri(organism="human", split_complexes=False)
            source = "CollecTRI"
        print(f"  Loaded {source}: {len(net):,} TF-target interactions")
        net.to_csv(cache_path, index=False)
        return net
    except ImportError:
        print("  WARNING: decoupler not installed. Using curated fallback regulon.")
        return _build_curated_fallback()
    except Exception as e:
        print(f"  WARNING: regulon load failed ({e}). Using curated fallback.")
        return _build_curated_fallback()


def _build_curated_fallback() -> pd.DataFrame:
    """
    Curated minimal regulon for the most important cancer/immune TFs.
    Used when decoupler is unavailable.
    Based on TRRUST v2, DoRothEA A-confidence, and published literature.
    """
    regulon = [
        # NF-κB / inflammatory
        ("RELA",  "IL6",     1.0, "A"), ("RELA",  "TNF",     1.0, "A"),
        ("RELA",  "CXCL8",   1.0, "A"), ("RELA",  "ICAM1",   1.0, "A"),
        ("RELA",  "CCL2",    1.0, "A"), ("RELA",  "NFKBIA",  1.0, "A"),
        ("NFKB1", "IL1B",    1.0, "A"), ("NFKB1", "IL6",     1.0, "A"),
        ("NFKB1", "TNF",     1.0, "A"), ("NFKB1", "CXCL10",  1.0, "A"),
        # MYC / proliferation
        ("MYC",   "CDK4",    1.0, "A"), ("MYC",   "CCND1",   1.0, "A"),
        ("MYC",   "PCNA",    1.0, "A"), ("MYC",   "TOP2A",   1.0, "A"),
        ("MYC",   "MKI67",   1.0, "A"), ("MYC",   "E2F1",    1.0, "A"),
        # E2F / cell cycle
        ("E2F1",  "CCNE1",   1.0, "A"), ("E2F1",  "PCNA",    1.0, "A"),
        ("E2F1",  "TYMS",    1.0, "A"), ("E2F1",  "TOP2A",   1.0, "A"),
        ("E2F3",  "CDK1",    1.0, "A"), ("E2F3",  "CCNA2",   1.0, "A"),
        # TP53 / apoptosis
        ("TP53",  "CDKN1A",  1.0, "A"), ("TP53",  "MDM2",    1.0, "A"),
        ("TP53",  "BAX",     1.0, "A"), ("TP53",  "BBC3",    1.0, "A"),
        # HNF4A / hepatocyte
        ("HNF4A", "ALB",     1.0, "A"), ("HNF4A", "APOB",    1.0, "A"),
        ("HNF4A", "CYP3A4",  1.0, "A"), ("HNF4A", "CYP2C9",  1.0, "A"),
        ("HNF4A", "APOE",    1.0, "A"), ("HNF4A", "TF",      1.0, "A"),
        # FOXA1/2 / hepatocyte + lung
        ("FOXA1", "ALB",     1.0, "A"), ("FOXA1", "AFP",     1.0, "A"),
        ("FOXA2", "SFTPC",   1.0, "A"), ("FOXA2", "NKX2-1",  1.0, "A"),
        # SPI1 / PU.1 / myeloid
        ("SPI1",  "CD68",    1.0, "A"), ("SPI1",  "CSF1R",   1.0, "A"),
        ("SPI1",  "FCGR2A",  1.0, "A"), ("SPI1",  "CD14",    1.0, "A"),
        ("SPI1",  "LYZ",     1.0, "A"), ("SPI1",  "TYROBP",  1.0, "A"),
        # CEBPB / macrophage
        ("CEBPB", "IL6",     1.0, "A"), ("CEBPB", "IL1B",    1.0, "A"),
        ("CEBPB", "CD14",    1.0, "A"), ("CEBPB", "LYZ",     1.0, "A"),
        # IRF4/8 / DC / macrophage
        ("IRF8",  "CLEC9A",  1.0, "A"), ("IRF8",  "XCR1",    1.0, "A"),
        ("IRF4",  "CD1C",    1.0, "A"), ("IRF4",  "FCER1A",  1.0, "A"),
        # TBX21 / NK / T cell
        ("TBX21", "IFNG",    1.0, "A"), ("TBX21", "GZMB",    1.0, "A"),
        ("TBX21", "PRF1",    1.0, "A"), ("TBX21", "KLRB1",   1.0, "A"),
        # FOXP3 / Treg
        ("FOXP3", "IL2RA",   1.0, "A"), ("FOXP3", "CTLA4",   1.0, "A"),
        ("FOXP3", "TIGIT",   1.0, "A"), ("FOXP3", "IKZF2",   1.0, "A"),
        # GATA3 / Th2 / mast
        ("GATA3", "IL4",     1.0, "A"), ("GATA3", "IL13",    1.0, "A"),
        ("GATA3", "MS4A2",   1.0, "A"), ("GATA3", "CPA3",    1.0, "A"),
        # BCL6 / B cell
        ("BCL6",  "AICDA",   1.0, "A"), ("BCL6",  "PRDM1",   1.0, "A"),
        # T cell TFs
        ("TCF7",  "CD3D",    1.0, "A"), ("TCF7",  "CD3E",    1.0, "A"),
        ("TCF7",  "IL7R",    1.0, "A"), ("TCF7",  "CCR7",    1.0, "A"),
        ("EOMES", "GZMB",    1.0, "A"), ("EOMES", "PRF1",    1.0, "A"),
        ("EOMES", "CD8A",    1.0, "A"), ("EOMES", "TBX21",   1.0, "A"),
        ("IKZF2", "FOXP3",   1.0, "A"), ("IKZF2", "IL2RA",   1.0, "A"),
        # NK cell
        ("TBX21", "NKG7",    1.0, "A"), ("TBX21", "KLRD1",   1.0, "A"),
        ("TBX21", "GNLY",    1.0, "A"), ("TBX21", "NCR1",    1.0, "A"),
        # Mast cell
        ("MITF",  "KIT",     1.0, "A"), ("MITF",  "CPA3",    1.0, "A"),
        ("MITF",  "TPSAB1",  1.0, "A"), ("MITF",  "MS4A2",   1.0, "A"),
        # Endothelial
        ("ERG",   "CDH5",    1.0, "A"), ("ERG",   "PECAM1",  1.0, "A"),
        ("ERG",   "VWF",     1.0, "A"), ("ERG",   "ENG",     1.0, "A"),
        # Fibroblast / CAF
        ("TWIST1","COL1A1",  1.0, "A"), ("TWIST1","FAP",     1.0, "A"),
        ("ZEB1",  "ACTA2",   1.0, "A"), ("ZEB1",  "POSTN",   1.0, "A"),
        # Proliferating
        ("MYC",   "MKI67",   1.0, "A"), ("MYC",   "BIRC5",   1.0, "A"),
        ("E2F1",  "CDK1",    1.0, "A"), ("E2F1",  "CCNB1",   1.0, "A"),
        # TWIST1 / ZEB1 / EMT
        ("TWIST1","ACTA2",   1.0, "A"), ("TWIST1","VIM",     1.0, "A"),
        ("ZEB1",  "CDH2",    1.0, "A"), ("ZEB1",  "VIM",     1.0, "A"),
        # STAT3 / HCC
        ("STAT3", "MYC",     1.0, "A"), ("STAT3", "CCND1",   1.0, "A"),
        ("STAT3", "BCL2",    1.0, "A"), ("STAT3", "VEGFA",   1.0, "A"),
        # CTNNB1 / Wnt
        ("CTNNB1","MYC",     1.0, "A"), ("CTNNB1","CCND1",   1.0, "A"),
        ("CTNNB1","AXIN2",   1.0, "A"), ("CTNNB1","LEF1",    1.0, "A"),
    ]
    df = pd.DataFrame(regulon, columns=["source","target","weight","confidence"])
    print(f"  Using curated fallback regulon: {len(df)} interactions")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TF activity scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_tf_activity(
    deg_df:    pd.DataFrame,
    regulon:   pd.DataFrame,
    top_n:     int = TOP_N_TFS,
    min_targets: int = MIN_TARGET_GENES,
) -> list:
    """
    Score TF activity from DEG list using weighted target gene overlap.

    Method: for each TF in the regulon, compute a weighted activity score
    based on how many of its target genes appear in the DEG list and their
    log-fold change direction (consistent with TF regulatory sign).

    Args:
        deg_df:   DataFrame with columns 'names' (gene), 'logfoldchanges'
        regulon:  DataFrame with columns 'source' (TF), 'target', 'weight'
        top_n:    Number of top TFs to return
        min_targets: Minimum target genes required

    Returns:
        List of dicts sorted by activity score descending.
    """
    if deg_df.empty or regulon.empty:
        return []

    # Normalise gene names
    deg_df = deg_df.copy()
    deg_df["names"] = deg_df["names"].str.strip().str.upper()

    deg_genes = set(deg_df["names"].tolist())
    lfc_map   = dict(zip(deg_df["names"],
                         deg_df.get("logfoldchanges",
                                    pd.Series([1.0]*len(deg_df)))))

    results = []
    for tf, group in regulon.groupby("source"):
        tf_upper = str(tf).upper()
        targets  = group["target"].str.upper().tolist()
        weights  = group["weight"].tolist()

        # Find overlapping target genes in DEG list
        overlapping = [(t, w) for t, w in zip(targets, weights)
                       if t in deg_genes]

        if len(overlapping) < min_targets:
            continue

        # Weighted activity score: sum(weight × sign(lfc)) / sqrt(n_targets)
        score = 0.0
        matched_genes = []
        for gene, weight in overlapping:
            lfc   = lfc_map.get(gene, 1.0)
            # Positive weight = TF activates gene: score increases if gene upregulated
            score += weight * np.sign(lfc)
            matched_genes.append(gene)

        # Normalise by sqrt(n_total_targets) to penalise TFs with tiny regulons
        n_total = max(len(targets), 1)
        activity = score / np.sqrt(n_total)

        # Precision: fraction of TF's regulon found in DEGs (quality signal)
        precision = len(overlapping) / max(n_total, 1)
        recall    = len(overlapping) / max(len(deg_genes), 1)

        if abs(activity) < MIN_TF_ACTIVITY:
            continue

        results.append({
            "tf":              tf_upper,
            "activity_score":  round(float(activity), 4),
            "n_targets_found": len(overlapping),
            "n_targets_total": n_total,
            "precision":       round(precision, 3),
            "recall":          round(recall, 3),
            "matched_genes":   sorted(matched_genes)[:8],
            "direction":       "activating" if activity > 0 else "repressing",
            "cancer_relevant": tf_upper in CANCER_TFS,
            "annotation":      CANCER_TFS.get(tf_upper, ""),
        })

    # Sort by absolute activity, cancer-relevant first
    results.sort(key=lambda x: (
        -int(x["cancer_relevant"]),
        -abs(x["activity_score"])
    ))
    return results[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_regulatory_confidence(tf_results: list) -> float:
    """
    Compute c_regulatory ∈ [0,1] based on TF evidence quality.

    High confidence: multiple cancer-relevant TFs found with strong activity.
    Low confidence: few TFs, low activity, no cancer-relevant TFs.
    """
    if not tf_results:
        return 0.0

    n_tfs           = len(tf_results)
    n_cancer        = sum(1 for r in tf_results if r["cancer_relevant"])
    top_activity    = abs(tf_results[0]["activity_score"])
    top_n_targets   = tf_results[0]["n_targets_found"]
    mean_precision  = np.mean([r["precision"] for r in tf_results])

    # Component scores
    score_n_tfs     = min(n_tfs / TOP_N_TFS, 1.0)           # 0-1
    score_cancer    = min(n_cancer / 2, 1.0)                  # 0-1 (cap at 2)
    score_activity  = min(top_activity / 2.0, 1.0)            # 0-1
    score_targets   = min(top_n_targets / 5, 1.0)             # 0-1
    score_precision = mean_precision                           # 0-1

    # Weighted combination
    conf = (
        0.20 * score_n_tfs +
        0.25 * score_cancer +
        0.25 * score_activity +
        0.15 * score_targets +
        0.15 * score_precision
    )
    return round(min(conf, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_regulatory_uncertainty(
    tf_results: list,
    confidence: float,
) -> list:
    """
    Generate uncertainty claims for the orchestrator.
    """
    claims = []

    if not tf_results:
        claims.append(
            "No significant TF activity detected — regulatory layer grounding is absent. "
            "Transcription factor-level interpretation should be treated as provisional."
        )
        return claims

    if confidence < 0.3:
        claims.append(
            f"Regulatory agent confidence is low (c_regulatory = {confidence:.2f}): "
            f"only {len(tf_results)} TF(s) detected with limited target gene overlap. "
            "TF activity annotations should be treated as preliminary."
        )

    if not any(r["cancer_relevant"] for r in tf_results):
        claims.append(
            "No cancer-relevant TFs identified in the regulatory analysis. "
            "The regulatory programme of this cluster may reflect normal tissue "
            "or non-canonical cancer biology not represented in the curated TF database."
        )

    # Check for conflicting regulatory signals
    activating  = [r for r in tf_results if r["direction"] == "activating"]
    repressing  = [r for r in tf_results if r["direction"] == "repressing"]
    if activating and repressing:
        if len(activating) > 0 and len(repressing) > 0:
            claims.append(
                f"Mixed regulatory signals detected: "
                f"{activating[0]['tf']} (activating) and "
                f"{repressing[0]['tf']} (repressing) are both active, "
                "which may reflect a transitional or heterogeneous cell state."
            )

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# Main agent function
# ─────────────────────────────────────────────────────────────────────────────

_REGULON_CACHE = None  # module-level cache to avoid reloading per cluster

def run_regulatory_agent(
    cluster_id:  str,
    deg_df:      pd.DataFrame,
    tissue:      str = "lung",
    top_n:       int = TOP_N_TFS,
) -> dict:
    """
    Full Regulatory Agent.
    Computes TF activity from DEG list using DoRothEA/CollecTRI.

    Args:
        cluster_id: Leiden cluster identifier (string)
        deg_df:     DataFrame with at minimum 'names' column (gene symbols)
                    Optional: 'logfoldchanges', 'scores' columns
        tissue:     'lung' or 'liver' (for cancer-relevant TF annotation)
        top_n:      Number of top TFs to return

    Returns:
        Dict with TF activity results, confidence, uncertainty claims.
    """
    global _REGULON_CACHE

    # Build cache key
    gene_key = "_".join(sorted(deg_df["names"].head(10).str.upper().tolist()))
    cache_file = CACHE_DIR / f"regulatory_{tissue}_{cluster_id}_{abs(hash(gene_key))%10**8}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    # Load regulon once per session
    if _REGULON_CACHE is None:
        _REGULON_CACHE = load_regulon(DOROTHEA_LEVELS)

    # Score TF activity
    tf_results = score_tf_activity(deg_df, _REGULON_CACHE, top_n=top_n)

    # Confidence
    confidence = compute_regulatory_confidence(tf_results)

    # Uncertainty
    uncertainty_claims = detect_regulatory_uncertainty(tf_results, confidence)

    # Format top TFs for narrator
    top_tfs_summary = []
    for r in tf_results[:3]:
        ann = f" ({r['annotation']})" if r["annotation"] else ""
        genes_str = ", ".join(r["matched_genes"][:4])
        top_tfs_summary.append(
            f"{r['tf']}{ann}: activity={r['activity_score']:+.2f}, "
            f"targets=[{genes_str}]"
        )

    output = {
        "cluster":           cluster_id,
        "tissue":            tissue,
        "n_degs_used":       len(deg_df),
        "tf_results":        tf_results,
        "top_tfs_summary":   top_tfs_summary,
        "top_tf":            tf_results[0]["tf"] if tf_results else "None",
        "top_tf_annotation": tf_results[0]["annotation"] if tf_results else "",
        "n_tfs_found":       len(tf_results),
        "n_cancer_relevant": sum(1 for r in tf_results if r["cancer_relevant"]),
        "agent_confidence":  confidence,
        "uncertain":         confidence < 0.30,
        "uncertainty_claims":uncertainty_claims,
    }

    # Serialize — convert all numpy types to Python native
    def _safe(o):
        if hasattr(o, 'item'): return o.item()
        raise TypeError(type(o))
    cache_file.write_text(json.dumps(output, indent=2, default=_safe))
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner for all clusters
# ─────────────────────────────────────────────────────────────────────────────

def run_all_clusters_regulatory(
    degs_dir:      Path,
    tissue:        str = "lung",
    dataset_label: str = "luad",
) -> list:
    """
    Run Regulatory Agent on all clusters in a DEG directory.
    """
    results = []
    deg_files = sorted(degs_dir.glob("cluster_*_degs.csv"),
                       key=lambda f: int(f.stem.split("_")[1]))

    if not deg_files:
        print(f"  No DEG files found in {degs_dir}")
        return []

    print(f"\n  Running Regulatory Agent on {len(deg_files)} clusters "
          f"(tissue={tissue!r}, dataset={dataset_label!r})...")

    for f in deg_files:
        cl  = f.stem.split("_")[1]
        df  = pd.read_csv(f)
        if df.empty or "names" not in df.columns:
            continue

        result = run_regulatory_agent(cl, df, tissue=tissue)
        results.append(result)

        top  = result["top_tf"] if result["n_tfs_found"] > 0 else "none"
        conf = result["agent_confidence"]
        ann  = f" — {result['top_tf_annotation'][:35]}" if result["top_tf_annotation"] else ""
        unc  = " [UNCERTAIN]" if result["uncertain"] else ""
        print(f"    Cluster {cl:>2}: top TF = {top:<12}{ann:<38} "
              f"conf={conf:.2f}{unc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Integration with GRACE orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def format_regulatory_evidence(reg_result: dict) -> str:
    """
    Format regulatory agent output as evidence string for the LLM narrator.
    """
    if reg_result["n_tfs_found"] == 0:
        return "Regulatory analysis: no significant TF activity detected."

    lines = [f"Regulatory analysis (DoRothEA, {reg_result['n_tfs_found']} TFs found):"]
    for tf_info in reg_result["tf_results"][:3]:
        ann   = f" [{tf_info['annotation']}]" if tf_info["annotation"] else ""
        genes = ", ".join(tf_info["matched_genes"][:4])
        lines.append(
            f"  • {tf_info['tf']}{ann}: activity {tf_info['activity_score']:+.2f} "
            f"({tf_info['direction']}), targets=[{genes}]"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Regulatory Agent (Agent 5) — standalone test")
    print("=" * 60)

    luad_degs_dir = Path("results/degs")
    hcc_degs_dir  = Path("results/hcc/degs")

    # ── LUAD ──────────────────────────────────────────────────────────────────
    if luad_degs_dir.exists():
        print("\n── LUAD (lung tissue) ──")
        luad_results = run_all_clusters_regulatory(
            luad_degs_dir, tissue="lung", dataset_label="luad"
        )
        out = Path("results/regulatory_luad.json")
        out.write_text(json.dumps(luad_results, indent=2, default=lambda o: o.item() if hasattr(o,'item') else str(o)))
        print(f"\n  Saved → {out}")
        confs = [r["agent_confidence"] for r in luad_results]
        print(f"  Mean confidence:      {np.mean(confs):.3f}")
        print(f"  Uncertain clusters:   {sum(1 for r in luad_results if r['uncertain'])}/{len(luad_results)}")
        print(f"  Cancer TF clusters:   {sum(1 for r in luad_results if r['n_cancer_relevant']>0)}/{len(luad_results)}")

    # ── HCC ───────────────────────────────────────────────────────────────────
    if hcc_degs_dir.exists():
        print("\n── HCC (liver tissue) ──")
        hcc_results = run_all_clusters_regulatory(
            hcc_degs_dir, tissue="liver", dataset_label="hcc"
        )
        out = Path("results/hcc/regulatory_hcc.json")
        out.write_text(json.dumps(hcc_results, indent=2, default=lambda o: o.item() if hasattr(o,'item') else str(o)))
        print(f"\n  Saved → {out}")
        confs = [r["agent_confidence"] for r in hcc_results]
        print(f"  Mean confidence:      {np.mean(confs):.3f}")

    # ── Demo: known TAM DEGs ──────────────────────────────────────────────────
    print("\n── Demo: Cluster 2 TAM markers ──")
    demo_df = pd.DataFrame({
        "names": ["CD68","CD163","TREM2","GPNMB","C1QA","C1QB","C1QC",
                  "APOE","MARCO","SPP1","MRC1","FN1","LYZ","S100A8"],
        "logfoldchanges": [2.1,1.9,1.8,1.6,2.3,2.1,2.0,1.4,1.2,1.8,1.5,1.3,2.5,2.2],
    })
    result = run_regulatory_agent("demo_tam", demo_df, tissue="lung")
    print(f"\n  Top TFs for TAM cluster:")
    for r in result["tf_results"][:3]:
        print(f"    {r['tf']:<10} activity={r['activity_score']:+.3f}  "
              f"targets={r['matched_genes'][:4]}  "
              f"({r['annotation'][:40]})")
    print(f"\n  Confidence: {result['agent_confidence']:.3f}")
    print(f"\n  Narrator evidence string:")
    print(f"  {format_regulatory_evidence(result)}")

    print("\n── Demo: Cluster 18 proliferating markers ──")
    demo_df2 = pd.DataFrame({
        "names": ["MKI67","TOP2A","TYMS","CDK1","CCNA2","BIRC5","CENPE",
                  "TPX2","BUB1","BUB1B","CDC20","PLK1","AURKB","PCNA"],
        "logfoldchanges": [3.1,2.9,2.4,2.6,2.3,2.1,1.9,1.8,1.7,1.6,1.5,2.0,1.8,2.2],
    })
    result2 = run_regulatory_agent("demo_prolif", demo_df2, tissue="lung")
    print(f"\n  Top TFs for proliferating cluster:")
    for r in result2["tf_results"][:3]:
        print(f"    {r['tf']:<10} activity={r['activity_score']:+.3f}  "
              f"targets={r['matched_genes'][:4]}  "
              f"({r['annotation'][:40]})")
    print(f"\n  Confidence: {result2['agent_confidence']:.3f}")

    print("\n" + "=" * 60)
    print("Agent 5 test complete.")
    print("Integration: add to day3_agents_orchestrator.py:")
    print("  from regulatory_agent import run_regulatory_agent,")
    print("                               format_regulatory_evidence")
    print("  c_overall += 0.15 × c_regulatory (reduce other weights)")
    print("=" * 60)
