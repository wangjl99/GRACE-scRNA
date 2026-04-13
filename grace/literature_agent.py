"""
Agent 7 — Literature Evidence Agent (PubMed)
=============================================
Given a cluster's top DEG list and cell identity prediction,
retrieves supporting literature from PubMed using the NCBI
E-utilities REST API (no API key required for basic use).

For each cluster returns:
  - Top 3-5 supporting publications per key gene/cell type
  - Publication titles, journal, year, PMID
  - Relevance score for each publication
  - c_literature confidence score ∈ [0,1]

This directly addresses hallucination: every biological claim
in the GRACE narrator can be anchored to a real PMID rather
than parametric LLM memory.

No API key required (uses NCBI E-utilities free tier).
Rate limit: 3 requests/second without API key.
With NCBI_API_KEY env var: 10 requests/second.

Run standalone test:
    python3 literature_agent.py
"""

import os, sys, json, time, re, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── NCBI E-utilities ──────────────────────────────────────────────────────────
NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  # optional — increases rate limit
RATE_LIMIT   = 0.35  # seconds between requests (3/sec without key, 0.1 with key)

# ── Search parameters ─────────────────────────────────────────────────────────
MAX_PAPERS_PER_GENE  = 3    # papers retrieved per gene query
MAX_PAPERS_PER_CELL  = 5    # papers retrieved for cell type query
MAX_TOTAL_PAPERS     = 8    # maximum papers returned per cluster
MIN_YEAR             = 2018  # only recent papers (≥2018)
CANCER_CONTEXT       = {
    "lung":  "lung adenocarcinoma LUAD single cell",
    "liver": "hepatocellular carcinoma HCC single cell",
}

# ── High-value gene-disease combinations ─────────────────────────────────────
# These are searched first as they are most likely to return relevant papers
PRIORITY_GENE_CONTEXT = {
    "TREM2":  "tumour associated macrophage",
    "CD163":  "tumour associated macrophage immunosuppression",
    "FOXP3":  "regulatory T cell tumour",
    "CD8A":   "cytotoxic T cell exhaustion cancer",
    "PDCD1":  "PD-1 T cell exhaustion checkpoint",
    "NKX2-1": "LUAD lung adenocarcinoma lineage",
    "KRAS":   "KRAS lung adenocarcinoma oncogene",
    "EPCAM":  "epithelial cell adhesion cancer",
    "MKI67":  "proliferating tumour cell cycle",
    "CDH5":   "tumour endothelial angiogenesis",
    "COL1A1": "cancer associated fibroblast ECM",
    "ACTA2":  "hepatic stellate cell fibrosis",
    "ALB":    "hepatocyte liver function",
    "AFP":    "hepatocellular carcinoma biomarker",
    "HNF4A":  "hepatocyte transcription factor",
    "CD3D":   "T lymphocyte single cell",
    "CD19":   "B cell lymphocyte",
    "KIT":    "mast cell activation",
    "NKG7":   "natural killer cell cytotoxicity",
    "GPNMB":  "macrophage tumour microenvironment",
}


# ─────────────────────────────────────────────────────────────────────────────
# PubMed query helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ncbi_request(url: str, retries: int = 3) -> dict:
    """Make a rate-limited NCBI E-utilities request, return parsed JSON."""
    if NCBI_API_KEY:
        url += f"&api_key={NCBI_API_KEY}"
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT)
            req = urllib.request.Request(
                url, headers={"User-Agent": "GRACE-ScRNA-Framework/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                return {}
            time.sleep(1.0)
    return {}


def search_pubmed(query: str, max_results: int = 5) -> list:
    """
    Search PubMed for a query string.
    Returns list of PMIDs.
    """
    params = urllib.parse.urlencode({
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
        "mindate": str(MIN_YEAR),
        "maxdate": "3000",
        "datetype":"pdat",
    })
    url  = f"{NCBI_BASE}/esearch.fcgi?{params}"
    data = _ncbi_request(url)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_paper_details(pmids: list) -> list:
    """
    Fetch title, journal, year, authors for a list of PMIDs.
    Returns list of paper dicts.
    """
    if not pmids:
        return []

    params = urllib.parse.urlencode({
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "json",
        "rettype": "abstract",
    })
    url  = f"{NCBI_BASE}/esummary.fcgi?{params}"
    data = _ncbi_request(url)

    papers = []
    result = data.get("result", {})
    for pmid in pmids:
        info = result.get(str(pmid), {})
        if not info or "error" in info:
            continue

        # Extract title
        title = info.get("title", "").strip()
        if not title:
            continue

        # Extract journal
        journal = info.get("source", "")

        # Extract year
        pub_date = info.get("pubdate", "")
        year_match = re.search(r"(20\d{2})", pub_date)
        year = int(year_match.group(1)) if year_match else 0

        # Extract authors (first + last)
        authors = info.get("authors", [])
        author_str = ""
        if authors:
            first = authors[0].get("name", "")
            if len(authors) > 1:
                author_str = f"{first} et al."
            else:
                author_str = first

        papers.append({
            "pmid":    str(pmid),
            "title":   title[:150],
            "journal": journal,
            "year":    year,
            "authors": author_str,
            "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return papers


# ─────────────────────────────────────────────────────────────────────────────
# Relevance scoring
# ─────────────────────────────────────────────────────────────────────────────

RELEVANCE_KEYWORDS = [
    "single.cell", "scRNA", "scrna-seq", "single-cell",
    "tumour microenvironment", "tumor microenvironment", "TME",
    "lung adenocarcinoma", "LUAD", "hepatocellular",
    "macrophage", "T cell", "NK cell", "fibroblast",
    "cancer-associated", "tumour-associated", "tumor-associated",
    "cell type", "marker gene", "DEG", "differentially expressed",
    "clustering", "UMAP", "leiden", "seurat",
]

def score_relevance(paper: dict, deg_list: list, cell_type: str) -> float:
    """Score how relevant a paper is to our cluster analysis."""
    score = 0.0
    text  = (paper["title"] + " " + paper["journal"]).lower()

    # Single-cell relevance
    for kw in RELEVANCE_KEYWORDS[:8]:
        if re.search(kw.lower(), text):
            score += 0.15

    # Gene mention
    for gene in deg_list[:10]:
        if gene.lower() in text:
            score += 0.2; break

    # Cell type mention
    for word in cell_type.lower().split()[:3]:
        if len(word) > 3 and word in text:
            score += 0.15; break

    # Recency bonus
    if paper["year"] >= 2022:
        score += 0.1
    elif paper["year"] >= 2020:
        score += 0.05

    # Top journal bonus
    journal_lower = paper["journal"].lower()
    if any(j in journal_lower for j in ["nature", "cell", "science",
                                          "immunity", "cancer", "lancet"]):
        score += 0.15

    return round(min(score, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_queries(
    deg_list:  list,
    cell_type: str,
    tissue:    str = "lung",
) -> list:
    """
    Build a prioritised list of PubMed queries for this cluster.
    Returns list of (query_string, purpose) tuples.
    """
    context = CANCER_CONTEXT.get(tissue, "cancer single cell")
    queries = []

    # 1. Cell type + cancer context (most important)
    if cell_type and cell_type != "Unknown":
        ct_clean = re.sub(r'\(.*?\)', '', cell_type).strip()[:40]
        queries.append((
            f'"{ct_clean}" {context}',
            f"cell type: {ct_clean}"
        ))

    # 2. Top priority genes with known context
    for gene in deg_list[:10]:
        gene_upper = gene.strip().upper()
        if gene_upper in PRIORITY_GENE_CONTEXT:
            queries.append((
                f'"{gene_upper}" {PRIORITY_GENE_CONTEXT[gene_upper]}',
                f"priority gene: {gene_upper}"
            ))

    # 3. Top DEGs with tissue context (for genes not in priority list)
    for gene in deg_list[:5]:
        gene_upper = gene.strip().upper()
        if gene_upper not in PRIORITY_GENE_CONTEXT:
            queries.append((
                f'"{gene_upper}" {context}',
                f"DEG: {gene_upper}"
            ))

    return queries[:6]  # limit total queries to avoid rate limiting


# ─────────────────────────────────────────────────────────────────────────────
# Main agent function
# ─────────────────────────────────────────────────────────────────────────────

def run_literature_agent(
    cluster_id:  str,
    deg_list:    list,
    cell_type:   str,
    tissue:      str = "lung",
    max_papers:  int = MAX_TOTAL_PAPERS,
) -> dict:
    """
    Agent 7 — Literature Evidence Agent.

    Args:
        cluster_id: Leiden cluster identifier
        deg_list:   Top DEG gene symbols
        cell_type:  Cell Identity Agent best prediction
        tissue:     'lung' or 'liver'
        max_papers: Maximum papers to return

    Returns:
        Dict with supporting publications, confidence, uncertainty.
    """
    # Cache key
    gene_key   = "_".join(sorted([g.upper() for g in deg_list[:8]]))
    cache_file = CACHE_DIR / f"literature_{tissue}_{cluster_id}_{abs(hash(gene_key))%10**8}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    queries    = build_queries(deg_list, cell_type, tissue)
    all_papers = []
    seen_pmids = set()

    for query, purpose in queries:
        if len(all_papers) >= max_papers:
            break
        pmids  = search_pubmed(query, max_results=MAX_PAPERS_PER_GENE)
        papers = fetch_paper_details(pmids)

        for p in papers:
            if p["pmid"] not in seen_pmids and p["year"] >= MIN_YEAR:
                p["query_purpose"] = purpose
                p["relevance"]     = score_relevance(p, deg_list, cell_type)
                all_papers.append(p)
                seen_pmids.add(p["pmid"])

    # Sort by relevance and take top papers
    all_papers.sort(key=lambda x: -x["relevance"])
    top_papers = all_papers[:max_papers]

    # Compute agent confidence
    if not top_papers:
        confidence = 0.0
    else:
        mean_rel   = np.mean([p["relevance"] for p in top_papers])
        n_relevant = sum(1 for p in top_papers if p["relevance"] >= 0.3)
        confidence = round(
            0.5 * min(n_relevant / 3, 1.0) +
            0.5 * min(mean_rel, 1.0),
            3
        )

    # Format citation summary for narrator
    citations = []
    for p in top_papers[:4]:
        citations.append(
            f"{p['authors']} ({p['year']}) {p['title'][:80]}... "
            f"[{p['journal']}] PMID:{p['pmid']}"
        )

    # Uncertainty
    uncertainty_claims = []
    if not top_papers:
        uncertainty_claims.append(
            "No supporting literature found for this cluster's marker genes "
            "and predicted cell type. Biological interpretation is based on "
            "database knowledge only, without direct publication support."
        )
    elif confidence < 0.3:
        uncertainty_claims.append(
            f"Limited literature support (confidence={confidence:.2f}): "
            f"retrieved papers may not directly describe this cell population "
            "in this cancer context."
        )

    output = {
        "cluster":           cluster_id,
        "tissue":            tissue,
        "cell_type_queried": cell_type,
        "n_papers_found":    len(top_papers),
        "papers":            top_papers,
        "citations":         citations,
        "queries_run":       [q for q, _ in queries],
        "agent_confidence":  confidence,
        "uncertain":         confidence < 0.30,
        "uncertainty_claims":uncertainty_claims,
    }

    cache_file.write_text(json.dumps(output, indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_clusters_literature(
    degs_dir:      Path,
    cell_id_file:  Path,
    tissue:        str = "lung",
    dataset_label: str = "luad",
) -> list:
    """Run Literature Agent on all clusters."""
    import pandas as pd

    ci_map = {}
    if cell_id_file.exists():
        ci_data = json.loads(cell_id_file.read_text())
        ci_map  = {r["cluster"]: r.get("best_cell_type", "Unknown")
                   for r in ci_data}

    deg_files = sorted(degs_dir.glob("cluster_*_degs.csv"),
                       key=lambda f: int(f.stem.split("_")[1]))

    results = []
    print(f"\n  Running Literature Agent on {len(deg_files)} clusters "
          f"(tissue={tissue!r}, PubMed)...")
    print("  Note: rate-limited to 3 req/sec — expect ~2 min for 20 clusters")

    for f in deg_files:
        cl   = f.stem.split("_")[1]
        df   = pd.read_csv(f)
        if df.empty or "names" not in df.columns:
            continue
        degs  = df["names"].head(15).str.upper().tolist()
        ctype = ci_map.get(cl, "Unknown")

        result = run_literature_agent(cl, degs, ctype, tissue=tissue)
        results.append(result)

        n     = result["n_papers_found"]
        conf  = result["agent_confidence"]
        top   = result["papers"][0]["title"][:50] if result["papers"] else "none"
        unc   = " [UNCERTAIN]" if result["uncertain"] else ""
        print(f"    Cluster {cl:>2}: {n} papers  conf={conf:.2f}  "
              f"top: {top}...{unc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Format for narrator
# ─────────────────────────────────────────────────────────────────────────────

def format_literature_evidence(lit_result: dict, max_cites: int = 3) -> str:
    """Format literature evidence as evidence string for LLM narrator."""
    if not lit_result["papers"]:
        return "Literature support: no supporting publications retrieved."

    lines = [f"Supporting literature ({lit_result['n_papers_found']} papers found):"]
    for p in lit_result["papers"][:max_cites]:
        lines.append(
            f"  • {p['authors']} ({p['year']}) \"{p['title'][:70]}...\" "
            f"[{p['journal']}] PMID:{p['pmid']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Literature Evidence Agent (Agent 7) — standalone test")
    print("=" * 60)

    luad_degs_dir  = Path("results/degs")
    luad_ci_file   = Path("results/cell_identity_luad.json")
    hcc_degs_dir   = Path("results/hcc/degs")
    hcc_ci_file    = Path("results/hcc/cell_identity_hcc.json")

    # Quick demo first — test API connectivity
    print("\n── Quick API test (Cluster 2 TAM) ──")
    demo_result = run_literature_agent(
        cluster_id = "2_demo",
        deg_list   = ["TREM2","CD163","GPNMB","C1QA","C1QB","CD68","APOE"],
        cell_type  = "Tumour-associated macrophage (TAM)",
        tissue     = "lung",
        max_papers = 4,
    )
    print(f"  Papers found: {demo_result['n_papers_found']}")
    print(f"  Confidence:   {demo_result['agent_confidence']:.3f}")
    if demo_result["papers"]:
        print(f"\n  Top citations:")
        for p in demo_result["papers"][:3]:
            print(f"    [{p['year']}] {p['title'][:70]}...")
            print(f"           {p['journal']} | PMID:{p['pmid']}")
    print()
    print(format_literature_evidence(demo_result))

    # Full LUAD run
    if luad_degs_dir.exists():
        print("\n── Full LUAD run ──")
        luad_results = run_all_clusters_literature(
            luad_degs_dir, luad_ci_file,
            tissue="lung", dataset_label="luad"
        )
        out = Path("results/literature_luad.json")
        out.write_text(json.dumps(luad_results, indent=2,
            default=lambda o: o.item() if hasattr(o,"item") else str(o)))
        confs = [r["agent_confidence"] for r in luad_results]
        print(f"\n  Saved → {out}")
        print(f"  Mean confidence:    {np.mean(confs):.3f}")
        print(f"  Uncertain clusters: "
              f"{sum(1 for r in luad_results if r['uncertain'])}/{len(luad_results)}")
        total_papers = sum(r['n_papers_found'] for r in luad_results)
        print(f"  Total papers found: {total_papers}")

    # HCC run
    if hcc_degs_dir.exists():
        print("\n── Full HCC run ──")
        hcc_results = run_all_clusters_literature(
            hcc_degs_dir, hcc_ci_file,
            tissue="liver", dataset_label="hcc"
        )
        out = Path("results/hcc/literature_hcc.json")
        out.write_text(json.dumps(hcc_results, indent=2,
            default=lambda o: o.item() if hasattr(o,"item") else str(o)))
        confs = [r["agent_confidence"] for r in hcc_results]
        print(f"\n  Saved → {out}")
        print(f"  Mean confidence:    {np.mean(confs):.3f}")

    print("\n" + "=" * 60)
    print("Agent 7 complete.")
    print("Integration: add to day3_agents_orchestrator.py:")
    print("  from literature_agent import run_literature_agent,")
    print("                              format_literature_evidence")
    print()
    print("Updated 5-agent confidence formula:")
    print("  c_overall = 0.15×c_DEG + 0.25×c_pathway + 0.15×c_disease")
    print("            + 0.25×c_cell_id + 0.10×c_regulatory + 0.10×c_literature")
    print("=" * 60)
