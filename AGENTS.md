# GRACE Agent API Reference

All agents follow the same interface contract:
- Accept `cluster_id`, `deg_list`, and context parameters
- Return a dict with `agent_confidence` ∈ [0,1], `uncertain` bool, `uncertainty_claims` list
- Cache results to `cache/` using content-hash keys
- Fail gracefully (return confidence=0.0) if the external API is unavailable

---

## Agent 1: DEG Validator (UniProt Swiss-Prot)

**File:** `grace/day3_agents_orchestrator.py` — `agent_deg_validator()`

**Purpose:** Verify that each DEG corresponds to a real, reviewed human protein entry. Reduces hallucination risk by confirming gene-level biological facts.

**Knowledge source:** UniProt REST API — Swiss-Prot reviewed human entries only (not TrEMBL).

**Input:**
```python
agent_deg_validator(
    deg_list: list[str],   # Top-20 DEG gene symbols
    cluster_id: str        # For caching
)
```

**Output:**
```python
{
    "cluster": "0",
    "verified_genes": ["CD3D", "CD8A", "GZMA"],  # Confirmed in UniProt
    "unverified_genes": ["TRAV12-1"],             # Not found (novel/non-canonical)
    "agent_confidence": 0.75,                     # fraction verified
    "uncertain": False,
    "uncertainty_claims": []
}
```

**Confidence formula:** `c_DEG = n_verified / n_queried`

**Note:** c_DEG = 0.0 for 10/20 LUAD clusters, reflecting that many cancer-specific marker genes are not yet in Swiss-Prot reviewed entries. This is expected and is itself a biologically meaningful signal.

---

## Agent 2: Pathway Agent (Reactome)

**File:** `grace/day3_agents_orchestrator.py` — `agent_pathway()`

**Purpose:** Confirm that enriched pathways exist in Reactome and retrieve canonical pathway descriptions. Detects biological contradictions (e.g. co-enrichment of proliferation and apoptosis).

**Knowledge source:** Reactome REST API — `https://reactome.org/ContentService/search/query?query={pathway}`

**Input:**
```python
agent_pathway(
    deg_list: list[str],         # DEG list for context
    top_pathways: list[str],     # Top 5 Enrichr pathways
    cluster_id: str
)
```

**Output:**
```python
{
    "confirmed_pathways": [
        {"name": "T Cell Receptor Signaling", "reactome_id": "R-HSA-202403",
         "description": "..."}
    ],
    "unconfirmed_pathways": [".."],
    "biological_conflicts": [],   # Empty if no contradictions detected
    "agent_confidence": 1.0,      # 18/20 LUAD clusters = 1.0
    "uncertain": False,
    "uncertainty_claims": []
}
```

**Confidence formula:** `c_pathway = n_confirmed / n_queried`

**Conflict detection:** Flags co-occurrence of functionally antagonistic pathway pairs (e.g. Cell Cycle + Apoptosis; EMT + Epithelial Differentiation).

---

## Agent 3: Disease Agent (MyGene.info / DisGeNET)

**File:** `grace/day3_agents_orchestrator.py` — `agent_disease()`

**Purpose:** Map DEGs to known disease associations and compute cancer-type relevance. Provides disease context that is absent from pathway enrichment alone.

**Knowledge sources:**
- MyGene.info REST API for gene–disease associations
- DisGeNET-curated gene–disease scores
- Hardcoded cancer driver gene lists (LUAD: KRAS, EGFR, TP53, STK11; HCC: TP53, CTNNB1, AXIN1, ARID1A)

**Input:**
```python
agent_disease(
    deg_list: list[str],
    top_pathways: list[str],
    cluster_id: str
)
```

**Output:**
```python
{
    "disease_associations": [
        {"gene": "KRAS", "disease": "Lung adenocarcinoma", "score": 0.89}
    ],
    "luad_relevance": 0.33,      # Fraction of DEGs matching LUAD drivers
    "luad_driver_genes": ["KRAS"],
    "agent_confidence": 0.45,
    "uncertain": False,
    "uncertainty_claims": []
}
```

**Confidence formula:** `c_disease = min(luad_relevance + 0.30, 1.0)`

---

## Agent 4: Cell Identity Agent (CellMarker 2.0)

**File:** `grace/cell_identity_agent.py`

**Purpose:** Match the cluster's DEG profile against 60,877 human cell type marker gene entries in CellMarker 2.0. Provides the primary cell type label.

**Knowledge source:** CellMarker 2.0 database — downloaded and cached as CSV at first run. Queried with tissue-specific filtering.

**Input:**
```python
run_cell_identity_agent(
    cluster_id: str,
    deg_list: list[str],
    tissue: str = "lung"    # "lung" for LUAD, "liver" for HCC
)
```

**Output:**
```python
{
    "cluster": "2",
    "best_cell_type": "Tumour-associated macrophage (TAM)",
    "best_matched_genes": ["C1QC", "CD163", "CD68", "GPNMB"],
    "best_score": 0.87,
    "agent_confidence": 0.87,  # harmonic(precision, recall, Jaccard)
    "uncertain": False,
    "uncertainty_claims": [],
    "all_candidates": [...]     # Top 5 candidates with scores
}
```

**Confidence formula:**
```
precision = |matched ∩ cluster_degs| / |cluster_degs|
recall    = |matched ∩ cluster_degs| / |cellmarker_markers|
jaccard   = |intersection| / |union|
c_cell_id = harmonic_mean(precision, recall, jaccard)
```

**Tissue filtering:** Setting `tissue="liver"` restricts CellMarker 2.0 lookup to liver-specific cell type entries. This is the mechanism that enables zero-shot cross-cancer generalisation — no other configuration changes are required.

---

## Agent 5: Regulatory Agent (DoRothEA — planned)

**File:** `grace/regulatory_agent.py`

**Purpose:** Estimate transcription factor activity per cluster using DoRothEA curated TF–target regulons. Provides regulatory-layer context to the narrator.

**Status:** Implemented with a 106-interaction curated fallback regulon. Full DoRothEA integration requires `decoupler` (not yet in the conda environment on all platforms).

**Knowledge source:** DoRothEA curated regulon (A+B confidence interactions) — fallback CSV bundled in `grace/data/dorothea_curated_fallback.csv`.

**Input:**
```python
run_regulatory_agent(
    cluster_id: str,
    deg_df: pd.DataFrame,   # columns: names, logfoldchanges
    tissue: str = "lung"
)
```

**Output:**
```python
{
    "top_tf": "SPI1/PU.1",
    "n_tfs_found": 3,
    "tf_activities": {"SPI1": 0.82, "IRF4": 0.61, "CEBPB": 0.45},
    "agent_confidence": 0.54,
    "uncertainty_claims": []
}
```

---

## Agent 6: Novel Population Agent (GPT-5.4 conditional)

**File:** `grace/novel_population_agent.py`

**Purpose:** For clusters that fail cell identity matching (c_cell_id ≤ 0.35 or c_overall ≤ 0.40), generate a structured biological hypothesis for the novel or transitional cell state. Outputs an explicit evidence gap statement and experimental validation recommendation.

**Trigger conditions:**
```python
if c_cell_id <= 0.35 or c_overall <= 0.40 or n_matched_genes <= 2:
    run_novel_population_agent(...)
```

**Input:**
```python
run_novel_population_agent(
    cluster_id: str,
    deg_list: list[str],
    pathway_evidence: dict,
    disease_evidence: dict,
    tissue: str
)
```

**Output:**
```python
{
    "novel_pop_narrative": "1. Closest known cell type: ...\n2. Evidence gaps: ...\n
                            3. Hypothesis: ...\n4. Experimental validation: ...",
    "agent_confidence": 0.40,
    "uncertainty_claims": ["Novel population: no CellMarker match above threshold"]
}
```

---

## Agent 7: Literature Evidence Agent (PubMed)

**File:** `grace/literature_agent.py`

**Purpose:** Retrieve supporting publications from PubMed for each cluster's interpretation. Anchors biological claims to real PMIDs, directly addressing hallucination risk.

**Knowledge source:** NCBI E-utilities REST API (esearch + esummary). No API key required (rate-limited to 3 req/sec). Set `NCBI_API_KEY` environment variable for 10 req/sec.

**Input:**
```python
run_literature_agent(
    cluster_id: str,
    deg_list: list[str],
    cell_type: str,         # From Agent 4
    tissue: str = "lung",
    max_papers: int = 8
)
```

**Output:**
```python
{
    "n_papers_found": 5,
    "papers": [
        {
            "pmid": "34686340",
            "title": "TREM2 macrophages in the tumour microenvironment...",
            "journal": "Nature Communications",
            "year": 2021,
            "authors": "Zheng H et al.",
            "relevance": 0.82
        }
    ],
    "agent_confidence": 0.68,
    "uncertainty_claims": []
}
```

**Relevance scoring:** Papers are scored on single-cell keyword presence, gene name mentions, cell type mentions, journal tier, and recency (≥2022 receives a bonus).

---

## Orchestrator

**File:** `grace/day3_agents_orchestrator.py` — `orchestrate()`

**Purpose:** Merge all agent outputs, compute weighted confidence, detect conflicts, propagate uncertainty, and assemble the evidence packet for the narrator.

**Confidence formula (GRACE v2, 4 agents):**
```
c_overall = 0.20 × c_DEG
          + 0.30 × c_pathway
          + 0.20 × c_disease
          + 0.30 × c_cell_id
```

**Confidence formula (GRACE v3, 6 agents):**
```
c_overall = 0.15 × c_DEG
          + 0.25 × c_pathway
          + 0.15 × c_disease
          + 0.25 × c_cell_id
          + 0.10 × c_regulatory
          + 0.10 × c_literature
```

**Uncertainty flagging:** Clusters with `c_overall < 0.50` are marked as uncertain. The narrator is instructed to use [UNCERTAIN] tags for any claim from an agent with confidence below its individual threshold.

**Conflict detection:** Pathway pairs known to be biologically antagonistic are checked; any confirmed conflict is propagated as a `conflict_flag` in the narrator prompt.

---

## LLM Narrator

**Configuration:** GPT-5.4, Azure OpenAI, `temperature=0`, `max_completion_tokens=800`

**Prompt contract:**
1. Every biological claim must be supported by evidence from the provided packet
2. Claims from low-confidence agents must be tagged [UNCERTAIN]
3. Any conflict flag must be explicitly acknowledged
4. Output must end with a confidence statement citing c_overall

**Caching:** All narrator responses are cached with a hash of the evidence packet. Identical evidence → identical response, regardless of run order or date.
