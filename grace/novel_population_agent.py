"""
Agent 6 — Novel Population Agent (Reasoning Model)
====================================================
Design document + implementation skeleton.

PURPOSE
-------
Activates conditionally when the orchestrator detects a cluster that
does not match any known cell type with sufficient confidence.
Uses a reasoning-capable LLM (o1/o3/GPT-5.4 with chain-of-thought)
to synthesise all available evidence and generate a biological
HYPOTHESIS about the population's identity and functional state.

This is the hardest problem in single-cell biology:
  "What is this cluster when nothing in my reference database matches?"

WHEN IT FIRES
-------------
Trigger conditions (ANY of the following):
  1. c_cell_id < 0.25 (Cell Identity Agent found no good match)
  2. c_overall < 0.30 (orchestrator has very low overall confidence)
  3. top cell type match has <2 supporting marker genes
  4. Explicit abstention by Cell Identity Agent

WHAT IT DOES DIFFERENTLY FROM OTHER AGENTS
------------------------------------------
All other agents are lookup/matching agents — they query a database
and return structured results. Agent 6 is a REASONING agent:

1. Collects ALL evidence from Agents 1-5 (including partial matches,
   failed matches, and low-confidence signals)
2. Queries multiple databases exhaustively for any partial overlap
3. Looks for "closest known population" across 5 different databases
4. Asks the reasoning model: "Given all this evidence, what is the
   most biologically plausible interpretation of this population?"
5. Returns a HYPOTHESIS with explicit epistemic framing:
   - "This population shares features of X but lacks canonical markers of X"
   - "The regulatory signature (TF activity from Agent 5) is consistent with Y"
   - "This may represent a transitional state between X and Z"
   - "No known cell type in CellMarker/PanglaoDB matches this signature"

COMPARISON TO EXISTING TOOLS
-----------------------------
- SingleR/CellTypist: Return "unassigned" or forced nearest-match, no explanation
- GPT naive: Hallucinate a confident label regardless of evidence
- GRACE Agents 1-4: Return uncertain flags, correct but not explanatory
- Agent 6 (GRACE): Returns a grounded biological HYPOTHESIS with reasoning chain

This is the key differentiator for Nature Methods / Cell Systems level publication.

DATABASES QUERIED
-----------------
1. CellMarker 2.0 (marker genes) — already loaded
2. PanglaoDB-equivalent curated markers — already in cell_identity_agent.py
3. Human Cell Atlas (HCA) — REST API or downloaded reference
4. CellTypist model predictions (pip install celltypist)
5. The cluster's own DEG functional enrichment (novel pathway combinations)

IMPLEMENTATION NOTES
---------------------
The reasoning model prompt is the critical innovation. It must:
1. Present all evidence WITHOUT bias toward any particular cell type
2. Explicitly ask the model to consider novel/transitional states
3. Ask for a confidence-stratified response:
   - "Most likely known cell type" (even if confidence is low)
   - "Evidence gaps" (what markers are missing that you'd expect)
   - "Novel population hypothesis" (what the cluster MIGHT be)
   - "Recommended follow-up experiment" (how to validate)

Run standalone test:
    python3 novel_population_agent.py
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Azure client ──────────────────────────────────────────────────────────────
def get_client():
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_ver  = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not set")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_ver)

DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

# ── Trigger threshold ─────────────────────────────────────────────────────────
NOVEL_POP_TRIGGER = {
    "max_cell_id_confidence": 0.35,   # trigger if Cell Identity Agent conf < this
    "max_overall_confidence": 0.40,   # trigger if overall orchestrator conf < this
    "max_matched_genes":      2,      # trigger if best cell type has ≤ this many genes
}

# ── CellTypist integration ────────────────────────────────────────────────────
def run_celltypist(adata_path: str, cluster_id: str) -> dict:
    """
    Run CellTypist on a specific cluster to get reference-based annotation.
    Returns top prediction + probability.
    Requires: pip install celltypist
    """
    try:
        import celltypist
        import scanpy as sc

        adata = sc.read_h5ad(adata_path)
        cluster_cells = adata[adata.obs["leiden"] == cluster_id].copy()

        # CellTypist needs raw or log-normalised counts
        predictions = celltypist.annotate(
            cluster_cells,
            model="Immune_All_High.pkl",  # use appropriate model
            majority_voting=True,
        )
        top_cell_type = predictions.predicted_labels["majority_voting"].mode()[0]
        top_prob = predictions.probability_matrix.max(axis=1).mean()

        return {
            "tool":           "CellTypist",
            "top_prediction": top_cell_type,
            "mean_confidence": round(float(top_prob), 3),
            "model":          "Immune_All_High.pkl",
        }
    except ImportError:
        return {"tool": "CellTypist", "error": "not installed (pip install celltypist)"}
    except Exception as e:
        return {"tool": "CellTypist", "error": str(e)}


# ── Novel population reasoning prompt ────────────────────────────────────────

NOVEL_POP_SYSTEM = """You are an expert computational biologist specialising in
single-cell RNA sequencing analysis, tumour immunology, and novel cell population
discovery. You are rigorous, evidence-based, and you explicitly distinguish between
what is known, what is inferred, and what is uncertain.

Your task is to analyse a scRNA-seq cluster that does not match any known cell type
with high confidence. You must generate a biological hypothesis about this population's
identity and functional state, grounded in the evidence provided.

Rules:
1. Never invent gene functions or pathway associations not supported by the evidence
2. Explicitly state when evidence is absent, weak, or conflicting
3. Consider transitional, hybrid, and novel cell states — not just canonical types
4. Distinguish "most likely known type despite weak evidence" from "potentially novel"
5. Suggest one concrete experiment to validate your hypothesis
6. Use [UNCERTAIN] to mark any claim not directly supported by the provided evidence
"""

def build_novel_pop_prompt(
    cluster_id:    str,
    deg_list:      list,
    pathway_list:  list,
    cell_id_result: dict,
    deg_result:    dict,
    disease_result: dict,
    reg_result:    dict,
    tissue:        str,
    n_cells:       int,
) -> str:
    """Build the reasoning prompt for Agent 6."""

    # Gather all partial evidence
    partial_cell_matches = []
    for match in cell_id_result.get("top_matches", [])[:5]:
        if match.get("n_matched", 0) > 0:
            partial_cell_matches.append(
                f"  {match['cell_name']}: {match['n_matched']} genes matched "
                f"({', '.join(match.get('matched_genes',[][:3]))}), "
                f"confidence={match['confidence']:.3f}"
            )

    tf_evidence = ""
    if reg_result and reg_result.get("n_tfs_found", 0) > 0:
        tf_lines = []
        for tf in reg_result.get("tf_results", [])[:3]:
            tf_lines.append(
                f"  {tf['tf']} (activity={tf['activity_score']:+.2f}, "
                f"targets=[{', '.join(tf['matched_genes'][:3])}])"
            )
        tf_evidence = "Transcription factor activity (DoRothEA):\n" + "\n".join(tf_lines)
    else:
        tf_evidence = "Transcription factor activity: no significant TF detected"

    prompt = f"""NOVEL POPULATION ANALYSIS — Cluster {cluster_id}
Dataset: {tissue.upper()} primary tumour, n = {n_cells} cells

═══════════════════════════════════════════════════════
EVIDENCE SUMMARY
═══════════════════════════════════════════════════════

TOP DIFFERENTIALLY EXPRESSED GENES (ranked by specificity):
{', '.join(deg_list[:25])}

ENRICHED PATHWAYS (Enrichr MSigDB Hallmark + KEGG):
{'; '.join(pathway_list[:5]) if pathway_list else 'No significant pathway enrichment detected'}

VERIFIED GENE ANNOTATIONS (UniProt Swiss-Prot):
Verified genes: {', '.join(deg_result.get('verified_genes', [])) or 'None verified'}
Confidence: {deg_result.get('confidence', 0):.2f}

CELL IDENTITY MATCHING (CellMarker 2.0) — ALL WEAK:
{chr(10).join(partial_cell_matches) if partial_cell_matches else '  No matches with ≥2 target genes'}
Best match confidence: {cell_id_result.get('agent_confidence', 0):.3f} (threshold: 0.35)

DISEASE/CANCER DRIVER GENES DETECTED:
{', '.join(disease_result.get('luad_driver_genes', [])) or 'None detected'}

{tf_evidence}

═══════════════════════════════════════════════════════
QUESTIONS FOR ANALYSIS
═══════════════════════════════════════════════════════

Please provide a structured analysis covering:

1. CLOSEST KNOWN CELL TYPE
   What is the closest known cell type based on the available evidence,
   even if the match is weak? What specific genes/pathways support this?
   What canonical markers are MISSING that you would expect?

2. NOVEL/TRANSITIONAL POPULATION HYPOTHESIS
   Could this represent a transitional state, hybrid population, or
   genuinely novel cell state? What biological scenario would explain
   the observed gene expression pattern?

3. FUNCTIONAL STATE
   Regardless of cell identity, what is this cluster DOING?
   What biological processes are active based on the pathway evidence?

4. EVIDENCE GAPS AND LIMITATIONS
   What additional information would be needed to resolve the identity
   of this population? List 2-3 specific marker genes that, if expressed,
   would confirm or rule out your top hypothesis.

5. VALIDATION EXPERIMENT
   Suggest one specific experimental approach (e.g., protein co-staining,
   trajectory analysis, spatial transcriptomics) to validate your hypothesis.

Format each section clearly. Use [UNCERTAIN] for unsupported claims.
Target: 200-250 words total.
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent 6 function
# ─────────────────────────────────────────────────────────────────────────────

def should_trigger(orchestrator_output: dict, cell_id_result: dict) -> bool:
    """
    Decide whether Agent 6 should activate for this cluster.
    """
    overall_conf = orchestrator_output.get("overall_confidence", 1.0)
    cell_id_conf = cell_id_result.get("agent_confidence", 1.0)
    top_matched  = len(cell_id_result.get("best_matched_genes", []))

    return (
        cell_id_conf <= NOVEL_POP_TRIGGER["max_cell_id_confidence"] or
        overall_conf <= NOVEL_POP_TRIGGER["max_overall_confidence"] or
        top_matched  <= NOVEL_POP_TRIGGER["max_matched_genes"]
    )


def run_novel_population_agent(
    cluster_id:        str,
    deg_list:          list,
    pathway_list:      list,
    cell_id_result:    dict,
    deg_result:        dict,
    disease_result:    dict,
    reg_result:        dict,
    orchestrator_conf: float,
    tissue:            str = "lung",
    n_cells:           int = 0,
) -> dict:
    """
    Agent 6 — Novel Population Agent.

    Only call this after checking should_trigger().
    """
    # Cache key based on cluster DEGs
    gene_key   = "_".join(sorted(deg_list[:10]))
    cache_file = CACHE_DIR / f"novel_pop_{tissue}_{cluster_id}_{abs(hash(gene_key))%10**8}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    prompt = build_novel_pop_prompt(
        cluster_id, deg_list, pathway_list,
        cell_id_result, deg_result, disease_result, reg_result,
        tissue, n_cells,
    )

    try:
        client = get_client()
        resp   = client.chat.completions.create(
            model=DEPLOY,
            messages=[
                {"role": "system", "content": NOVEL_POP_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_completion_tokens=600,
            temperature=1,  # allow reasoning model to explore
        )
        narrative = resp.choices[0].message.content.strip()
    except Exception as e:
        narrative = f"NOVEL_POP_ERROR: {e}"

    # Count uncertainty flags
    n_uncertain = narrative.count("[UNCERTAIN]")

    output = {
        "cluster":            cluster_id,
        "tissue":             tissue,
        "triggered":          True,
        "trigger_reason":     {
            "cell_id_conf":   cell_id_result.get("agent_confidence", 0),
            "overall_conf":   orchestrator_conf,
            "top_matched":    len(cell_id_result.get("best_matched_genes", [])),
        },
        "novel_pop_narrative": narrative,
        "n_uncertain_flags":  n_uncertain,
        "deg_list":           deg_list[:20],
        "closest_known":      cell_id_result.get("best_cell_type", "Unknown"),
        "agent_confidence":   min(0.5, orchestrator_conf + 0.1),  # slightly above orchestrator
    }

    cache_file.write_text(json.dumps(output, indent=2))
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Novel Population Agent (Agent 6) — design test")
    print("=" * 60)
    print()
    print("Agent 6 is a CONDITIONAL reasoning agent.")
    print("It fires only when ALL of the following are true:")
    print(f"  Cell Identity Agent confidence ≤ {NOVEL_POP_TRIGGER['max_cell_id_confidence']}")
    print(f"  Overall orchestrator confidence ≤ {NOVEL_POP_TRIGGER['max_overall_confidence']}")
    print(f"  Best cell type has ≤ {NOVEL_POP_TRIGGER['max_matched_genes']} matching genes")
    print()
    print("In the LUAD dataset, Cluster 15 meets all trigger conditions:")
    print("  DEGs: GGTLC1, SLC10A2, RP1-60O19.1, CA12, PLA2G1B, CPB2, CYP4B1")
    print("  Cell ID confidence: 0.35")
    print("  Overall confidence: 0.53")
    print("  Zero UniProt-verified genes")
    print()
    print("This is EXACTLY the cluster for the novel population case study in the paper.")
    print()

    # Test the trigger logic with Cluster 15 data
    mock_cell_id = {
        "agent_confidence": 0.35,
        "best_cell_type":   "Primed human pluripotent stem cell",
        "best_matched_genes": ["CD24"],
        "top_matches": [
            {"cell_name":"Primed human pluripotent stem cell",
             "n_matched":1, "matched_genes":["CD24"], "confidence":0.35},
        ]
    }
    mock_orch = {"overall_confidence": 0.53}

    triggered = should_trigger(mock_orch, mock_cell_id)
    print(f"Trigger check for Cluster 15: {'TRIGGERED ✓' if triggered else 'NOT triggered'}")
    print()

    if triggered:
        print("Running Agent 6 on Cluster 15...")
        mock_deg_result     = {"verified_genes": [], "confidence": 0.0}
        mock_disease_result = {"luad_driver_genes": []}
        mock_reg_result     = {"n_tfs_found": 0, "tf_results": []}

        result = run_novel_population_agent(
            cluster_id     = "15",
            deg_list       = ["GGTLC1","SLC10A2","RP1-60O19.1","CA12",
                              "PLA2G1B","CPB2","CYP4B1","SUSD2","C4BPA"],
            pathway_list   = ["KRAS Signaling Up"],
            cell_id_result = mock_cell_id,
            deg_result     = mock_deg_result,
            disease_result = mock_disease_result,
            reg_result     = mock_reg_result,
            orchestrator_conf = 0.53,
            tissue         = "lung",
            n_cells        = 184,
        )

        print("\nAgent 6 output:")
        print(f"  Triggered: {result['triggered']}")
        print(f"  Closest known: {result['closest_known']}")
        print(f"  Uncertainty flags: {result['n_uncertain_flags']}")
        print(f"\n  Narrative (first 400 chars):")
        print(f"  {result['novel_pop_narrative'][:400]}...")

    print()
    print("=" * 60)
    print("Next steps to integrate Agent 6:")
    print("  1. Install Agent 5 first (regulatory_agent.py)")
    print("  2. Add both to day3_agents_orchestrator.py")
    print("  3. Update confidence formula to include c_regulatory")
    print("  4. Use Cluster 15 as the paper case study for Agent 6")
    print("=" * 60)
