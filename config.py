"""
config.py
=========
Central configuration for GRACE.
All paths, API settings, and hyperparameters live here.

Usage:
    from grace.config import AZURE_CLIENT, RESULTS_DIR, TISSUE
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
RESULTS_DIR  = ROOT_DIR / "results"
HCC_DIR      = RESULTS_DIR / "hcc"
CACHE_DIR    = ROOT_DIR / "cache"
FIGURES_DIR  = ROOT_DIR / "figures"
DATA_DIR     = ROOT_DIR / "data"

for d in [RESULTS_DIR, HCC_DIR, CACHE_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY        = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
AZURE_API_VER    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

def get_azure_client() -> AzureOpenAI:
    """Return configured Azure OpenAI client."""
    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise ValueError(
            "Azure OpenAI credentials not set. "
            "Copy .env.example to .env and fill in AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_KEY."
        )
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version=AZURE_API_VER,
    )

# ── GRACE hyperparameters ─────────────────────────────────────────────────────

# Orchestrator confidence weights (4-agent GRACE v2)
WEIGHTS_V2 = {
    "deg":         0.20,
    "pathway":     0.30,
    "disease":     0.20,
    "cell_id":     0.30,
}

# Orchestrator confidence weights (6-agent GRACE v3)
WEIGHTS_V3 = {
    "deg":         0.15,
    "pathway":     0.25,
    "disease":     0.15,
    "cell_id":     0.25,
    "regulatory":  0.10,
    "literature":  0.10,
}

# Uncertainty threshold
UNCERTAINTY_THRESHOLD = 0.50

# Agent 6 trigger conditions
NOVEL_POP_TRIGGER_CELL_ID   = 0.35   # c_cell_id below this → trigger
NOVEL_POP_TRIGGER_OVERALL   = 0.40   # c_overall below this → trigger
NOVEL_POP_TRIGGER_MIN_GENES = 2      # fewer matched genes → trigger

# ── DEG settings ──────────────────────────────────────────────────────────────
N_DEGS_FOR_LLM   = 20    # DEGs passed to LLM narrator
N_DEGS_FOR_AGENT = 20    # DEGs passed to knowledge agents
N_PATHWAYS       = 5     # Top pathways from Enrichr

# ── LLM narrator ──────────────────────────────────────────────────────────────
NARRATOR_TEMPERATURE    = 0
NARRATOR_MAX_TOKENS     = 800
NARRATOR_MODEL          = AZURE_DEPLOYMENT

# ── Leiden clustering ─────────────────────────────────────────────────────────
LEIDEN_RESOLUTION   = 0.5
N_HVG               = 2000
N_PCS               = 30
N_NEIGHBORS         = 15

# ── Dataset paths ─────────────────────────────────────────────────────────────
LUAD_H5AD     = RESULTS_DIR / "gse131907_tlung_processed.h5ad"
HCC_H5AD      = HCC_DIR    / "gse149614_hcc_processed.h5ad"
LUAD_LABELS   = RESULTS_DIR / "author_labels_per_cluster.csv"
HCC_LABELS    = HCC_DIR    / "hcc_author_labels_per_cluster.csv"

# ── GEO download URLs ─────────────────────────────────────────────────────────
LUAD_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/"
    "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
)
LUAD_ANNOT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/"
    "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
)
HCC_COUNTS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/"
    "GSE149614_HCC_counts.txt.gz"
)
HCC_META_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/"
    "GSE149614_HCC_metadata.txt.gz"
)
