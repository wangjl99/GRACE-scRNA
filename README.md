# GRACE: Grounded multi-agent Reasoner for Annotating Cells with Evidence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-red.svg)](https://doi.org/XXXXX)

GRACE is a knowledge-grounded multi-agent orchestration framework for interpretable functional summarisation of single-cell RNA sequencing (scRNA-seq) data. It decomposes the interpretation pipeline into four specialist agents — each querying a dedicated curated biological database at runtime — before an orchestrator assembles a calibrated evidence packet for GPT-based narrative generation.

---

## Key results

| Metric | SingleR | GPT-5.4 naive | **GRACE v2** |
|--------|---------|---------------|-------------|
| Cell type accuracy — LUAD (weighted) | 91.1% | 85.7% | **100.0%** |
| Cell type accuracy — HCC zero-shot (weighted) | 80.4% | 43.9% | **93.3%** |
| GO-term F1 — LUAD | N/A | 0.572 | **0.689** |
| GO-term Precision — LUAD | N/A | 0.470 | **0.601** |
| Uncertainty flags/cluster | 0 | 0 | **1.65** |
| Calibration gap | — | — | **+0.132 ✓** |

---

## Architecture

```
scRNA-seq data
      │
      ▼
Leiden clusters + Wilcoxon DEGs
      │
      ├──▶ Agent 1: DEG Validator    (UniProt Swiss-Prot REST API)
      ├──▶ Agent 2: Pathway Agent    (Reactome REST API)
      ├──▶ Agent 3: Disease Agent    (MyGene.info / DisGeNET)
      └──▶ Agent 4: Cell Identity    (CellMarker 2.0)
                │
                ▼
        Orchestrator
        c_overall = 0.20×c_DEG + 0.30×c_pathway
                  + 0.20×c_disease + 0.30×c_cell_id
        · Conflict detection
        · Uncertainty flags (c_overall < 0.50)
                │
                ▼
        LLM Narrator (GPT-5.4, temperature=0)
        · All claims evidence-anchored
        · [UNCERTAIN] tags injected
        · Calibrated confidence statement
                │
                ▼
        Grounded narrative + uncertainty flags
        + confidence score + novel population hypothesis
```

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/GRACE-scRNA.git
cd GRACE-scRNA
pip install -r requirements.txt

# Set Azure OpenAI credentials
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-5.4"

# Run on the LUAD demo dataset (downloads ~500MB)
python preprocessing/day1_download_preprocess.py --dataset luad
python preprocessing/day2_deg_pathway_baseline.py
python grace/day3_agents_orchestrator.py
python evaluation/day5_metrics.py
```

---

## Repository structure

```
GRACE-scRNA/
├── grace/                          # Core GRACE framework
│   ├── config.py                   # Azure OpenAI + paths config
│   ├── day3_agents_orchestrator.py # Main 4-agent orchestrator (LUAD)
│   ├── day3_hcc.py                 # HCC orchestrator (zero-shot)
│   ├── cell_identity_agent.py      # Agent 4: CellMarker 2.0
│   ├── regulatory_agent.py         # Agent 5: DoRothEA (planned)
│   ├── novel_population_agent.py   # Agent 6: conditional reasoning
│   └── literature_agent.py         # Agent 7: PubMed E-utilities
│
├── preprocessing/                  # Data acquisition and QC
│   ├── day1_download_preprocess.py # GEO download, Scanpy QC, clustering
│   ├── day2_deg_pathway_baseline.py# DEG analysis, Enrichr, GPT naive
│   └── hcc_preprocess.py           # HCC-specific preprocessing
│
├── evaluation/                     # Metrics and benchmarking
│   ├── day5_metrics.py             # BERTScore, GO-term F1, calibration
│   ├── day6_accuracy_comparison.py # Cell type accuracy vs author labels
│   └── run_singleR_python.py       # SingleR Python implementation
│
├── figures/                        # Figure generation scripts
│   ├── draw_all_figures_final.py   # All 8 main paper figures
│   ├── draw_hcc_novel_populations.py # HCC novel population figures
│   └── draw_singleR_comparison.py  # SingleR comparison figures
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_preprocessing.ipynb      # Step-by-step data preprocessing
│   ├── 02_grace_pipeline.ipynb     # Running GRACE interactively
│   └── 03_results_analysis.ipynb   # Reproducing paper figures/tables
│
├── docs/                           # Extended documentation
│   ├── METHODS.md                  # Detailed methods for each agent
│   ├── AGENTS.md                   # Agent API reference
│   ├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
│   └── CUSTOMISATION.md            # Adding new agents / datasets
│
├── results/                        # Pre-computed results (cached)
│   ├── table1_definitive_final.csv # Table 1 from paper
│   ├── table2_definitive_final.csv # Table 2 from paper
│   ├── singleR_luad_results.csv    # SingleR LUAD results
│   └── hcc/
│       └── singleR_hcc_results.csv # SingleR HCC results
│
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── .env.example                    # Environment variable template
└── LICENSE
```

---

## Installation

### Option A: pip

```bash
pip install -r requirements.txt
```

### Option B: conda (recommended for reproducibility)

```bash
conda env create -f environment.yml
conda activate grace_env
```

### Requirements summary

| Package | Version | Purpose |
|---------|---------|---------|
| scanpy | ≥1.9.6 | scRNA-seq preprocessing |
| anndata | ≥0.9.0 | Data structures |
| gseapy | ≥1.0.0 | Pathway enrichment (Enrichr) |
| openai | ≥1.0.0 | Azure OpenAI API |
| bert-score | ≥0.3.13 | BERTScore evaluation |
| scipy | ≥1.10.0 | SingleR Spearman correlation |
| pandas | ≥1.5.0 | Data handling |
| matplotlib | ≥3.7.0 | Figure generation |
| requests | ≥2.28.0 | REST API calls (UniProt, Reactome) |
| celltypist | ≥1.3.0 | CellTypist baseline (optional) |

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```ini
# .env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Optional: NCBI E-utilities API key (increases rate limit 3→10 req/sec)
NCBI_API_KEY=your-ncbi-key
```

---

## Datasets

### LUAD (primary dataset)

```bash
# Automatic download from GEO (~1.5GB)
python preprocessing/day1_download_preprocess.py --dataset luad

# Or manual download:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907
# File: GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz
# Annotation: GSE131907_Lung_Cancer_cell_annotation.txt.gz
```

### HCC (cross-cancer validation)

```bash
python preprocessing/day1_download_preprocess.py --dataset hcc

# Or manual:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149614
```

---

## Reproducing paper results

Full reproduction guide is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### One-command reproduction (LUAD)

```bash
bash reproduce_luad.sh
```

This runs the complete pipeline:
1. Download and preprocess GSE131907
2. Compute DEGs and pathway enrichment
3. Run GPT-5.4 naive baseline
4. Run 4-agent GRACE orchestrator
5. Compute all evaluation metrics
6. Generate all paper figures

Expected runtime: ~45 minutes (API calls dominate). All API responses are cached — subsequent runs are instant.

### Expected outputs

```
results/
├── versionB_results.json           # GRACE narratives (20 clusters, ~120KB)
├── baseline_results.json           # GPT naive narratives (~375KB)
├── table1_full_metrics.csv         # Table 1 (BERTScore, GO-term, calibration)
├── singleR_luad_results.csv        # SingleR comparison
└── figures/
    ├── fig1_grace_architecture.png
    ├── fig2_pathway_heatmaps.png
    ├── fig3_confidence_scores.png
    ├── fig4_novel_case_study.png
    ├── fig5_metrics_comparison.png
    ├── fig6A_confidence_vs_recall.png
    ├── fig6B_calibration_boxplot.png
    ├── fig6C_uncertainty_flags.png
    ├── fig7_luad_comparison.png
    └── fig8A-E_cross_cancer/
```

---

## Using GRACE on your own data

```python
from grace.day3_agents_orchestrator import orchestrate
import scanpy as sc
import pandas as pd

# Load your processed AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# Run GRACE on a single cluster
degs = ["CD3D", "CD8A", "GZMA", "GZMB", "NKG7"]  # your top DEGs
pathways = ["T cell receptor signaling", "Cytotoxic killing"]
tissue = "lung"  # or "liver", "breast", etc.

result = orchestrate(
    cluster_id="0",
    deg_list=degs,
    top_pathways=pathways,
    tissue=tissue,
    baseline_narrative="CD8+ cytotoxic T cells"  # GPT naive output
)

print(result["orchestration"]["overall_confidence"])
print(result["grounded_narrative"])
print(result["orchestration"]["uncertainty_claims"])
```

---

## Agent API

### Agent 4: Cell Identity (CellMarker 2.0)

```python
from grace.cell_identity_agent import run_cell_identity_agent

result = run_cell_identity_agent(
    cluster_id="0",
    deg_list=["CD3D","CD8A","GZMA","GZMB"],
    tissue="lung"
)
# result["best_cell_type"]    → "CD8+ cytotoxic T cell"
# result["agent_confidence"]  → 0.87
# result["best_matched_genes"]→ ["CD8A", "GZMA", "GZMB"]
```

### Agent 7: Literature (PubMed)

```python
from grace.literature_agent import run_literature_agent

result = run_literature_agent(
    cluster_id="0",
    deg_list=["CD3D","CD8A","GZMA"],
    cell_type="CD8+ cytotoxic T cell",
    tissue="lung",
    max_papers=5
)
# result["papers"]           → list of {pmid, title, journal, year}
# result["agent_confidence"] → 0.74
```

See [`docs/AGENTS.md`](docs/AGENTS.md) for full API reference for all 7 agents.

---

## Caching

All external API calls (UniProt, Reactome, MyGene.info, DisGeNET, CellMarker, PubMed, Azure OpenAI) are cached in `cache/` as JSON files. Cache keys are derived from input content hashes. This ensures:

- Complete reproducibility regardless of API availability
- Near-instant re-runs after first execution
- The `cache/` directory in this repo contains all responses used in the paper

**To force re-query** (bypass cache):
```bash
rm -rf cache/  # clear all cached responses
```

---

## Citation

If you use GRACE in your research, please cite:

```bibtex
@article{wang2026grace,
  title={GRACE: A Knowledge-Grounded Multi-Agent Orchestration Framework
         for Interpretable Functional Summarisation of Single-Cell Transcriptomics},
  author={Wang, Jinlian and Li, Hui and Ju, Cynthia and Liu, Hongfang},
  journal={Nature Communications},
  year={2026},
  doi={10.XXXXX/XXXXX}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Jinlian Wang — [jinlian.wang@institution.edu]
Issues and pull requests welcome.
