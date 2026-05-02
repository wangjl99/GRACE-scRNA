# GRACE Reproducibility Guide

This document provides a complete step-by-step guide to reproduce all results, figures, and tables from the paper. Every number in the paper can be reproduced by following these steps.

---

## Environment

Tested on:
- Ubuntu 22.04 / CentOS 7
- Python 3.10.x
- 32GB RAM minimum (64GB recommended for full datasets)
- No GPU required

All API responses used in the paper are cached in `cache/`. If you have the cache directory, you can reproduce all results without any API keys or internet access.

```bash
# Clone with cache (full reproducibility, ~800MB)
git clone --depth 1 https://github.com/YOUR_USERNAME/GRACE-scRNA.git
cd GRACE-scRNA

# Or without cache (requires API keys, downloads data fresh)
git clone --filter=blob:none https://github.com/YOUR_USERNAME/GRACE-scRNA.git
```

---

## Step 0: Environment setup

```bash
conda env create -f environment.yml
conda activate grace_env

# Or with pip:
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint and key
```

---

## Step 1: Data acquisition and preprocessing

### LUAD (GSE131907)

```bash
python preprocessing/day1_download_preprocess.py --dataset luad

# What this does:
#  1. Downloads GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz from GEO FTP
#  2. Downloads cell annotation file
#  3. Filters to primary tumour cells (Sample_Origin == 'tLung')
#  4. Scanpy QC: min_genes=200, max_genes=5000, pct_mt<15%
#  5. Normalise to 10,000 counts, log1p transform
#  6. Select 2,000 HVGs, PCA (30 components), k-NN (k=15)
#  7. Leiden clustering (resolution=0.5) → 20 clusters
#  8. UMAP (n_components=2, min_dist=0.3)
#  9. Saves: results/gse131907_tlung_processed.h5ad
#            results/author_labels_per_cluster.csv (20 rows)

# Expected output:
#   9,708 cells × 2,000 genes
#   20 Leiden clusters
#   Runtime: ~3 minutes
```

### HCC (GSE149614)

```bash
python preprocessing/day1_download_preprocess.py --dataset hcc

# What this does:
#  1. Downloads GSE149614_HCC_counts.txt.gz + metadata from GEO
#  2. Filters to primary tumour cells (site == 'Tumor')
#  3. Same QC and clustering pipeline
#  4. Saves: results/hcc/gse149614_hcc_processed.h5ad
#            results/hcc/hcc_author_labels_per_cluster.csv (25 rows)

# Expected output:
#   8,868 cells × 2,000 genes
#   25 Leiden clusters
#   Runtime: ~4 minutes
```

---

## Step 2: DEG analysis and baseline

```bash
python preprocessing/day2_deg_pathway_baseline.py

# What this does:
#   LUAD:
#   1. Wilcoxon DEG analysis (one-vs-rest), top 50 per cluster
#   2. gseapy Enrichr: MSigDB Hallmark + KEGG 2021
#   3. GPT-5.4 naive baseline: top 20 DEGs → ungrounded narrative
#   4. Saves: results/degs/cluster_{i}_degs.csv (20 files)
#             results/baseline_results.json (GPT naive narratives)
#   HCC (run automatically):
#   5. Same pipeline with tissue="liver"
#   6. Saves: results/hcc/degs/cluster_{i}_degs.csv (25 files)
#             results/hcc/hcc_baseline_results.json

# Runtime: ~8 minutes (GPT naive calls cached after first run)
```

---

## Step 3: GRACE 4-agent orchestrator

### LUAD

```bash
python grace/day3_agents_orchestrator.py

# What this does (per cluster, 20 clusters):
#   Agent 1: Query top-20 DEGs against UniProt Swiss-Prot
#            → c_DEG = fraction verified
#   Agent 2: Query top pathways against Reactome REST API
#            → c_pathway = fraction confirmed
#   Agent 3: Query DEGs against MyGene.info / DisGeNET
#            → c_disease = cancer relevance score
#   Agent 4: Query DEGs against CellMarker 2.0 (tissue="lung")
#            → c_cell_id = harmonic(precision, recall, Jaccard)
#   Orchestrator: c_overall = 0.20*c_DEG + 0.30*c_pathway
#                           + 0.20*c_disease + 0.30*c_cell_id
#                 Flag if c_overall < 0.50
#   Narrator: GPT-5.4 (temp=0) with structured evidence packet
#
# Saves: results/versionB_results.json

# Expected results:
#   Mean confidence: 0.55
#   Clusters with c_overall >= 0.50: 17/20
#   Runtime: ~15 minutes (all API calls cached)
```

### HCC (zero-shot)

```bash
python grace/day3_hcc.py

# Identical pipeline with TISSUE="liver"
# No HCC-specific configuration required
# CellMarker 2.0 queried with tissue="liver" automatically

# Saves: results/hcc/hcc_versionB_results.json

# Expected results:
#   Mean confidence: 0.42
#   Runtime: ~18 minutes
```

---

## Step 4: Evaluation metrics

```bash
python evaluation/day5_metrics.py

# Computes (LUAD):
#   1. BERTScore F1 (distilbert-base-uncased vs Kim 2020 abstract)
#      GSEA: 0.715 | GPT naive: 0.736 | GRACE v2: 0.725
#   2. GO-term precision/recall/F1
#      GRACE: Prec=0.601, Rec=0.875, F1=0.689
#      GPT naive: Prec=0.470, Rec=0.887, F1=0.572
#   3. Uncertainty calibration gap:
#      High-conf clusters (c>=0.50): mean GO-recall=0.833
#      Low-conf clusters  (c<0.50):  mean GO-recall=0.800
#      Gap: +0.033 (GRACE v3) / +0.132 (GRACE v2) — well-calibrated ✓
#
# Saves: results/table1_full_metrics.csv
#        results/calibration_results.json
```

---

## Step 5: Cell type accuracy

```bash
python evaluation/day6_accuracy_comparison.py

# Computes:
#   LUAD: GRACE v2 vs GPT naive vs Kim 2020 author labels
#         GRACE:     weighted=100.0%, macro=100.0%
#         GPT naive: weighted=85.7%,  macro=80.0%
#   HCC:  GRACE v2 vs GPT naive vs Ma 2021 author labels
#         GRACE:     weighted=93.3%,  macro=92.0%
#         GPT naive: weighted=43.9%,  macro=40.0%
```

---

## Step 6: SingleR comparison

```bash
python evaluation/run_singleR_python.py

# Implements SingleR algorithm in Python (no R required):
#   - Spearman correlation against HumanPrimaryCellAtlas reference
#   - Fine-tuning step on most variable genes
#   - Delta-score pruning (threshold = 0.05)
#
# Expected results:
#   LUAD: weighted=91.1%, macro=90.0%, abstained=2/20
#   HCC:  weighted=80.4%, macro=68.0%, abstained=3/25
#
# Saves: results/singleR_luad_results.csv
#        results/hcc/singleR_hcc_results.csv
#        results/singleR_summary.json
```

---

## Step 7: Generate all figures

```bash
python figures/draw_all_figures_final.py
python figures/draw_hcc_novel_populations.py
python figures/draw_singleR_comparison.py

# Outputs (300 DPI PNG + PDF) in figures/:
#   fig1_grace_architecture
#   fig2_pathway_heatmaps
#   fig3_confidence_scores
#   fig4_novel_case_study
#   fig5_metrics_comparison
#   fig6A/B/C_calibration (LUAD and HCC versions)
#   fig7_luad_comparison
#   fig8A-E_cross_cancer
#   hcc_umap_novel_populations
#   figS2_singleR_comparison
```

---

## Expected results summary (Table 1)

| Method | GO-F1 | GO-Prec | CellType-W (LUAD) | CellType-W (HCC) | Calib |
|--------|-------|---------|-------------------|------------------|-------|
| GSEA | 0.267 | 0.350 | ~2% | N/A | — |
| SingleR | N/A | N/A | 91.1% | 80.4% | — |
| GPT naive | 0.572 | 0.470 | 85.7% | 43.9% | — |
| GRACE v1 | 0.663 | 0.577 | 55.6% | — | +0.134 |
| **GRACE v2** | **0.689** | **0.601** | **100.0%** | **93.3%** | **+0.132** |

---

## Verification checksums

After running the full pipeline, verify your results match the paper:

```bash
python docs/verify_results.py

# Expected output:
#   ✓  LUAD GRACE weighted accuracy: 100.0%
#   ✓  LUAD GPT naive weighted accuracy: 85.7%
#   ✓  LUAD SingleR weighted accuracy: 91.1%
#   ✓  HCC GRACE weighted accuracy: 93.3%
#   ✓  GO-term F1 (GRACE v2): 0.689
#   ✓  GO-term Precision (GRACE v2): 0.601
#   ✓  Calibration gap (GRACE v2): +0.132
#   ✓  Mean confidence (LUAD): 0.55
#   ✓  Mean confidence (HCC): 0.42
#   All 9 checks passed ✓
```

---

## Runtime summary

| Step | Runtime (first run) | Runtime (cached) |
|------|---------------------|-----------------|
| LUAD preprocessing | ~3 min | ~1 min |
| HCC preprocessing | ~4 min | ~1 min |
| DEG + Enrichr + GPT naive | ~8 min | ~1 min |
| GRACE LUAD (4 agents) | ~15 min | ~2 min |
| GRACE HCC (4 agents) | ~18 min | ~2 min |
| Metrics (BERTScore + GO) | ~5 min | ~3 min |
| SingleR Python | ~3 min | <1 min |
| All figures | ~4 min | ~4 min |
| **Total** | **~60 min** | **~15 min** |

All Azure OpenAI and external API responses are cached. If you have the `cache/` directory from the repository, the entire pipeline runs in ~15 minutes without any API calls.

---

## Troubleshooting

### h5ad file not found

Make sure preprocessing has completed and the file path is correct:
```bash
ls results/*.h5ad results/hcc/*.h5ad
```

### Azure OpenAI rate limit

The orchestrator is rate-limited by default (1 request/second). All responses are cached — if a run is interrupted, simply re-run and it will resume from the last cached point.

### Different GO-term F1 values

The GO-term evaluation uses a curated 13-category reference set defined in `evaluation/day5_metrics.py`. This reference was fixed before running any experiments. Do not modify the reference set after running.

### SingleR label mismatch

The `LUAD_AUTHOR_TO_VOCAB` and `SINGLER_TO_VOCAB` mapping dictionaries in `evaluation/run_singleR_python.py` normalise labels to a common vocabulary. If you use a different dataset, extend these mappings appropriately.
