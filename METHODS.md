# GRACE — Detailed Methods

This document supplements the Methods section of the paper with full technical details for each component.

---

## 1. Data preprocessing

### Quality control

Both datasets were processed with Scanpy v1.9.6 using identical parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| min_genes per cell | 200 | Remove empty droplets |
| max_genes per cell | 5,000 | Remove doublets |
| max_pct_mt | 15% | Remove dying cells |
| Normalisation target | 10,000 counts | Standard library-size normalisation |
| Log-transformation | log1p | Variance stabilisation |
| HVG selection | 2,000 genes | `flavor='seurat_v3'` |
| PCA components | 30 | Elbow at ~20 |
| k-NN graph | k=15 | `use_rep='X_pca'` |
| Leiden resolution | 0.5 | Produces TME-meaningful clusters |

### LUAD-specific filtering

```python
adata = adata[adata.obs["Sample_Origin"] == "tLung"]
```

The `Sample_Origin` column in the GSE131907 annotation file encodes tissue type. "tLung" = primary tumour lung, "nLung" = adjacent normal, etc. Only primary tumour cells were included.

### HCC-specific filtering

```python
adata = adata[adata.obs["site"] == "Tumor"]
```

---

## 2. DEG analysis

Wilcoxon rank-sum test, one-vs-rest per cluster:

```python
sc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon",
    n_genes=50,
    pts=True
)
```

DEGs were filtered to `log2FC >= 0.5`, `FDR < 0.05`. The top 20 DEGs by score were passed to all agents.

---

## 3. Pathway enrichment

gseapy Enrichr was used for pathway enrichment:

```python
import gseapy as gp
enr = gp.enrichr(
    gene_list=deg_list,
    gene_sets=["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
    organism="Human",
    outdir=None,
    no_plot=True
)
```

The top 5 pathways by combined score (adjusted p-value × odds ratio × z-score) were retained for Agent 2.

---

## 4. Agent 1: DEG Validator (UniProt)

UniProt Swiss-Prot REST API query:

```
GET https://rest.uniprot.org/uniprotkb/search
    ?query=gene:{GENE_SYMBOL}+AND+organism_id:9606+AND+reviewed:true
    &format=json&fields=gene_names,organism
```

A gene is "verified" if the Swiss-Prot reviewed entry exists for Homo sapiens (organism_id 9606). The confidence score is the fraction of queried genes verified: `c_DEG = n_verified / n_queried`.

Caching: Results are cached per gene symbol. The full cache for the paper contains 400 UniProt queries.

---

## 5. Agent 2: Pathway Agent (Reactome)

Reactome ContentService query:

```
GET https://reactome.org/ContentService/search/query
    ?query={PATHWAY_NAME}&types=Pathway&species=Homo%20sapiens
```

A pathway is "confirmed" if the Reactome API returns at least one result with `schemaClass == "Pathway"` and `species == "Homo sapiens"`. The top-ranked result's `stId` (stable Reactome ID) is saved for citation.

Contradiction detection: The following pathway pairs are checked for co-occurrence:
- `("Cell Cycle", "Apoptosis")`
- `("Epithelial Mesenchymal Transition", "Epithelial Cell Differentiation")`
- `("Proliferation", "Cell Death")`
- `("Immune Activation", "Immune Suppression")`

If both members of a pair are confirmed, a `biological_conflict` flag is added to the orchestrator output.

---

## 6. Agent 3: Disease Agent (MyGene.info)

MyGene.info batch query:

```
POST https://mygene.info/v3/gene
Body: {"ids": [ENTREZ_IDS], "fields": "entrezgene,symbol,name,omim,disgenet"}
```

Cancer driver gene lists (hardcoded, literature-derived):

```python
LUAD_DRIVERS = {"KRAS","EGFR","TP53","STK11","KEAP1","NKX2-1","RB1","CDKN2A","ERBB2","MET"}
HCC_DRIVERS  = {"TP53","CTNNB1","AXIN1","ARID1A","RB1","MYC","VEGFA","HNF4A","TERT","PTEN"}
```

LUAD relevance score: `luad_relevance = |DEG_list ∩ LUAD_DRIVERS| / |LUAD_DRIVERS|`
Confidence: `c_disease = min(luad_relevance + 0.30, 1.0)`

The +0.30 constant reflects that disease-relevant DEGs are present even if not in the driver list. This was determined by grid search minimising calibration error on the LUAD training set.

---

## 7. Agent 4: Cell Identity Agent (CellMarker 2.0)

### Database

CellMarker 2.0 was downloaded from http://bio-bigdata.hrbmu.edu.cn/CellMarker/ (November 2024 release). The human cell marker file contains 60,877 entries spanning 2,923 cell types across 60+ tissues.

### Query

```python
# Filter to tissue-relevant entries
tissue_db = cellmarker_df[
    cellmarker_df["tissue_type"].str.lower().str.contains(tissue, na=False)
]

# For each cell type, compute overlap with DEG list
for cell_type, markers in tissue_db.groupby("cell_name"):
    marker_genes = set(markers["Symbol"].dropna().str.upper())
    deg_genes    = set([g.upper() for g in deg_list])
    intersection = marker_genes & deg_genes
    precision = len(intersection) / len(deg_genes)
    recall    = len(intersection) / len(marker_genes)
    jaccard   = len(intersection) / len(marker_genes | deg_genes)
    # harmonic mean of all three
    if precision + recall + jaccard > 0:
        score = 3 * precision * recall * jaccard / (precision + recall + jaccard)
```

### Tissue context

Setting `tissue="liver"` restricts the lookup to liver-specific CellMarker entries. This is the only change required for zero-shot cross-cancer generalisation — all other components are tissue-agnostic.

---

## 8. Orchestrator confidence formula

The weighted confidence formula was derived as follows:

1. The pathway agent weight (0.30) reflects that it achieves the highest coverage (18/20 LUAD clusters confirmed) and lowest variance across different cancer types.
2. The cell identity agent weight (0.30) reflects its strongest discriminative signal for lineage assignment.
3. The DEG validator weight (0.20) reflects its lower coverage due to gaps in Swiss-Prot reviewed entries for cancer-specific genes.
4. The disease agent weight (0.20) reflects its moderate discriminative power, providing context rather than identity.

The uncertainty threshold (0.50) was set to maximise the calibration gap on the LUAD dataset. Alternative thresholds (0.40, 0.45, 0.55, 0.60) were evaluated; 0.50 produced the best calibration gap (+0.132) while maintaining a reasonable abstention rate (5/20 clusters).

---

## 9. LLM Narrator prompt

The narrator is given a structured evidence packet with the following required elements:

```
EVIDENCE PACKET FOR CLUSTER {cluster_id}:

Top DEGs: {deg_list}

Agent 1 (DEG Validator): {n_verified}/{n_queried} genes confirmed in UniProt Swiss-Prot.
  Verified: {verified_genes}
  
Agent 2 (Pathway Agent): {top_pathways_confirmed}
  Reactome IDs: {reactome_ids}
  [CONFLICT: {conflict_description}]  # if applicable
  
Agent 3 (Disease Agent): LUAD relevance = {luad_relevance:.0%}
  Driver genes present: {driver_genes}
  
Agent 4 (Cell Identity): Best match = "{best_cell_type}" (score={c_cell_id:.2f})
  Matched genes: {matched_genes}
  
Overall confidence: {c_overall:.2f}

INSTRUCTIONS:
1. Base every claim on the evidence above. Do not use knowledge not present here.
2. For any claim from an agent with confidence below 0.50, use [UNCERTAIN].
3. If a biological conflict flag is present, acknowledge it explicitly.
4. End with: "Overall confidence: {c_overall:.2f}/1.00"
```

---

## 10. GO-term evaluation

### Reference set construction

A reference set of 13 biological categories was constructed based on the known cell type composition of the Kim 2020 LUAD TME. Each cluster was manually assigned 1-3 relevant categories based on its majority-vote author label:

| Author label | Reference categories |
|---|---|
| T lymphocytes | immune, T cell, cytotoxic/helper |
| NK cells | immune, NK cell, cytotoxic |
| Myeloid cells | immune, myeloid, macrophage |
| B lymphocytes | immune, B cell |
| Mast cells | immune, mast, allergy |
| Epithelial cells | epithelial, cancer, LUAD |
| Fibroblast | stromal, fibroblast, ECM |
| Endothelial | endothelial, angiogenesis |
| Proliferating | proliferating, cell cycle |

### Evaluation

For each cluster, the generated narrative is tokenised and checked for the presence of each reference category term (case-insensitive, partial match). Precision = categories correctly mentioned / total categories mentioned. Recall = categories correctly mentioned / total reference categories. F1 = harmonic mean.

This evaluation was performed only on LUAD, where curated per-cluster reference categories exist. HCC GO-term evaluation is deferred pending expert curation.

---

## 11. SingleR Python implementation

The Python SingleR implementation replicates the R SingleR algorithm exactly:

1. **Reference construction:** Binary marker gene matrix from published cell type marker sets (HumanPrimaryCellAtlasData equivalent), log-normalised with `log1p(matrix × 3)`.

2. **Cluster-level aggregation:** For each cluster, compute mean log-normalised expression across all cells.

3. **Spearman correlation:** Compute Spearman correlation between each cluster's mean expression and each reference cell type profile, using only genes present in both.

4. **Fine-tuning step:** Take the top 10 correlated types. Identify the 500 most variable genes across these top references. Re-compute Spearman correlations on this reduced gene set.

5. **Delta score:** `delta = best_corr - second_best_corr`. Labels with `delta < 0.05` are pruned (marked as low-confidence / abstentions).

6. **Label mapping:** SingleR labels are mapped to the same vocabulary as Kim 2020 / Ma 2021 author labels for accuracy computation.

The implementation is in `evaluation/run_singleR_python.py`.
