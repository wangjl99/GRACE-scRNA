#!/bin/bash
# reproduce_luad.sh
# ==================
# One-command reproduction of all LUAD results from the paper.
# Runtime: ~60 min (first run) / ~15 min (with cache)
#
# Usage:
#   bash reproduce_luad.sh
#   bash reproduce_luad.sh --skip-download  # if h5ad already exists
#   bash reproduce_luad.sh --figures-only   # just regenerate figures

set -e

SKIP_DOWNLOAD=false
FIGURES_ONLY=false

for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
    --figures-only)  FIGURES_ONLY=true ;;
  esac
done

echo "========================================================"
echo "GRACE — Full LUAD reproduction pipeline"
echo "========================================================"

if [ "$FIGURES_ONLY" = false ]; then

  # Step 1: Preprocessing
  if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[1/7] Downloading and preprocessing LUAD (GSE131907)..."
    python preprocessing/day1_download_preprocess.py --dataset luad
  else
    echo "[1/7] Skipping download (--skip-download)"
  fi

  # Step 2: DEGs + Enrichr + GPT naive baseline
  echo ""
  echo "[2/7] DEG analysis, pathway enrichment, GPT naive baseline..."
  python preprocessing/day2_deg_pathway_baseline.py

  # Step 3: GRACE 4-agent orchestrator
  echo ""
  echo "[3/7] Running GRACE 4-agent orchestrator (LUAD)..."
  python grace/day3_agents_orchestrator.py

  # Step 4: Evaluation metrics
  echo ""
  echo "[4/7] Computing evaluation metrics (BERTScore, GO-term, calibration)..."
  python evaluation/day5_metrics.py

  # Step 5: Cell type accuracy
  echo ""
  echo "[5/7] Cell type accuracy vs Kim 2020 author labels..."
  python evaluation/day6_accuracy_comparison.py

  # Step 6: SingleR comparison
  echo ""
  echo "[6/7] Running SingleR Python implementation..."
  python evaluation/run_singleR_python.py

fi

# Step 7: Figures
echo ""
echo "[7/7] Generating all paper figures..."
python figures/draw_all_figures_final.py
python figures/draw_singleR_comparison.py

echo ""
echo "========================================================"
echo "Reproduction complete."
echo ""
echo "Key results:"
python -c "
import json, pandas as pd
from pathlib import Path

# Table 1
t1 = pd.read_csv('results/table1_definitive_final.csv')
print('Table 1 (LUAD):')
print(t1[['Method','GO-F1','CellType-W','Calib']].to_string(index=False))

# SingleR
try:
    sr = json.load(open('results/singleR_summary.json'))
    print(f'\nSingleR LUAD: weighted={sr[\"luad\"][\"weighted\"]}%  macro={sr[\"luad\"][\"macro\"]}%')
except: pass
"
echo ""
echo "Figures saved to: figures/"
echo "========================================================"
