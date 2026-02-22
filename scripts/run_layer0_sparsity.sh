#!/bin/bash
# Run Wanda pruning across sparsity levels and save Hamming-distance-permuted
# layer-0 weight matrices for each condition.
#
# Sparsity sweep: 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 99%
#
# Target hardware: g6.2xlarge (1x NVIDIA L4, 24 GB VRAM)
#
# Usage:
#   bash scripts/run_layer0_sparsity.sh
#
# Outputs land in:
#   permutation_results/workloads/sparsity_<X>/   <- permuted .pt + perm_order .pt
#   permutation_results/metrics/sparsity_<X>/     <- per-group text metrics
#   permutation_results/images/sparsity_<X>/      <- visualizations
#   permutation_results/logs/sparsity_<X>/        <- perplexity log

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL="baffo32/decapoda-research-llama-7B-hf"
CACHE_DIR="llm_weights"
PRUNE_METHOD="wanda"
TARGET_LAYER=0
GROUP_SIZE=8
PERMUTE_AXIS="columns"   # 'columns' (horizontal) or 'rows' (vertical)
BASE_OUTPUT="./permutation_results"

export CUDA_VISIBLE_DEVICES=0

# ── Sparsity levels ───────────────────────────────────────────────────────────
SPARSITY_LEVELS=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99)

# ── Run ───────────────────────────────────────────────────────────────────────
TOTAL=${#SPARSITY_LEVELS[@]}
IDX=0

for SPARSITY in "${SPARSITY_LEVELS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  [${IDX}/${TOTAL}]  sparsity=${SPARSITY}  layer=${TARGET_LAYER}  group_size=${GROUP_SIZE}  axis=${PERMUTE_AXIS}"
    echo "══════════════════════════════════════════════════════════════"

    python main.py \
        --model          "${MODEL}" \
        --prune_method   "${PRUNE_METHOD}" \
        --sparsity_ratio "${SPARSITY}" \
        --sparsity_type  unstructured \
        --cache_dir      "${CACHE_DIR}" \
        --save           "${BASE_OUTPUT}/logs/sparsity_${SPARSITY}" \
        --save_permuted \
        --target_layer   "${TARGET_LAYER}" \
        --group_size     "${GROUP_SIZE}" \
        --permute_axis   "${PERMUTE_AXIS}"

    echo "  Done.  Matrices → ${BASE_OUTPUT}/workloads/sparsity_${SPARSITY}/"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  All ${TOTAL} sparsity levels complete."
echo "  Permuted matrices: ${BASE_OUTPUT}/workloads/"
echo "══════════════════════════════════════════════════════════════"
