#!/bin/bash
# Run Wanda pruning across sparsity levels and save Hamming-distance-permuted
# layer-0 weight matrices for each condition.
#
# Sparsity sweep: 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 99%
#
# Target hardware: g6.2xlarge (1x NVIDIA L4, 24 GB VRAM)
#
# Usage:
#   bash scripts/run_layer0_sparsity.sh [OPTIONS]
#
# Options:
#   --prune_method  wanda|sparsegpt          (default: wanda)
#   --group_size    <int>                    (default: 8)
#   --permute_axis  columns|rows             (default: columns)
#   --base_output   <path>                   (default: ./permutation_results)
#   --model         <hf_model_id_or_path>    (default: baffo32/decapoda-research-llama-7B-hf)
#   --cache_dir     <path>                   (default: llm_weights)
#   --target_layer  <int>                    (default: 0)
#
# Outputs land in <base_output>/:
#   workloads/sparsity_<X>/   <- permuted .pt + perm_order .pt
#   metrics/sparsity_<X>/     <- per-group text metrics
#   images/sparsity_<X>/      <- visualizations
#   logs/sparsity_<X>/        <- perplexity log

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL="baffo32/decapoda-research-llama-7B-hf"
CACHE_DIR="llm_weights"
PRUNE_METHOD="wanda"
TARGET_LAYER=0
GROUP_SIZE=8
PERMUTE_AXIS="columns"   # 'columns' (horizontal) or 'rows' (vertical)
BASE_OUTPUT=""           # derived automatically if not set via --base_output

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --prune_method)  PRUNE_METHOD="$2";  shift 2 ;;
        --group_size)    GROUP_SIZE="$2";    shift 2 ;;
        --permute_axis)  PERMUTE_AXIS="$2";  shift 2 ;;
        --base_output)   BASE_OUTPUT="$2";   shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --cache_dir)     CACHE_DIR="$2";     shift 2 ;;
        --target_layer)  TARGET_LAYER="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Derive BASE_OUTPUT from params if not explicitly set ──────────────────────
if [[ -z "${BASE_OUTPUT}" ]]; then
    AXIS_SHORT="col"
    [[ "${PERMUTE_AXIS}" == "rows" ]] && AXIS_SHORT="row"
    BASE_OUTPUT="./${PRUNE_METHOD}_grp_${GROUP_SIZE}_${AXIS_SHORT}_perm_results"
fi

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
        --base_output    "${BASE_OUTPUT}" \
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
