#!/bin/bash
# Run the 5 remaining layer-0 sparsity experiments (sequential, single GPU).
#
# Already completed:
#   wanda    grp=8  rows     -> wanda_grp_8_row_perm_results
#   wanda    grp=8  columns  -> wanda_grp_8_col_perm_results
#   sparsegpt grp=8 columns  -> sparsegpt_grp_8_col_perm_results
#
# This script runs:
#   wanda    grp=16  rows
#   wanda    grp=16  columns
#   sparsegpt grp=8  rows
#   sparsegpt grp=16 rows
#   sparsegpt grp=16 columns
#
# Usage:
#   nohup bash scripts/run_remaining_experiments.sh > logs/remaining_experiments.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_layer0_sparsity.sh"

export CUDA_VISIBLE_DEVICES=0

# Each entry: "prune_method group_size permute_axis"
# BASE_OUTPUT is auto-derived by run_layer0_sparsity.sh:
#   ./<method>_grp_<size>_<row|col>_perm_results
EXPERIMENTS=(
    "wanda    16  rows"
    "wanda    16  columns"
    "sparsegpt  8  rows"
    "sparsegpt 16  rows"
    "sparsegpt 16  columns"
)

TOTAL=${#EXPERIMENTS[@]}
IDX=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    read -r METHOD GROUP AXIS <<< "${ENTRY}"

    # Mirror the naming logic from run_layer0_sparsity.sh for the log message
    AXIS_SHORT="col"; [[ "${AXIS}" == "rows" ]] && AXIS_SHORT="row"
    OUTDIR="${METHOD}_grp_${GROUP}_${AXIS_SHORT}_perm_results"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  [${IDX}/${TOTAL}]  method=${METHOD}  group=${GROUP}  axis=${AXIS}"
    echo "  output -> ./${OUTDIR}"
    echo "══════════════════════════════════════════════════════════════"

    bash "${BASE_SCRIPT}" \
        --prune_method  "${METHOD}" \
        --group_size    "${GROUP}" \
        --permute_axis  "${AXIS}"

    echo "  [${IDX}/${TOTAL}] Done -> ./${OUTDIR}"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  All ${TOTAL} remaining experiments complete."
echo "══════════════════════════════════════════════════════════════"
