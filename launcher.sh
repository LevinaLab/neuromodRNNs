#!/bin/bash
# =============================================================================
# launcher.sh
# Usage: bash launcher.sh <seeds_file> <task> <experiment>
#
# Examples:
#   bash launcher.sh seeds/seeds_pattern_generation.txt pattern_generation diffusion
#   bash launcher.sh seeds/seeds_cue_acummulation_generation.txt pattern_generation diffusion
# =============================================================================
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS_FILE="${1}"
# If the path is relative, resolve it against the script's directory
[[ "$SEEDS_FILE" != /* ]] && SEEDS_FILE="${SCRIPT_DIR}/${SEEDS_FILE}"
TASK="${2}"
EXPERIMENT="${3}"
 
# --- Validate arguments ------------------------------------------------------
if [[ -z "$SEEDS_FILE" || -z "$TASK" || -z "$EXPERIMENT" ]]; then
    echo "Usage: bash launcher.sh <seeds_file> <task> <experiment>"
    exit 1
fi
 
if [[ ! -f "$SEEDS_FILE" ]]; then
    echo "ERROR: Seeds file '$SEEDS_FILE' not found."
    exit 1
fi
 
# Adjust accordingly
VALID_TASKS=("pattern_generation" "cue_accumulation" "delayed_match")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "ERROR: Unknown task '$TASK'."
    echo "Valid options: ${VALID_TASKS[*]}"
    exit 1
fi
 
VALID_EXPERIMENTS=("BPTT" "e_prop_hardcoded" "diffusion" "per_step_shuffle_diffusion" "fixed_shuffle_diffusion" "random_eprop" "sparse_recurrent" "diffusion_nn" "fixed_shuffle_diffusion_nn" "align_local_connectivity_BPTT" "align_local_connectivity_eprop" "align_local_connectivity_diffusion_aligned" "align_local_connectivity_diffusion_fixed" "align_local_connectivity_diffusion_per_step" "align_local_connectivity_random_eprop" "align_nn_connectivity_diffusion_fixed" "align_nn_connectivity_diffusion_aligned")
if [[ ! " ${VALID_EXPERIMENTS[*]} " =~ " ${EXPERIMENT} " ]]; then
    echo "ERROR: Unknown experiment '$EXPERIMENT'."
    echo "Valid options: ${VALID_EXPERIMENTS[*]}"
    exit 1
fi
 
# --- Submit one job per seed -------------------------------------------------
echo "Submitting jobs | task='$TASK' | experiment='$EXPERIMENT' | seeds='$SEEDS_FILE'"
 
while IFS= read -r seed || [[ -n "$seed" ]]; do
    seed="${seed//$'\r'/}"   # strip carriage returns
 
    # skip comments and empty lines
    [[ "$seed" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$seed" ]] && continue
 
    echo "  Submitting seed=$seed ..."
    sbatch --export=SEED="$seed",TASK="$TASK",EXPERIMENT="$EXPERIMENT" worker.sh
done < "$SEEDS_FILE"
 
echo "Done. Check queue with: squeue -u \$USER"