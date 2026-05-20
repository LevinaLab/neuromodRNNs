#!/bin/bash
# =============================================================================
# launcher.sh
# Usage: bash launcher.sh <seeds_file> <task> <experiment>
#
# Tasks:
#   pattern_generation
#   cue_accumulation              
#   delayed_match               (
#
# Experiments (available for every task):
#   base
#   sparse_recurrent
#   shuffled_diffusion
#
# Examples:
#   bash launcher.sh seeds.txt pattern_generation base
#   bash launcher.sh seeds.txt pattern_generation sparse_recurrent
#
# To launch all combinations at once:
#   for task in pattern_generation cue_accumulation delayed_match; do
#       for exp in base sparse_recurrent shuffled_diffusion; do
#           bash launcher.sh seeds.txt $task $exp
#       done
#   done
# =============================================================================
 
SEEDS_FILE="${1}"
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
 
VALID_TASKS=("pattern_generation" "cue_accumulation" "delayed_match" "long_LS_delayed_match" "long_LS_cue_accumulation")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "ERROR: Unknown task '$TASK'."
    echo "Valid options: ${VALID_TASKS[*]}"
    exit 1
fi
 
VALID_EXPERIMENTS=("BPTT" "e_prop_hardcoded" "diffusion" "per_step_shuffle_diffusion" "fixed_shuffle_diffusion" "sparse_recurrent" "align_local_connectivity_BPTT" "align_local_connectivity_eprop" "align_local_connectivity_diffusion_aligned" "align_local_connectivity_diffusion_fixed" "align_local_connectivity_diffusion_per_step")
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