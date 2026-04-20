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
 
VALID_TASKS=("pattern_generation" "cue_accumulation" "delayed_match")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "ERROR: Unknown task '$TASK'."
    echo "Valid options: ${VALID_TASKS[*]}"
    exit 1
fi
 
VALID_EXPERIMENTS=("base" "sparse_recurrent" "shuffled_diffusion")
if [[ ! " ${VALID_EXPERIMENTS[*]} " =~ " ${EXPERIMENT} " ]]; then
    echo "ERROR: Unknown experiment '$EXPERIMENT'."
    echo "Valid options: ${VALID_EXPERIMENTS[*]}"
    exit 1
fi
 
# --- Submit one job per seed -------------------------------------------------
echo "Submitting jobs | task='$TASK' | experiment='$EXPERIMENT' | seeds='$SEEDS_FILE'"
 
while IFS= read -r seed || [[ -n "$seed" ]]; do
    seed="${seed//$'\r'/}"   # strip carriage returns
    echo "  Submitting seed=$seed ..."
    sbatch --export=SEED="$seed",TASK="$TASK",EXPERIMENT="$EXPERIMENT" worker.sh
done < "$SEEDS_FILE"
 
echo "Done. Check queue with: squeue -u \$USER"