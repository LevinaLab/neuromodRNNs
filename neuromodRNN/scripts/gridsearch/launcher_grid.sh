#!/bin/bash
# =============================================================================
# launcher_grid.sh
# Usage: bash launcher_grid.sh <combinations_file> <task> <experiment>
#
# Submits one SLURM array job, with one array task per line of
# <combinations_file>. Concurrency is capped at MAX_CONCURRENT (default
# 20) — submission queues all jobs, but only the cap runs at once.
#
# Each line of <combinations_file> is three tab-separated fields:
#     <lr>    <c_reg>    <seed>
# Generate this file with `generate_combinations.py`.
#
# Tasks:
#   pattern_generation
#   cue_accumulation
#   delayed_match
#
# Experiments: any YAML in conf/experiment/, e.g.,
#   e_prop_hardcoded
#   diffusion
#   per_step_shuffle_diffusion
#   fixed_shuffle_diffusion
#
# Examples:
#   bash launcher_grid.sh combinations/combinations_cue.txt cue_accumulation e_prop_hardcoded
#   bash launcher_grid.sh combinations/combinations_cue.txt cue_accumulation diffusion
#
# Output: outputs/<task>/gridsearch/<experiment>/lr_<lr>__creg_<c_reg>/seed_<seed>/
# =============================================================================
 
set -euo pipefail
 
# Resolve where this script lives, and from there the project root.
# Assumption: this script lives at <project_root>/scripts/gridsearch/.
# If you move it, adjust PROJECT_ROOT accordingly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
 
COMBINATIONS_FILE="${1:-}"
TASK="${2:-}"
EXPERIMENT="${3:-}"
 
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
 
# --- Validate arguments ------------------------------------------------------
if [[ -z "$COMBINATIONS_FILE" || -z "$TASK" || -z "$EXPERIMENT" ]]; then
    echo "Usage: bash launcher_grid.sh <combinations_file> <task> <experiment>"
    exit 1
fi
 
if [[ ! -f "$COMBINATIONS_FILE" ]]; then
    echo "ERROR: combinations file '$COMBINATIONS_FILE' not found."
    exit 1
fi
 
VALID_TASKS=("pattern_generation" "cue_accumulation" "delayed_match")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "ERROR: unknown task '$TASK'."
    echo "Valid options: ${VALID_TASKS[*]}"
    exit 1
fi
 
# Skip blank/comment lines for the count and the array bound.
N_LINES=$(grep -cv -E '^[[:space:]]*(#|$)' "$COMBINATIONS_FILE" || true)
if [[ "$N_LINES" -eq 0 ]]; then
    echo "ERROR: combinations file '$COMBINATIONS_FILE' has no usable lines."
    exit 1
fi
 
TOTAL_LINES=$(wc -l < "$COMBINATIONS_FILE")
if [[ "$N_LINES" -ne "$TOTAL_LINES" ]]; then
    echo "ERROR: file has $TOTAL_LINES lines but only $N_LINES are usable"
    echo "       (others are blank or commented). Comments/blanks confuse"
    echo "       SLURM array indexing — please remove them and re-run."
    exit 1
fi
 
# --- Submit ------------------------------------------------------------------
COMBINATIONS_FILE_ABS=$(readlink -f "$COMBINATIONS_FILE")
LOG_DIR="$PROJECT_ROOT/logs/gridsearch/${TASK}/${EXPERIMENT}"
mkdir -p "$LOG_DIR"
 
# cd to project root so that relative paths in any downstream script
# (worker_grid.sh, main.py) resolve from a known location.
cd "$PROJECT_ROOT"
 
WORKER="$SCRIPT_DIR/worker_grid.sh"
if [[ ! -f "$WORKER" ]]; then
    echo "ERROR: worker script not found at $WORKER"
    exit 1
fi
 
echo "Submitting array job"
echo "  project root  = $PROJECT_ROOT"
echo "  task          = $TASK"
echo "  experiment    = $EXPERIMENT"
echo "  combinations  = $COMBINATIONS_FILE_ABS"
echo "  total jobs    = $N_LINES"
echo "  concurrency   = $MAX_CONCURRENT"
echo "  logs          = $LOG_DIR/"
 
sbatch \
    --job-name="gs_${TASK}_${EXPERIMENT}" \
    --array="1-${N_LINES}%${MAX_CONCURRENT}" \
    --output="${LOG_DIR}/job_%A_%a.out" \
    --error="${LOG_DIR}/job_%A_%a.err" \
    --export=COMBINATIONS_FILE="$COMBINATIONS_FILE_ABS",TASK="$TASK",EXPERIMENT="$EXPERIMENT",PROJECT_ROOT="$PROJECT_ROOT" \
    "$WORKER"
 
echo "Done. Check queue with: squeue -u \$USER"
echo "Cancel array with:      scancel <jobid>   (one ID for the whole array)"