#!/bin/bash
# =============================================================================
# worker.sh
# Submitted by launcher.sh. Receives SEED, TASK, EXPERIMENT via --export.
# Do not run directly; use launcher.sh instead.
#
# Configuration is composed by Hydra from:
#   conf/task/<task>.yaml         — task-specific hyperparameters
#   conf/experiment/<exp>.yaml    — experiment-specific hyperparameters
#   CLI overrides below           — per-job parameters (just seeds for now)
# =============================================================================

# SLURM Configs
#SBATCH --ntasks=1
#...
 
# --- Validate inputs ---------------------------------------------------------
if [[ -z "$SEED" || -z "$TASK" || -z "$EXPERIMENT" ]]; then
    echo "ERROR: SEED, TASK, and EXPERIMENT must all be set." >&2
    exit 1
fi
 
# --- Environment setup -------------------------------------------------------

# Activate environment 
 
# --- Job info ----------------------------------------------------------------
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo "TASK       = $TASK"
echo "EXPERIMENT = $EXPERIMENT"
echo "SEED       = $SEED"
echo "---------------------------------"
 
# --- Run --------------------------------------------------------------------
# All task and experiment configuration lives in YAML now. The only per-job
# overrides are the seed and the run-naming fields.
python main.py \
    task=${TASK} \
    +experiment=${EXPERIMENT} \
    net_params.seed=${SEED} \
    task.seed=${SEED} \
    save_paths.experiment_name="${EXPERIMENT}" \
    save_paths.condition="seed_${SEED}"
 
# If necessary, deactivate environment