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
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=0-02:30
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --open-mode=append
#SBATCH --output=logs/job_%j_seed_%x.out
#SBATCH --error=logs/job_%j_seed_%x.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joao.barretto-bittar@student.uni-tuebingen.de
 
# --- Validate inputs ---------------------------------------------------------
if [[ -z "$SEED" || -z "$TASK" || -z "$EXPERIMENT" ]]; then
    echo "ERROR: SEED, TASK, and EXPERIMENT must all be set." >&2
    exit 1
fi
 
# --- Environment setup -------------------------------------------------------
source $HOME/.bashrc
conda activate /mnt/lustre/work/martius/mot736/.conda/modRNN
 
set -x
 
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
    save_paths.experiment_name="${TASK}_${EXPERIMENT}" \
    save_paths.condition="seed_${SEED}"
 
conda deactivate