#!/bin/bash
# =============================================================================
# worker_grid.sh
# Submitted as a SLURM array job by launcher_grid.sh. Receives via --export:
#   COMBINATIONS_FILE  — path to the combinations file
#   TASK               — task name
#   EXPERIMENT         — experiment YAML name
#   PROJECT_ROOT       — absolute path of the project root (where main.py lives)
# plus SLURM_ARRAY_TASK_ID indicating which line of COMBINATIONS_FILE
# this job handles.
#
# Do not run directly; use launcher_grid.sh instead.
# =============================================================================
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=0-01:30
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joao.barretto-bittar@student.uni-tuebingen.de
 
# --- Validate inputs ---------------------------------------------------------
if [[ -z "${COMBINATIONS_FILE:-}" || -z "${TASK:-}" || -z "${EXPERIMENT:-}" || -z "${PROJECT_ROOT:-}" ]]; then
    echo "ERROR: COMBINATIONS_FILE, TASK, EXPERIMENT, PROJECT_ROOT must all be set." >&2
    exit 1
fi
 
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set — was this launched as an array job?" >&2
    exit 1
fi
 
if [[ ! -f "$COMBINATIONS_FILE" ]]; then
    echo "ERROR: combinations file '$COMBINATIONS_FILE' not found." >&2
    exit 1
fi
 
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT '$PROJECT_ROOT' is not a directory." >&2
    exit 1
fi
 
if [[ ! -f "$PROJECT_ROOT/main.py" ]]; then
    echo "ERROR: main.py not found at $PROJECT_ROOT/main.py" >&2
    exit 1
fi
 
# --- Read this job's line ----------------------------------------------------
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMBINATIONS_FILE")
if [[ -z "$LINE" ]]; then
    echo "ERROR: line ${SLURM_ARRAY_TASK_ID} of $COMBINATIONS_FILE is empty." >&2
    exit 1
fi
 
LR=$(printf '%s' "$LINE" | cut -f1)
C_REG=$(printf '%s' "$LINE" | cut -f2)
SEED=$(printf '%s' "$LINE" | cut -f3)
 
if [[ -z "$LR" || -z "$C_REG" || -z "$SEED" ]]; then
    echo "ERROR: malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'" >&2
    echo "       expected three tab-separated fields: <lr> <c_reg> <seed>" >&2
    exit 1
fi
 
# --- Environment setup -------------------------------------------------------
source $HOME/.bashrc
conda activate /mnt/lustre/work/martius/mot736/.conda/modRNN
 
set -euxo pipefail
 
# Switch to project root so `python main.py` finds main.py and so any
# relative paths inside the project resolve from a known location.
cd "$PROJECT_ROOT"
 
# --- Job info ----------------------------------------------------------------
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "TASK         = $TASK"
echo "EXPERIMENT   = $EXPERIMENT"
echo "LR           = $LR"
echo "C_REG        = $C_REG"
echo "SEED         = $SEED"
echo "ARRAY_ID     = $SLURM_ARRAY_TASK_ID"
echo "---------------------------------"
 
# --- Run --------------------------------------------------------------------
python main.py \
    task=${TASK} \
    +experiment=${EXPERIMENT} \
    train_params.lr=${LR} \
    train_params.c_reg=${C_REG} \
    net_params.seed=${SEED} \
    task.seed=${SEED} \
    save_paths.experiment_name="gridsearch/${EXPERIMENT}/lr_${LR}__creg_${C_REG}" \
    save_paths.condition="seed_${SEED}"
 
conda deactivate


