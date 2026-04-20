#!/bin/bash
# =============================================================================
# worker.sh
# Submitted by launcher.sh. Receives SEED, TASK, EXPERIMENT via --export.
# Do not run directly; use launcher.sh instead.
#
# Config is assembled in three layers:
#   1. TASK_ARGS    — all shared args for a given task
#   2. EXP_ARGS     — what differs between experiments (within any task)
#   3. SEED_ARGS    — seed-dependent overrides
# =============================================================================
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=0-02:30
#SBATCH --partition=2080-galvani 
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --open-mode=append
#SBATCH --output=logs/job_%j_seed_%x.out   # %j=jobID, %x=job-name
#SBATCH --error=logs/job_%j_seed_%x.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joao.barretto-bittar@student.uni-tuebingen.de   # Email to which notifications will be sent
 

# --- Validate inputs ---------------------------------------------------------
if [[ -z "$SEED" || -z "$TASK" || -z "$EXPERIMENT" ]]; then
    echo "ERROR: SEED, TASK, and EXPERIMENT must all be set."
    exit 1
fi
 
# --- Environment setup -------------------------------------------------------
source $HOME/.bashrc
conda activate /mnt/lustre/work/martius/mot736/.conda/modRNN
 
# --- Job info ----------------------------------------------------------------
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo "TASK       = $TASK"
echo "EXPERIMENT = $EXPERIMENT"
echo "SEED       = $SEED"
echo "---------------------------------"
 
# =============================================================================
# LAYER 1 — Task args: all shared args for this task
# =============================================================================
case "$TASK" in
    pattern_generation)
        TASK_ARGS=(
            task="pattern_generation"
            task.trial_dur=2000
            task.f_input=50
            net_arch.n_neurons_channel=100
            net_arch.n_ALIF=0
            net_arch.n_LIF=400
            net_arch.gridshape=[20,20]
            net_arch.n_out=1
            net_arch.sparse_input=True
            net_arch.sparse_readout=True
            net_params.tau_m=30
            net_params.tau_out=30
            net_params.refractory_period=2
            net_params.k=0.75
            net_params.w_init_gain=[1.0,1.0,1.0,1.0]
            train_params.train_batch_size=8
            train_params.train_mini_batch_size=8
            train_params.test_batch_size=8
            train_params.test_mini_batch_size=8
            train_params.c_reg=0.01
            train_params.lr=0.0100
            train_params.iterations=2000
            train_params.stop_criteria=0.001
        )
        ;;

    cue_accumulation)
        TASK_ARGS=(
            task="cue_accumulation"
            task.p=0.5
            task.min_delay=1000
            task.max_delay=1001
            net_arch.sparse_input=True 
            net_arch.sparse_readout=True 
            net_params.k=0.75 
            train_params.lr=0.005 
            train_params.iterations=2000 
            train_params.c_reg=0.005
        )
        ;;

    delayed_match)
        TASK_ARGS=(
            task="delayed_match"
            task.LS_avail=50
            task.cue_delay_time=700
            task.decision_delay=0
            net_arch.sparse_readout=True
            net_arch.n_neurons_channel=20
            net_params.k=0.75
            net_params.tau_adaptation=1400
            net_params.w_init_gain=[0.5,0.1,0.5,0.5]
            train_params.lr=0.0050
            train_params.iterations=2000
            train_params.c_reg=0.01
        )
        ;;
    *)
        echo "ERROR: Unknown task '$TASK'."
        exit 1
        ;;
esac

 


# =============================================================================
# LAYER 2 — Experiment args: what differs between experiments (any task)
# =============================================================================
case "$EXPERIMENT" in
    base)
        EXP_ARGS=(
            save_paths.experiment_name="${TASK}_BPTT"
            net_arch.connectivity_rec_layer="local"
            train_params.learning_rule="BPTT"
        )
        ;;
    sparse_recurrent)
        EXP_ARGS=(
            save_paths.experiment_name="${TASK}_Diffusion_sparse_recurrent"
            net_arch.connectivity_rec_layer="sparse"
            train_params.learning_rule="diffusion"
        )
        ;;
    shuffled_diffusion)
        EXP_ARGS=(
            save_paths.experiment_name="${TASK}_Diffusion_shuffled"
            net_arch.connectivity_rec_layer="local"
            train_params.shuffle=True
            train_params.learning_rule="diffusion"
        )
        ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'."
        exit 1
        ;;
esac
 
# =============================================================================
# LAYER 3 — Seed args
# =============================================================================
SEED_ARGS=(
    save_paths.condition="seed_${SEED}"
    net_params.seed=${SEED}
    task.seed=${SEED}
)
 
# =============================================================================
# Run
# =============================================================================
python main.py "${TASK_ARGS[@]}" "${EXP_ARGS[@]}" "${SEED_ARGS[@]}"
 
conda deactivate
 