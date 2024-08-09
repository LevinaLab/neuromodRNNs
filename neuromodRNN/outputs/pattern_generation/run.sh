#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-10:00            # Runtime in D-HH:MM
#SBATCH --partition=2080-galvani    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=16G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --open-mode=append        # update the output file periodically (?)
#SBATCH --output=logs/hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/datahostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=joao.barretto-bittar@student.uni-tuebingen.de   # Email to which notifications will be sent

# some bug
source $HOME/.bashrc

conda activate /mnt/qb/work/martius/mot736/.conda/modRNN
# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo "---------------------------------"



# insert your commands here
# takes the same arguments as train.py but instead of number of runs you need to label the run number manually with -n
# e.g. python train.py -c cumulative -t parity -s 0 -n 0
python main.py task="pattern_generation" save_paths.experiment_name="diffusion_test" save_paths.condition="e_prop_c_reg_100" net_params.seed=1 net_arch.gridshape=[20,20] train_params.stop_criteria=0.1 net_arch.n_neurons_channel=100 net_arch.n_ALIF=0 net_arch.n_LIF=400 net_arch.n_out=1 net_params.tau_m=30 net_params.tau_out=30 net_params.refractory_period=2 train_params.c_reg=100 train_params.train_sub_batch_size=32 train_params.test_sub_batch_size=32 train_params.test_batch_size=128 train_params.iterations=500 task.trial_dur=2000











conda deactivate
