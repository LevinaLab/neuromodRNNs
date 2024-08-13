#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-03:00            # Runtime in D-HH:MM
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


python main.py save_paths.experiment_name="test_BPTT" save_paths.condition="BPTT" train_params.learning_rule="BPTT"

python main.py save_paths.experiment_name="test_BPTT" save_paths.condition="test_BPTT_full_connectivity" train_params.learning_rule="BPTT" net_arch.local_connectivity=False




conda deactivate
