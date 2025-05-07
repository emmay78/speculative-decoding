#!/bin/bash
#SBATCH --job-name=speculative_decode
#SBATCH --partition=seas_gpu,gpu_test
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb

#SBATCH -o /n/netscratch/idreos_lab/Lab/emyang/cs2241/speculative_decoding/logs/%x_%j.out        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/netscratch/idreos_lab/Lab/emyang/cs2241/speculative_decoding/logs/%x_%j.err        # File to which STDERR will be written, %j inserts jobid
#SBATCH --chdir=/n/netscratch/idreos_lab/Lab/emyang/cs2241/speculative_decoding/

# Activate environment
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
module load gcc/12.2.0-fasrc01

HOME_DIR=/n/netscratch/idreos_lab/Lab/emyang
eval "$(conda shell.bash hook)"
conda activate $HOME_DIR/cs2241/pytorch-3.12

# Run the script
srun --output logs/%x_%j_%t.out --error logs/%x_%j_%t.err \
    /n/netscratch/idreos_lab/Lab/emyang/cs2241/speculative_decoding/run_experiments.sh