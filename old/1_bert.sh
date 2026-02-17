#!/bin/bash
#SBATCH --job-name=bert_train
#SBATCH --account=project_2011109
#SBATCH --partition=gpusmall
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/bert_train_%j.out
#SBATCH --error=logs/bert_train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Disable wandb
export WANDB_DISABLED=true

# Run the script
python3 1_bert.py

echo "Job completed at $(date)"