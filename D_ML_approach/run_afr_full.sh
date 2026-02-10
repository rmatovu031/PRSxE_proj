#!/bin/bash
#SBATCH --job-name=afr_full_gpu
#SBATCH --output=logs/afr_full_gpu_%j.out
#SBATCH --error=logs/afr_full_gpu_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

module load anaconda3
module load cuda/12.4.0_550.54.14
# Activate conda env
conda activate gxenv

# Set environment variables
export INPUT_FILE="/users/rmatovu/proj_GxE/new_dataset.csv"
export OUTPUT_DIR="/users/rmatovu/proj_GxE/results_afr/suzuki"
mkdir -p /users/rmatovu/proj_GxE/results_afr/suzuki

# Run the Python script
python afr_full_script.py
