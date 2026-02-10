#!/bin/bash
#SBATCH --job-name=eur_full_gpu
#SBATCH --output=logs/eur_full_gpu_%j.out
#SBATCH --error=logs/eur_full_gpu_%j.err
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
export INPUT_FILE="/users/rmatovu/proj_GxE/dataset_with_nan.csv"
export OUTPUT_DIR="/users/rmatovu/proj_GxE/results_eur"
mkdir -p /users/rmatovu/proj_GxE/results_eur

# Run the Python script
python eur_full_script.py
