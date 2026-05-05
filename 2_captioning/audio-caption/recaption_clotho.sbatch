#!/bin/bash
#SBATCH --job-name=cl_recap
#SBATCH --output=logs/cl_recap_%A_%a.out
#SBATCH --error=logs/cl_recap_%A_%a.err
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>

# Process all Clotho train (0-7) and valid (8-10) tar files
# 11 jobs total, one per GPU

# Change to the script directory
cd "$(dirname "$0")"

# Create logs directory
mkdir -p logs

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate af3

# Use cached model (no network required on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "Clotho Recaptioning - Full Run"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Date: $(date)"
echo "=========================================="

# Run the recaptioning script for this tar file (all samples)
python recaption.py \
    --dataset clotho \
    --tar_index ${SLURM_ARRAY_TASK_ID} \
    --num_workers 4 \
    --max_new_tokens 500

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
