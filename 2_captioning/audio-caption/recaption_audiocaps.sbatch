#!/bin/bash
#SBATCH --job-name=ac_recap
#SBATCH --output=logs/ac_recap_%A_%a.out
#SBATCH --error=logs/ac_recap_%A_%a.err
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>

# AudioCaps recaptioning: 98 tar files total split across 32 GPUs
# - train: 0-96 (tar_index 0-96)
# - valid: 0 (tar_index 97)
# Each GPU processes ~3-4 tar files

# Change to the script directory
cd "$(dirname "$0")"

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust path as needed)
eval "$(mamba shell hook --shell bash)"
mamba activate af3

# Use cached model (no network required on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Calculate which tar files this job should process
NUM_GPUS=32
TOTAL_TARS=98
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate start and end tar indices for this job
TARS_PER_GPU=$((TOTAL_TARS / NUM_GPUS))
REMAINDER=$((TOTAL_TARS % NUM_GPUS))

# Distribute remainder across first jobs
if [ $TASK_ID -lt $REMAINDER ]; then
    START_TAR=$((TASK_ID * (TARS_PER_GPU + 1)))
    END_TAR=$((START_TAR + TARS_PER_GPU))
else
    START_TAR=$((REMAINDER * (TARS_PER_GPU + 1) + (TASK_ID - REMAINDER) * TARS_PER_GPU))
    END_TAR=$((START_TAR + TARS_PER_GPU - 1))
fi

echo "=========================================="
echo "AudioCaps Recaptioning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Processing tar indices: ${START_TAR} to ${END_TAR}"
echo "Date: $(date)"
echo "=========================================="

# Process each tar file assigned to this job
for TAR_INDEX in $(seq $START_TAR $END_TAR); do
    echo ""
    echo "=========================================="
    echo "Processing tar_index: ${TAR_INDEX}"
    echo "Time: $(date)"
    echo "=========================================="
    
    python recaption.py \
        --dataset audiocaps \
        --tar_index ${TAR_INDEX} \
        --num_workers 4 \
        --max_new_tokens 500
done

echo "=========================================="
echo "Finished all tars: $(date)"
echo "=========================================="
