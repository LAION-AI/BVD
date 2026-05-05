#!/bin/bash
#SBATCH --job-name=pv_merge
#SBATCH --output=logs/pv_merge_%j.out
#SBATCH --error=logs/pv_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>

# Merge shard tar files into single tar files
# Run this AFTER all recaption jobs have completed
#
# Usage:
#   sbatch merge_shards.sbatch
#   OR submit with dependency:
#   sbatch --dependency=afterok:<recap_job_id> merge_shards.sbatch

# Change to the script directory
cd "$(dirname "$0")"

# Create logs directory
mkdir -p logs

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate af3

OUTPUT_DIR="<YOUR_OUTPUT_DIR>"  # e.g. /path/to/prio_video_scenes_00002_recap

echo "=========================================="
echo "Merging Shard Files"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "=========================================="

python merge_shards.py \
    --input_dir ${OUTPUT_DIR} \
    --delete_shards

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="

