#!/bin/bash
#SBATCH --job-name=pv_recap
#SBATCH --output=logs/pv_recap_%A_%a.out
#SBATCH --error=logs/pv_recap_%A_%a.err
#SBATCH --array=0-45
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>

# prio_video_scenes_cli/00002 recaptioning: 23 tar files (0000.tar to 0022.tar)
# Each tar is split across 2 GPUs (shards), so 46 total tasks across ~12 nodes
# Task 0,1 -> tar 0000 (shards 0,1)
# Task 2,3 -> tar 0001 (shards 0,1)
# etc.
#
# Test results: ~2.05 sec/sample, ~4.5 hours for 8000 samples per GPU
# Time limit set to 6 hours with buffer

NUM_SHARDS=2  # GPUs per tar file

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

# Calculate tar file and shard from array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}
TAR_INDEX=$((TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS))
TAR_NAME=$(printf "%04d.tar" $TAR_INDEX)

echo "=========================================="
echo "prio_video_scenes Recaptioning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Processing tar: ${TAR_NAME} (shard ${SHARD_INDEX}/${NUM_SHARDS})"
echo "Date: $(date)"
echo "=========================================="

OUTPUT_DIR="<YOUR_OUTPUT_DIR>"  # e.g. /path/to/prio_video_scenes_00002_recap
TAR_BASE=$(printf "%04d" $TAR_INDEX)

python recaption_prio_video.py \
    --tar_name ${TAR_NAME} \
    --input_dir <YOUR_INPUT_DIR> \  # e.g. /path/to/prio_video_scenes_cli/00002
    --output_dir ${OUTPUT_DIR} \
    --num_workers 4 \
    --max_new_tokens 500 \
    --shard_index ${SHARD_INDEX} \
    --num_shards ${NUM_SHARDS}

echo "=========================================="
echo "Shard ${SHARD_INDEX} of ${TAR_NAME} finished: $(date)"
echo "=========================================="

# Shard 0 of each tar waits for other shards and then merges
if [ "$SHARD_INDEX" -eq 0 ]; then
    echo ""
    echo "Waiting for other shards of ${TAR_NAME} to complete..."
    
    # Wait for all shard output files for this tar to exist
    for i in $(seq 1 $((NUM_SHARDS - 1))); do
        SHARD_OUTPUT="${OUTPUT_DIR}/${TAR_BASE}_shard$(printf '%02d' $i).tar"
        echo "  Waiting for shard $i: ${SHARD_OUTPUT}"
        while [ ! -f "$SHARD_OUTPUT" ]; do
            sleep 30
        done
        # Wait a bit more to ensure file is fully written
        sleep 10
    done
    
    echo ""
    echo "=========================================="
    echo "All shards of ${TAR_NAME} complete. Merging..."
    echo "=========================================="
    
    # Merge just this tar's shards
    python -c "
import tarfile
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
tar_base = '${TAR_BASE}'
num_shards = ${NUM_SHARDS}

# Collect all shard files for this tar
shard_files = sorted(output_dir.glob(f'{tar_base}_shard*.tar'))
print(f'Merging {len(shard_files)} shards into {tar_base}.tar')

# Merge into single tar
output_path = output_dir / f'{tar_base}.tar'
total_files = 0

with tarfile.open(output_path, 'w') as out_tar:
    for shard_path in shard_files:
        with tarfile.open(shard_path, 'r') as in_tar:
            for member in in_tar.getmembers():
                if member.isfile():
                    f = in_tar.extractfile(member)
                    if f is not None:
                        out_tar.addfile(member, f)
                        total_files += 1

print(f'Merged {total_files} files ({total_files // 2} samples)')

# Update sizes.json
sizes_path = output_dir / 'sizes.json'
if sizes_path.exists():
    with open(sizes_path, 'r') as f:
        sizes_data = json.load(f)
else:
    sizes_data = {}

# Remove shard entries for this tar, add merged entry
for shard_path in shard_files:
    sizes_data.pop(shard_path.name, None)
sizes_data[f'{tar_base}.tar'] = total_files // 2

with open(sizes_path, 'w') as f:
    json.dump(sizes_data, f, indent=4, sort_keys=True)

# Delete shard files
for shard_path in shard_files:
    print(f'Deleting: {shard_path.name}')
    shard_path.unlink()

print('Merge complete!')
"
    
    echo ""
    echo "=========================================="
    echo "Merge of ${TAR_NAME} complete: $(date)"
    echo "=========================================="
fi

