#!/bin/bash

# Determine index:
# - If running as an array job, SLURM provides SLURM_ARRAY_TASK_ID
# - Otherwise, allow a single positional argument as fallback
START_INDEX=$1
END_INDEX=$2

if [ -z "$START_INDEX" ] || [ -z "$END_INDEX" ]; then
    echo "Usage: $0 <start_index> <end_index>"
    exit 1
fi

cd ./1_curation/clip_extraction

for INDEX in $(seq $START_INDEX $END_INDEX); do
    # Zero-pad the index to 5 characters (0 -> 00000, 1 -> 00001, etc.)
    FOLDER_NAME=$(printf "%05d" "$INDEX")
    echo "Starting job for folder: $FOLDER_NAME"

    sbatch --job-name=scene_detect_cli_$INDEX --dependency=singleton \
        --output=./logs/scene_detect_cli_$INDEX.out \
        --error=./logs/scene_detect_cli_$INDEX.err \
        detect_scenes_cli.sbatch.sh $INDEX
done
