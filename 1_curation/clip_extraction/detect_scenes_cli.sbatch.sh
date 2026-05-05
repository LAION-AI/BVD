#!/bin/bash
#SBATCH --job-name=scene_detect_cli
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=
#SBATCH --account=
# Array usage: submit with --array=START-END (e.g., --array=0-99)


set -euo pipefail
set -x

# Determine index:
# - If running as an array job, SLURM provides SLURM_ARRAY_TASK_ID
# - Otherwise, allow a single positional argument as fallback
INDEX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"

# Zero-pad the index to 5 characters (0 -> 00000, 1 -> 00001, etc.)
FOLDER_NAME=$(printf "%05d" "$INDEX")

# ------ USER EDITABLE PATHS ----------------------------------------------------------
INPUT_BASE="/path/to/videos"
OUTPUT_BASE="/output/path/for/scenes"
PROJECT_ROOT="/path/to/project/root"
SIF_FILE="./scene_detect.sif"
# ------

SCENE_DETECT_DIR="${PROJECT_ROOT}/1_curation/clip_extraction"
SCRIPT_PATH="${SCENE_DETECT_DIR}/detect_scenes_and_split_cli.py"

INPUT_FOLDER="${INPUT_BASE}/${FOLDER_NAME}"
OUTPUT_FOLDER="${OUTPUT_BASE}"

# ------ Singularity setup ------------------------------------------------------------
DEF_FILE="${SCENE_DETECT_DIR}/scene_detect.def"

# Build the image from the definition file if it doesn't exist
if [ ! -f "$SIF_FILE" ]; then
    echo "Building Singularity image from definition file..."
    cd "$SCENE_DETECT_DIR"
    singularity build --fakeroot "$SIF_FILE" "$DEF_FILE"
fi
# ------------------------------------------------------------

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "ERROR: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

echo "Processing folder: $FOLDER_NAME"
echo "Input: $INPUT_FOLDER"
echo "Output: $OUTPUT_FOLDER"

singularity exec \
    --bind "$INPUT_BASE:$INPUT_BASE:ro" \
    --bind "$OUTPUT_BASE:$OUTPUT_BASE:rw" \
    --bind "$SCENE_DETECT_DIR:$SCENE_DETECT_DIR:ro" \
    --pwd /app \
    "$SIF_FILE" \
    /venv/bin/python "$SCRIPT_PATH" \
    "$INPUT_FOLDER" \
    "$OUTPUT_FOLDER" \
    --threshold 30.0 \
    --pattern "*.360p.mp4" \
    --tar \
    --cutoff-time 84600 # 23.5 hours

