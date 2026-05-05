#!/bin/bash
#SBATCH --job-name=video_caption_vllm
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=./video_caption/logs/video_caption_vllm_%j.out
#SBATCH --error=./video_caption/logs/video_caption_vllm_%j.err
#SBATCH --partition=
#SBATCH --account=

# Usage:
#   sbatch video_caption_vllm.sbatch.sh <tar_path>
#   sbatch video_caption_vllm.sbatch.sh ./00000.tar
#
# Output will be written to <tar_path>.json (e.g., ./00000.tar.json)

set -euo pipefail
set -x

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="."
VIDEO_CAPTIONING_DIR="${PROJECT_DIR}/2_captioning/video_caption"
SCRIPT_PATH="${VIDEO_CAPTIONING_DIR}/video_caption_vllm.py"
LOG_DIR="${VIDEO_CAPTIONING_DIR}/logs"

# Model configuration
MODEL="Qwen/Qwen3-VL-2B-Instruct"
BATCH_SIZE=4
# MAX_TOKENS=512
# TEMPERATURE=0.7
# MAX_NUM_FRAMES=32
# TENSOR_PARALLEL_SIZE=4
PROMPT="Describe the video in 20 words or less."

# ============================================================================
# Arguments
# ============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <tar_path>"
    echo ""
    echo "Arguments:"
    echo "  tar_path    Path to .tar file or directory containing .tar files"
    echo ""
    echo "Output will be written to <tar_path>.json"
    exit 1
fi

TAR_PATH="$1"
OUTPUT_PATH="${TAR_PATH}.json"

# Validate input path exists
if [ ! -e "$TAR_PATH" ]; then
    echo "ERROR: Path does not exist: $TAR_PATH"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Load required modules
module load Stages/2025
module load GCC/13.3.0
module load CUDA/12
module load Python/3.12.3


export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
# export HF_HOME=""

# Activate virtual environment
cd "$VIDEO_CAPTIONING_DIR"
source .venv/bin/activate

# ============================================================================
# Run Video Captioning
# ============================================================================

echo "============================================"
echo "Video Captioning with vLLM"
echo "============================================"
echo "Input:  $TAR_PATH"
echo "Output: $OUTPUT_PATH"
echo "Model:  $MODEL"
echo "============================================"

python "$SCRIPT_PATH" \
    "$TAR_PATH" \
    --output "$OUTPUT_PATH" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --prompt "$PROMPT" \
    --data-parallel-size 4 \
    --resume

    # --max-tokens "$MAX_TOKENS" \
    # --temperature "$TEMPERATURE" \
    # --max-num-frames "$MAX_NUM_FRAMES" \
    # --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
echo "============================================"
echo "Captioning complete!"
echo "Output saved to: $OUTPUT_PATH"
echo "============================================"

