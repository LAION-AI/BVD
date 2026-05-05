# Audio Captioning with Audio Flamingo 3

Recaption audio datasets (Clotho, AudioCaps) using [NVIDIA Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf).

## Overview

This project provides scripts to:
- Generate detailed audio captions using Audio Flamingo 3
- Process tar-based audio datasets (WebDataset format)
- Run distributed captioning jobs on SLURM clusters

## Setup

### Environment

```bash
mamba env create -f environment_cleaned.yml
mamba activate af3
```

## Scripts

### `recaption.py`

Unified recaptioning script for both Clotho and AudioCaps datasets.

```bash
# AudioCaps
python recaption.py \
    --dataset audiocaps \
    --tar_index 0 \
    --num_workers 4 \
    --max_new_tokens 500

# Clotho
python recaption.py \
    --dataset clotho \
    --tar_index 0 \
    --num_workers 4 \
    --max_new_tokens 500
```

**Tar index mapping:**

| Dataset | Split | Tar Indices |
|---------|-------|-------------|
| AudioCaps | train | 0-96 |
| AudioCaps | valid | 97 |
| Clotho | train | 0-7 |
| Clotho | valid | 8-10 |
| Clotho | test | 11-13 |

### `af_inference.py`

General-purpose inference on a directory of audio files.

```bash
python af_inference.py \
    --audio_dir /path/to/audio/files \
    --output_file results.json \
    --prompt "Describe this audio in detail."
```

### `audio_dataset.py`

PyTorch Dataset/DataLoader utilities for audio files with automatic format conversion (m4a → wav).

## SLURM Usage

### AudioCaps (98 tar files on 32 GPUs)

```bash
sbatch recaption_audiocaps.sbatch
```

Each GPU processes 3-4 tar files. Total runtime ~6 hours.

### Clotho (14 tar files)

```bash
sbatch recaption_clotho.sbatch
```

## Dataset Structure

### Input (WebDataset tar format)

```
audiocaps/
├── train/
│   ├── 0.tar
│   ├── 1.tar
│   └── ...
├── valid/
│   └── 0.tar
└── test/
    └── ...
```

Each tar contains paired files:
```
sample_id.flac   # Audio file
sample_id.json   # Metadata: {"text": ["original caption"]}
```

### Output

```
audiocaps_recap_af/
├── train/
│   ├── 0.tar
│   ├── sizes.json
│   └── ...
└── valid/
    └── ...
```

Output JSON format:
```json
{
  "text": ["A dog barks loudly while birds chirp in the background..."],
  "recaption_timestamp": "2025-12-20T04:30:25.226689",
  "recaption_model": "nvidia/audio-flamingo-3-hf",
  "recaption_prompt": "Generate a detailed caption..."
}
```

## Captioning Prompt

Prompt used for recaptioning (hardcoded in `recaption.py`):

> Generate a brief caption of around 10 words describing the main sounds in this audio.

