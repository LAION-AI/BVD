#!/usr/bin/env python
"""
Recaption audio datasets (Clotho, AudioCaps) using Audio Flamingo.
Loads audio from tar, generates new captions, saves to output directory.
"""
import os
import io
import json
import tarfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor


# Dataset configurations
DATASET_CONFIGS = {
    "clotho": {
        "input_dir": "./Clotho",
        "output_dir": "./Clotho_recap_af_short_10",
        "tar_mapping": {
            # tar_index -> (split, tar_num)
            # 0-7: train, 8-10: valid, 11-13: test
            "train": (0, 8),    # indices 0-7
            "valid": (8, 11),   # indices 8-10
            "test": (11, 14),   # indices 11-13
        },
        "total_tars": 14,
    },
    "audiocaps": {
        "input_dir": "./audiocaps",
        "output_dir": "./audiocaps_recap_af_short_10",
        "tar_mapping": {
            # 0-96: train, 97: valid
            "train": (0, 97),   # indices 0-96
            "valid": (97, 98),  # index 97
        },
        "total_tars": 98,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recaption audio datasets using Audio Flamingo"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process (clotho or audiocaps)",
    )
    parser.add_argument(
        "--tar_index",
        type=int,
        required=True,
        help="Index of the tar file to process",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Base directory containing dataset (default: dataset-specific)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for output (default: dataset-specific)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/audio-flamingo-3-hf",
        help="HuggingFace model ID for Audio Flamingo",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing). If None, process all.",
    )
    return parser.parse_args()


def get_tar_info(tar_index: int, dataset: str) -> Tuple[str, int]:
    """Map tar index to split and tar number based on dataset configuration."""
    config = DATASET_CONFIGS[dataset]
    tar_mapping = config["tar_mapping"]
    
    for split, (start, end) in tar_mapping.items():
        if start <= tar_index < end:
            return split, tar_index - start
    
    raise ValueError(
        f"Invalid tar_index {tar_index} for {dataset}. "
        f"Must be 0-{config['total_tars'] - 1}."
    )


class TarAudioDataset(Dataset):
    """Dataset for loading audio samples from a tar file."""
    
    def __init__(self, tar_path: str, cache_dir: Optional[str] = None, max_samples: Optional[int] = None):
        """
        Args:
            tar_path: Path to the tar file containing audio and json files
            cache_dir: Directory to cache extracted audio files. If None, uses temp directory.
            max_samples: Maximum number of samples to load. If None, load all.
        """
        self.tar_path = tar_path
        self.max_samples = max_samples
        
        # Set up cache directory for extracted audio files
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="audio_cache_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all samples from tar
        self.samples = self._load_tar_samples()
        
        # Limit samples if max_samples is set
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {tar_path}")
    
    def _load_tar_samples(self) -> list:
        """Load all samples from the tar file."""
        samples = {}
        
        with tarfile.open(self.tar_path, 'r') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                
                path = Path(member.name)
                key = path.stem
                ext = path.suffix.lower()
                
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()
                
                if key not in samples:
                    samples[key] = {'key': key, 'path_prefix': str(path.parent)}
                
                if ext in ['.flac', '.wav', '.mp3']:
                    samples[key]['audio_bytes'] = content
                    samples[key]['audio_ext'] = ext
                elif ext == '.json':
                    samples[key]['json_data'] = json.loads(content.decode('utf-8'))
        
        # Filter to only complete samples
        complete_samples = [
            s for s in samples.values() 
            if 'audio_bytes' in s and 'json_data' in s
        ]
        
        return complete_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample, extracting audio to cache if needed."""
        sample = self.samples[idx]
        key = sample['key']
        ext = sample['audio_ext']
        
        # Extract audio to cache file if not already done
        audio_path = self.cache_dir / f"{key}{ext}"
        if not audio_path.exists():
            with open(audio_path, 'wb') as f:
                f.write(sample['audio_bytes'])
        
        return {
            'idx': idx,
            'key': key,
            'audio_path': str(audio_path),
            'json_data': sample['json_data'],
            'path_prefix': sample['path_prefix'],
            'audio_bytes': sample['audio_bytes'],
            'audio_ext': ext,
        }
    
    def update_sample(self, idx: int, json_data: dict):
        """Update the JSON data for a sample."""
        self.samples[idx]['json_data'] = json_data
    
    def cleanup_cache(self):
        """Remove cached audio files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def collate_fn(batch):
    """Custom collate function that keeps samples as a list of dicts."""
    return batch


def generate_caption(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int = 500
) -> str:
    """Generate a caption for an audio file using Audio Flamingo."""
    inputs = processor.apply_transcription_request(
        audio=audio_path,
        prompt=prompt
    ).to(model.device)
    
    # Cast float tensors to bfloat16 to match model dtype
    for key in inputs:
        if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
            inputs[key] = inputs[key].to(torch.bfloat16)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    
    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        strip_prefix=True
    )
    
    return decoded[0] if decoded else ""


def create_output_tar(
    samples: list,
    output_path: str,
    path_prefix: str = None
):
    """Create a new tar file with the updated samples.
    
    Args:
        samples: List of sample dicts with 'key', 'audio_bytes', 'json_data', 'audio_ext'
        output_path: Path for the output tar file
        path_prefix: Optional prefix for paths inside the tar
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tarfile.open(output_path, 'w') as tar:
        for sample in samples:
            key = sample['key']
            prefix = sample.get('path_prefix', path_prefix or '')
            
            # Add audio file
            audio_ext = sample['audio_ext']
            audio_name = f"{prefix}/{key}{audio_ext}" if prefix else f"{key}{audio_ext}"
            audio_data = sample['audio_bytes']
            audio_info = tarfile.TarInfo(name=audio_name)
            audio_info.size = len(audio_data)
            tar.addfile(audio_info, io.BytesIO(audio_data))
            
            # Add JSON file
            json_name = f"{prefix}/{key}.json" if prefix else f"{key}.json"
            json_data = json.dumps(sample['json_data'], indent=2, ensure_ascii=False).encode('utf-8')
            json_info = tarfile.TarInfo(name=json_name)
            json_info.size = len(json_data)
            tar.addfile(json_info, io.BytesIO(json_data))


def main():
    args = parse_args()
    
    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    
    # Determine which tar file to process
    split, tar_num = get_tar_info(args.tar_index, args.dataset)
    
    # Configuration - use paths relative to script location
    script_dir = Path(__file__).parent
    
    input_dir = args.input_dir or config["input_dir"]
    output_dir = args.output_dir or config["output_dir"]
    
    input_base = Path(input_dir) if Path(input_dir).is_absolute() else script_dir / input_dir
    output_base = Path(output_dir) if Path(output_dir).is_absolute() else script_dir / output_dir
    
    input_tar = input_base / split / f"{tar_num}.tar"
    output_split_dir = output_base / split
    output_tar = output_split_dir / f"{tar_num}.tar"
    cache_dir = output_split_dir / f"audio_cache_{tar_num}"
    
    model_id = args.model_id
    num_workers = args.num_workers
    max_new_tokens = args.max_new_tokens
    
    # DataLoader settings
    batch_size = 1  # Audio Flamingo processes one at a time
    prefetch_factor = 2  # Number of batches to prefetch per worker
    
    #prompt = "Generate a short caption describing the sounds in the input audio."
    prompt = "Describe the audio sounds in 10 words or less."
    #prompt = "Generate a detailed caption for the input audio, describing all notable speech, sound, and musical events comprehensively. In the caption, transcribe all spoken content by all speakers in the audio precisely."
    
    
    print(f"Dataset: {args.dataset}")
    print(f"Tar index: {args.tar_index} -> {split}/{tar_num}.tar")
    print(f"Input tar: {input_tar}")
    print(f"Output tar: {output_tar}")
    print(f"Model: {model_id}")
    print(f"Num workers: {num_workers}")
    print()
    
    # Create dataset and dataloader
    print("Loading samples from tar file...")
    dataset = TarAudioDataset(str(input_tar), cache_dir=str(cache_dir), max_samples=args.max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Load model
    print(f"\nLoading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")
    
    # Process samples using dataloader
    print(f"\nProcessing {len(dataset)} samples...")
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Recaptioning", total=len(dataloader)):
                for sample in batch:
                    idx = sample['idx']
                    audio_path = sample['audio_path']
                    json_data = sample['json_data'].copy()
                    
                    try:
                        # Generate new caption
                        new_caption = generate_caption(
                            model, processor, audio_path, prompt, max_new_tokens
                        )
                        
                        # Update JSON data with new caption (as list to match original format)
                        json_data['text'] = [new_caption]
                        json_data['recaption_timestamp'] = datetime.now().isoformat()
                        json_data['recaption_model'] = model_id
                        json_data['recaption_prompt'] = prompt
                        
                    except Exception as e:
                        print(f"\nError processing {sample['key']}: {e}")
                        json_data['text'] = []
                        json_data['recaption_error'] = str(e)
                        json_data['recaption_timestamp'] = datetime.now().isoformat()
                    
                    # Update the dataset's sample
                    dataset.update_sample(idx, json_data)
    
    finally:
        # Clean up cached audio files
        print("\nCleaning up cache...")
        dataset.cleanup_cache()
    
    # Prepare samples for output
    output_samples = [
        {
            'key': s['key'],
            'audio_bytes': s['audio_bytes'],
            'audio_ext': s['audio_ext'],
            'json_data': s['json_data'],
            'path_prefix': s['path_prefix'],
        }
        for s in dataset.samples
    ]
    
    # Save output tar
    print(f"\nSaving output tar to: {output_tar}")
    create_output_tar(output_samples, str(output_tar))
    
    # Update sizes.json (append/update entry for this tar)
    sizes_path = output_split_dir / "sizes.json"
    if sizes_path.exists():
        with open(sizes_path, 'r') as f:
            sizes_data = json.load(f)
    else:
        sizes_data = {}
    sizes_data[f"{tar_num}.tar"] = len(output_samples)
    with open(sizes_path, 'w') as f:
        json.dump(sizes_data, f, indent=4)
    print(f"Updated sizes.json: {sizes_path}")
    
    print(f"\nCompleted! Processed {len(output_samples)} samples.")
    print(f"Output saved to: {output_tar}")


if __name__ == "__main__":
    main()
