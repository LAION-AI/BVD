#!/usr/bin/env python
"""
Recaption prio_video_scenes audio data using Audio Flamingo.
Loads audio (.m4a) from tar, generates new captions, saves to output directory.
"""
import os
import io
import json
import tarfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recaption prio_video_scenes audio data using Audio Flamingo"
    )
    parser.add_argument(
        "--tar_name",
        type=str,
        required=True,
        help="Name of the tar file to process (e.g., 0000.tar)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input tar files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output tar files",
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
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Shard index for multi-GPU processing (0-indexed). If None, process all samples.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU processing. Required if --shard_index is set.",
    )
    return parser.parse_args()


class PrioVideoTarDataset(Dataset):
    """Dataset for loading audio samples from prio_video_scenes tar files."""
    
    def __init__(
        self, 
        tar_path: str, 
        cache_dir: Optional[str] = None, 
        max_samples: Optional[int] = None,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ):
        """
        Args:
            tar_path: Path to the tar file containing audio and metadata files
            cache_dir: Directory to cache extracted audio files. If None, uses temp directory.
            max_samples: Maximum number of samples to load. If None, load all.
            shard_index: Shard index for multi-GPU processing (0-indexed).
            num_shards: Total number of shards for multi-GPU processing.
        """
        self.tar_path = tar_path
        self.max_samples = max_samples
        self.shard_index = shard_index
        self.num_shards = num_shards
        
        # Set up cache directory for extracted audio files
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="audio_cache_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all samples from tar
        self.samples = self._load_tar_samples()
        
        # Apply sharding if specified (before max_samples limit)
        if self.shard_index is not None and self.num_shards is not None:
            total_samples = len(self.samples)
            samples_per_shard = total_samples // self.num_shards
            remainder = total_samples % self.num_shards
            
            # Calculate start and end indices for this shard
            # Distribute remainder across first shards
            if self.shard_index < remainder:
                start_idx = self.shard_index * (samples_per_shard + 1)
                end_idx = start_idx + samples_per_shard + 1
            else:
                start_idx = remainder * (samples_per_shard + 1) + (self.shard_index - remainder) * samples_per_shard
                end_idx = start_idx + samples_per_shard
            
            self.samples = self.samples[start_idx:end_idx]
            print(f"Shard {self.shard_index}/{self.num_shards}: samples {start_idx}-{end_idx} ({len(self.samples)} samples)")
        
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
                name = path.name
                
                # Parse filename to get key and type
                # Filenames are like: 000000.m4a, 000000.metadata.json, 000000.info.json
                parts = name.split('.')
                if len(parts) < 2:
                    continue
                
                key = parts[0]  # e.g., "000000"
                
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()
                
                if key not in samples:
                    samples[key] = {
                        'key': key,
                        'path_prefix': str(path.parent),  # e.g., "0000"
                    }
                
                if name.endswith('.m4a'):
                    samples[key]['audio_bytes'] = content
                    samples[key]['audio_ext'] = '.m4a'
                elif name.endswith('.metadata.json'):
                    samples[key]['metadata'] = json.loads(content.decode('utf-8'))
                elif name.endswith('.info.json'):
                    samples[key]['info'] = json.loads(content.decode('utf-8'))
        
        # Filter to only samples with audio
        complete_samples = [
            s for s in samples.values() 
            if 'audio_bytes' in s
        ]
        
        # Sort by key for consistent ordering
        complete_samples.sort(key=lambda x: x['key'])
        
        return complete_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample, extracting audio to cache if needed."""
        sample = self.samples[idx]
        key = sample['key']
        ext = sample.get('audio_ext', '.m4a')
        
        # Extract audio to cache file if not already done
        audio_path = self.cache_dir / f"{key}{ext}"
        if not audio_path.exists():
            with open(audio_path, 'wb') as f:
                f.write(sample['audio_bytes'])
        
        return {
            'idx': idx,
            'key': key,
            'audio_path': str(audio_path),
            'metadata': sample.get('metadata', {}),
            'info': sample.get('info', {}),
            'path_prefix': sample['path_prefix'],
            'audio_bytes': sample['audio_bytes'],
            'audio_ext': ext,
        }
    
    def update_sample(self, idx: int, caption_data: dict):
        """Update the caption data for a sample."""
        self.samples[idx]['caption_data'] = caption_data
    
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
):
    """Create a new tar file with audio and caption JSON.
    
    Args:
        samples: List of sample dicts with 'key', 'audio_bytes', 'caption_data', 'audio_ext'
        output_path: Path for the output tar file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tarfile.open(output_path, 'w') as tar:
        for sample in samples:
            key = sample['key']
            prefix = sample.get('path_prefix', '')
            
            # Add audio file
            audio_ext = sample.get('audio_ext', '.m4a')
            audio_name = f"{prefix}/{key}{audio_ext}" if prefix else f"{key}{audio_ext}"
            audio_data = sample['audio_bytes']
            audio_info = tarfile.TarInfo(name=audio_name)
            audio_info.size = len(audio_data)
            tar.addfile(audio_info, io.BytesIO(audio_data))
            
            # Add caption JSON file
            json_name = f"{prefix}/{key}.json" if prefix else f"{key}.json"
            caption_data = sample.get('caption_data', {})
            json_bytes = json.dumps(caption_data, indent=2, ensure_ascii=False).encode('utf-8')
            json_info = tarfile.TarInfo(name=json_name)
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))


def main():
    args = parse_args()
    
    # Validate shard arguments
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard_index and --num_shards must both be specified or both be omitted")
    if args.shard_index is not None and args.shard_index >= args.num_shards:
        raise ValueError(f"--shard_index ({args.shard_index}) must be less than --num_shards ({args.num_shards})")
    
    # Configuration
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    input_tar = input_dir / args.tar_name
    
    # Add shard suffix to output tar name if sharding
    if args.shard_index is not None:
        tar_base = args.tar_name.replace('.tar', '')
        output_tar_name = f"{tar_base}_shard{args.shard_index:02d}.tar"
        cache_dir = output_dir / f"audio_cache_{tar_base}_shard{args.shard_index:02d}"
    else:
        output_tar_name = args.tar_name
        cache_dir = output_dir / f"audio_cache_{args.tar_name.replace('.tar', '')}"
    
    output_tar = output_dir / output_tar_name
    
    model_id = args.model_id
    num_workers = args.num_workers
    max_new_tokens = args.max_new_tokens
    
    # DataLoader settings
    batch_size = 1  # Audio Flamingo processes one at a time
    prefetch_factor = 2  # Number of batches to prefetch per worker
    
    prompt = "Describe the audio sounds in 10 words or less."
    
    print(f"Input tar: {input_tar}")
    print(f"Output tar: {output_tar}")
    print(f"Model: {model_id}")
    print(f"Num workers: {num_workers}")
    if args.shard_index is not None:
        print(f"Shard: {args.shard_index}/{args.num_shards}")
    print()
    
    # Check if input exists
    if not input_tar.exists():
        raise FileNotFoundError(f"Input tar not found: {input_tar}")
    
    # Create dataset and dataloader
    print("Loading samples from tar file...")
    dataset = PrioVideoTarDataset(
        str(input_tar), 
        cache_dir=str(cache_dir), 
        max_samples=args.max_samples,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    
    if len(dataset) == 0:
        print("No samples found in tar file. Exiting.")
        return
    
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
                    
                    # Build caption data from original metadata
                    caption_data = {
                        'key': sample['key'],
                    }
                    
                    # Include relevant info from original metadata
                    if sample.get('metadata'):
                        for key in ['clip_id', 'scene_index', 'start_time', 'end_time', 
                                    'duration', 'original_video']:
                            if key in sample['metadata']:
                                caption_data[key] = sample['metadata'][key]
                    
                    if sample.get('info'):
                        for key in ['id', 'title', 'description', 'channel', 'uploader',
                                    'upload_date', 'duration', 'categories', 'tags']:
                            if key in sample['info']:
                                caption_data[f'original_{key}'] = sample['info'][key]
                    
                    try:
                        # Generate new caption
                        new_caption = generate_caption(
                            model, processor, audio_path, prompt, max_new_tokens
                        )
                        
                        # Add caption to data
                        caption_data['text'] = [new_caption]
                        caption_data['recaption_timestamp'] = datetime.now().isoformat()
                        caption_data['recaption_model'] = model_id
                        caption_data['recaption_prompt'] = prompt
                        
                    except Exception as e:
                        print(f"\nError processing {sample['key']}: {e}")
                        caption_data['text'] = []
                        caption_data['recaption_error'] = str(e)
                        caption_data['recaption_timestamp'] = datetime.now().isoformat()
                    
                    # Update the dataset's sample
                    dataset.update_sample(idx, caption_data)
    
    finally:
        # Clean up cached audio files
        print("\nCleaning up cache...")
        dataset.cleanup_cache()
    
    # Prepare samples for output
    output_samples = [
        {
            'key': s['key'],
            'audio_bytes': s['audio_bytes'],
            'audio_ext': s.get('audio_ext', '.m4a'),
            'caption_data': s.get('caption_data', {}),
            'path_prefix': s['path_prefix'],
        }
        for s in dataset.samples
    ]
    
    # Save output tar
    print(f"\nSaving output tar to: {output_tar}")
    create_output_tar(output_samples, str(output_tar))
    
    # Update sizes.json (append/update entry for this tar)
    sizes_path = output_dir / "sizes.json"
    if sizes_path.exists():
        with open(sizes_path, 'r') as f:
            sizes_data = json.load(f)
    else:
        sizes_data = {}
    sizes_data[output_tar_name] = len(output_samples)
    with open(sizes_path, 'w') as f:
        json.dump(sizes_data, f, indent=4)
    print(f"Updated sizes.json: {sizes_path}")
    
    print(f"\nCompleted! Processed {len(output_samples)} samples.")
    print(f"Output saved to: {output_tar}")


if __name__ == "__main__":
    main()

