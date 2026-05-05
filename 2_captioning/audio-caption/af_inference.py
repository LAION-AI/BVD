#!/usr/bin/env python
"""
Audio Flamingo inference script for batch processing audio files.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm.auto import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from audio_dataset import get_audio_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Audio Flamingo inference on a directory of audio files"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files (m4a, wav, etc.)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="audio_flamingo_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache converted wav files",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/audio-flamingo-3-hf",
        help="HuggingFace model ID for Audio Flamingo",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (Note: Audio Flamingo processes one at a time)",
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
        "--prompt",
        type=str,
        default="Generate a detailed caption for the input audio, describing all notable speech, sound, and musical events comprehensively. In the caption, transcribe all spoken content by all speakers in the audio precisely.",
        help="Prompt for audio captioning",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    return parser.parse_args()


def load_existing_results(output_file: str) -> dict:
    """Load existing results for resuming."""
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)
    return {}


def save_results(results: dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_chunk(chunk_results: dict, output_file: str, chunk_num: int):
    """Save a chunk of results to a separate JSON file."""
    base, ext = os.path.splitext(output_file)
    chunk_file = f"{base}_chunk_{chunk_num:04d}{ext}"
    with open(chunk_file, "w") as f:
        json.dump(chunk_results, f, indent=2, ensure_ascii=False)
    print(f"Saved chunk {chunk_num} to {chunk_file}")


def main():
    args = parse_args()
    
    print(f"Starting Audio Flamingo inference")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output file: {args.output_file}")
    
    # Load model and processor
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")
    
    # Set up cache directory
    if args.cache_dir is None:
        args.cache_dir = os.path.join(
            os.path.dirname(args.output_file),
            "wav_cache"
        )
    
    # Create dataloader
    dataloader = get_audio_dataloader(
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    
    # Load existing results if resuming
    if args.resume:
        results = load_existing_results(args.output_file)
        print(f"Resuming with {len(results)} existing results")
    else:
        results = {}
    
    # Process audio files
    total_processed = 0
    total_skipped = 0
    chunk_num = 0
    chunk_results = {}
    chunk_size = 10
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing audio files"):
            for i in range(len(batch["audio_paths"])):
                audio_path = batch["audio_paths"][i]
                original_path = batch["original_paths"][i]
                file_id = batch["file_ids"][i]
                
                # Skip if already processed
                if file_id in results:
                    total_skipped += 1
                    continue
                
                try:
                    # Prepare inputs
                    inputs = processor.apply_transcription_request(
                        audio=audio_path,
                        prompt=args.prompt
                    ).to(model.device)
                    
                    # Cast float tensors to bfloat16 to match model dtype
                    for key in inputs:
                        if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                            inputs[key] = inputs[key].to(torch.bfloat16)
                    
                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                    )
                    
                    # Decode
                    decoded = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True,
                        strip_prefix=True
                    )
                    
                    caption = decoded[0] if decoded else ""
                    
                    # Store result in both main results and current chunk
                    result_entry = {
                        "file_id": file_id,
                        "original_path": original_path,
                        "caption": caption,
                        "timestamp": datetime.now().isoformat(),
                    }
                    results[file_id] = result_entry
                    chunk_results[file_id] = result_entry
                    
                    total_processed += 1
                    
                    # Save chunk every 10 files
                    if len(chunk_results) >= chunk_size:
                        save_chunk(chunk_results, args.output_file, chunk_num)
                        chunk_num += 1
                        chunk_results = {}
                        
                except Exception as e:
                    print(f"Error processing {file_id}: {e}")
                    error_entry = {
                        "file_id": file_id,
                        "original_path": original_path,
                        "caption": None,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    results[file_id] = error_entry
                    chunk_results[file_id] = error_entry
    
    # Save any remaining results in the last chunk
    if chunk_results:
        save_chunk(chunk_results, args.output_file, chunk_num)
    
    # Also save the complete results file
    save_results(results, args.output_file)
    
    print(f"\nCompleted!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

