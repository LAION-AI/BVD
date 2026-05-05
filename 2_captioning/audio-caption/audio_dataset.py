#!/usr/bin/env python
"""
PyTorch Dataset and DataLoader for audio files with m4a to wav conversion.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Dataset for loading audio files, with automatic m4a to wav conversion."""
    
    def __init__(
        self,
        audio_dir: str,
        extensions: Tuple[str, ...] = (".m4a", ".wav", ".mp3", ".flac"),
        cache_dir: Optional[str] = None,
        convert_to_wav: bool = True,
    ):
        """
        Args:
            audio_dir: Directory containing audio files
            extensions: Tuple of audio file extensions to include
            cache_dir: Directory to cache converted wav files. If None, uses temp directory.
            convert_to_wav: Whether to convert non-wav files to wav format
        """
        self.audio_dir = Path(audio_dir)
        self.convert_to_wav = convert_to_wav
        
        # Set up cache directory for converted files
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="audio_cache_"))
        
        # Find all audio files
        self.audio_files: List[Path] = []
        for ext in extensions:
            self.audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
        
        self.audio_files = sorted(self.audio_files)
        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def _convert_to_wav(self, input_path: Path) -> Path:
        """Convert audio file to wav format using ffmpeg."""
        # Generate output path in cache directory
        relative_path = input_path.relative_to(self.audio_dir)
        output_path = self.cache_dir / relative_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already converted
        if output_path.exists():
            return output_path
        
        # Convert using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",  # Standard wav format
            "-ar", "16000",  # 16kHz sample rate (common for speech models)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            str(output_path)
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error converting {input_path}: {e.stderr}")
            raise
        
        return output_path
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - audio_path: Path to the (possibly converted) wav file
                - original_path: Original file path
                - file_id: Filename without extension (for identification)
        """
        original_path = self.audio_files[idx]
        
        if self.convert_to_wav and original_path.suffix.lower() != ".wav":
            audio_path = self._convert_to_wav(original_path)
        else:
            audio_path = original_path
        
        return {
            "audio_path": str(audio_path),
            "original_path": str(original_path),
            "file_id": original_path.stem,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader - just returns list of dicts."""
    return {
        "audio_paths": [item["audio_path"] for item in batch],
        "original_paths": [item["original_path"] for item in batch],
        "file_ids": [item["file_id"] for item in batch],
    }


def get_audio_dataloader(
    audio_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for audio files."""
    dataset = AudioDataset(
        audio_dir=audio_dir,
        cache_dir=cache_dir,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )

