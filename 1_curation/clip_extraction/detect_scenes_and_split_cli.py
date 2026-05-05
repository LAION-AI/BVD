#!/usr/bin/env python3
"""
Script to detect scenes in videos using scenedetect CLI and split them into smaller clips.
Also splits corresponding audio files and preserves all metadata.

This version uses the `scenedetect` command-line tool with `split-video` for efficient
scene detection and video splitting in a single operation.

Output format: WebDataset-compatible format with continuous numeric IDs.
All clips are stored in the root output folder with 6-digit IDs (000000, 000001, etc.).
Each clip has files like: 000000.mp4, 000000.m4a, 000000.description, etc.

The script is resumable: if interrupted, it will continue from where it left off.
Progress is saved to a checkpoint file after each video is processed.

Requirements:
    - scenedetect CLI: pip install scenedetect[opencv] (must be available in PATH)
    - ffmpeg: Must be installed and available in PATH (for audio splitting)
    - rich: pip install rich

Usage:
    python detect_scenes_and_split_cli.py <input_folder> <output_folder> [--threshold 30.0] [--pattern "*.mp4"]
    
    # To start fresh (ignore previous progress):
    python detect_scenes_and_split_cli.py <input_folder> <output_folder> --no-resume
"""

import os
import re
import json
import time
import sys
import subprocess
import shutil
import logging
import tempfile
import csv
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple, Union
from multiprocessing.pool import ThreadPool
import numpy as np
from queue import Queue, Empty
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
import argparse

# Maximum number of clips per folder (WebDataset limit)
MAX_CLIPS_PER_FOLDER = 8000

# Number of digits for clip IDs (WebDataset format)
CLIP_ID_DIGITS = 6

# Video duration limits (in seconds)
DURATION_FILTER = True
MIN_VIDEO_DURATION = 10.0  # Skip videos shorter than 10 seconds
MAX_VIDEO_DURATION = 30 * 60  # Skip videos longer than 30 minutes

# Summary/progress file name (used for both progress tracking and final summary)
SUMMARY_FILE = "processing_summary.json"

# Global flag for graceful shutdown
_shutdown_requested = False

# Setup rich console
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

# def _signal_handler(signum, frame):
#     """Handle interrupt signals for graceful shutdown."""
#     global _shutdown_requested
#     if _shutdown_requested:
#         # Second interrupt, force exit
#         logger.warning("Force quitting...")
#         sys.exit(1)
#     _shutdown_requested = True
#     logger.warning("Shutdown requested. Finishing current task... (Ctrl+C again to force quit)")


class SceneInfo(NamedTuple):
    """Information about a detected scene."""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


def parse_timecode(timecode: str) -> float:
    """
    Parse a timecode string (HH:MM:SS.mmm) to seconds.
    
    Args:
        timecode: Timecode string in format HH:MM:SS.mmm
    
    Returns:
        Time in seconds as float
    """
    parts = timecode.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(timecode)


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get the duration of a video file in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Duration in seconds, or None if duration cannot be determined
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        duration_str = result.stdout.decode().strip()
        return float(duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Failed to get duration for {video_path}: {e}")
        return None


def detect_and_split_scenes_cli(
    video_path: str,
    output_dir: str,
    threshold: float = 30.0,
    copy: bool = False
) -> Tuple[List[SceneInfo], List[Path]]:
    """
    Detect scenes and split video using scenedetect CLI.
    
    Uses the scenedetect command-line tool to detect scenes and split the video
    in one operation, which is more efficient than separate detection and splitting.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to output split video files
        threshold: Sensitivity threshold for scene detection (lower = more sensitive)
        copy: If True, use stream copy for fast splitting without re-encoding.
              If False, re-encode the video (slower but more compatible).
    
    Returns:
        Tuple of (list of SceneInfo objects, list of output video paths in order)
    
    Raises:
        ValueError: If threshold is not positive
        FileNotFoundError: If video file does not exist
        subprocess.CalledProcessError: If scenedetect CLI fails
    """
    # Input validation
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get video base name for output files
    video_name = Path(video_path).stem
    scenes_csv = output_path / f"{video_name}-Scenes.csv"
    
    # Run scenedetect CLI to detect scenes, list them to CSV, and split video
    cmd = [
        '/venv/bin/scenedetect',
        '-i', video_path,
        'detect-content', '-t', str(threshold),
        'list-scenes', '-f', str(scenes_csv), '-s',  # -s to skip scene list output to stdout
        'split-video', '-o', str(output_path)
    ]
    # Add --copy for fast splitting without re-encoding (if enabled)
    if copy:
        cmd.append('--copy')
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"scenedetect CLI failed: {e.stderr.decode()}")
        raise
    
    # Parse scenes from CSV file
    scenes = []
    if scenes_csv.exists():
        with open(scenes_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV has columns: Scene Number, Start Frame, Start Timecode, Start Time (seconds),
                # End Frame, End Timecode, End Time (seconds), Length (frames), Length (timecode), Length (seconds)
                try:
                    scenes.append(SceneInfo(
                        start_time=float(row.get('Start Time (seconds)', 0)),
                        end_time=float(row.get('End Time (seconds)', 0)),
                        start_frame=int(row.get('Start Frame', 0)),
                        end_frame=int(row.get('End Frame', 0))
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse scene row: {row}, error: {e}")
                    continue
    
    # Find the created video files (scenedetect creates files like video-Scene-001.mp4)
    video_files = sorted(output_path.glob(f"{video_name}-Scene-*.mp4"))
    
    # Verify we have matching counts
    if len(video_files) != len(scenes):
        logger.warning(f"Mismatch: {len(scenes)} scenes detected but {len(video_files)} video files created")
    
    return scenes, video_files


def is_static_video(
    path: str,
    sample_fps: float = 2.0,
    width: int = 160,
    height: int = 90,
    seconds_to_check: int = 10,
    mad_threshold: float = 2.0,   # ~0–255 scale after downscale; tune
    max_motion_frames: int = 1.5,   # allow a little noise if you want
) -> bool:
    """
    Returns True if the video appears static (no motion) in the sampled segment.
    Uses mean absolute difference (MAD) between consecutive frames on low-res RGB.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    # How many frames we expect to read
    n_frames = int(sample_fps * seconds_to_check)
    if n_frames < 2:
        n_frames = 2

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-an",
        "-vf", f"fps={sample_fps},scale={width}:{height}",
        "-frames:v", str(n_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = width * height * 3
    prev = None
    motion_frames = 0

    try:
        for _ in range(n_frames):
            raw = p.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            if prev is not None:
                mad = float(np.mean(np.abs(frame.astype(np.int16) - prev)))
                if mad > mad_threshold:
                    motion_frames += 1
                    if motion_frames > max_motion_frames:
                        return False  # not static (motion detected) -> early exit

            prev = frame
    finally:
        # Make sure the process is cleaned up
        p.stdout.close()
        p.terminate()
        p.wait(timeout=2)

    return True  # no meaningful motion found


def split_media(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    media_type: str = "media",
    copy: bool = True
) -> Tuple[bool, str]:
    """
    Split a media file (video or audio) using ffmpeg.
    
    Uses input seeking (-ss before -i) for fast seeking without decoding.
    
    Args:
        input_path: Input media path
        output_path: Output media path
        start_time: Start time in seconds
        end_time: End time in seconds
        media_type: Type of media for error messages ("video" or "audio")
    
    Returns:
        Tuple of (success, error message)
    """
    if start_time >= end_time:
        # logger.error(f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})")
        return False, f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})"
    
    duration = end_time - start_time
    if copy:
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Input seeking (before -i) for fast seek
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',  # Copy codec (fast, no re-encoding)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            output_path
        ]
    elif media_type == "audio":
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Input seeking (before -i) for fast seek
            '-i', input_path,
            '-t', str(duration),
            "-c:a", "aac", "-b:a", "192k",
            '-y',  # Overwrite output file
            output_path
        ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        # logger.error(f"Error splitting {media_type}: {e.stderr.decode()}")
        return False, f"Error splitting {media_type}: {e.stderr.decode()}"


def split_video(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float
) -> Tuple[bool, str]:
    """
    Split a video using ffmpeg.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        True if successful, False otherwise
    """
    return split_media(video_path, output_path, start_time, end_time, media_type="video")


def split_audio(
    audio_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    copy: bool = True
) -> Tuple[bool, str]:
    """
    Split an audio file using ffmpeg.
    
    Args:
        audio_path: Input audio path
        output_path: Output audio path
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        True if successful, False otherwise
    """
    return split_media(audio_path, output_path, start_time, end_time, media_type="audio", copy=copy)


def get_base_name(file_path: str) -> str:
    """
    Extract base name from a video file path.
    
    Handles filenames with resolution suffixes like '.360p', '.720p', '.1080p'.
    
    Examples:
        '0000.360p.mp4' -> '0000'
        'my.video.name.360p.mp4' -> 'my.video.name'
        'simple.mp4' -> 'simple'
    """
    stem = Path(file_path).stem
    # Remove resolution suffix like .360p, .480p, .720p, .1080p
    return re.sub(r'\.\d+p$', '', stem)


def find_related_files(base_name: str, input_folder: str) -> Dict[str, Union[Optional[str], List[str]]]:
    """
    Find all related files for a given base name.
    
    Returns:
        Dictionary mapping file type to file path (or None if not found).
        'vtt_files' contains a list of all VTT subtitle files found.
    """
    input_path = Path(input_folder)
    files: Dict[str, Union[Optional[str], List[str]]] = {
        'video': None,
        'audio': None,
        'description': None,
        'vtt_files': [],  # List of all VTT files
        'id_txt': None,
        'info_json': None,
        'comments_json': None,
    }
    
    # Common video extensions
    # for ext in ['.360p.mp4', '.480p.mp4', '.720p.mp4', '.1080p.mp4', '.mp4']:
    #     video_path = input_path / f"{base_name}{ext}"
    #     if video_path.exists():
    #         files['video'] = str(video_path)
    #         break
    # Only process 360p videos
    video_path = input_path / f"{base_name}.360p.mp4"
    if video_path.exists():
        files['video'] = str(video_path)
    
    # Common audio extensions
    for ext in ['.m4a', '.mp3', '.wav', '.aac']:
        audio_path = input_path / f"{base_name}{ext}"
        if audio_path.exists():
            files['audio'] = str(audio_path)
            break
    
    # Find all VTT files (all languages)
    vtt_files = sorted(input_path.glob(f"{base_name}.*.vtt"))
    files['vtt_files'] = [str(vtt) for vtt in vtt_files]
    
    # Metadata files
    metadata_patterns = {
        'description': f"{base_name}.description",
        'id_txt': f"{base_name}.id.txt",
        'info_json': f"{base_name}.info.json",
        'comments_json': f"{base_name}.comments.json",
    }
    
    for key, pattern in metadata_patterns.items():
        file_path = input_path / pattern
        if file_path.exists():
            files[key] = str(file_path)
    
    return files


def copy_metadata_file(src_path: str, dst_path: str) -> bool:
    """Copy a metadata file."""
    try:
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        logger.error(f"Error copying metadata file {src_path}: {e}")
        return False


def tar_folder(folder_path: Path) -> Tuple[bool, str]:
    """
    Create a tar archive of a folder using the tar command and rename the original folder for deletion.
    
    After successful tarring, the folder is renamed to 'foldername_DELETE' instead of being
    immediately deleted. This provides a safety buffer - only folders ending with '_DELETE'
    are cleaned up separately.
    
    Args:
        folder_path: Path to the folder to tar
    
    Returns:
        Tuple of (success, message)
    """
    tar_path = folder_path.with_suffix('.tar')
    parent_dir = folder_path.parent
    folder_name = folder_path.name
    delete_folder_path = folder_path.with_name(f"{folder_name}_DELETE")
    
    def cleanup_tar():
        """Remove incomplete tar file if it exists."""
        if tar_path.exists():
            try:
                tar_path.unlink()
            except Exception:
                pass
    
    try:
        # rm -r /path/to/prio_video_scenes_cli/**/*_DELETE
        # Create tar archive using tar command with sorted filenames for reproducibility
        cmd = [
            'tar',
            '--sort=name',
            '-cf', str(tar_path),
            '-C', str(folder_path),  # Change to subfolder
            "."  # Archive the current directory
        ]
        
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Verify the tar file was created and has content
        if tar_path.exists() and tar_path.stat().st_size > 0:
            # Rename the folder to mark it for deletion (safer than immediate delete)
            folder_path.rename(delete_folder_path)
            # Delete the original folder
            shutil.rmtree(delete_folder_path)
            return True, f"Successfully tarred {folder_name}"
        
        cleanup_tar()
        return False, f"Tar file creation failed for {folder_name}"
    
    except subprocess.CalledProcessError as e:
        cleanup_tar()
        return False, f"Error tarring {folder_name}: {e.stderr.decode(errors='replace')}"
    except Exception as e:
        cleanup_tar()
        return False, f"Error tarring {folder_name}: {e}"


def tar_subfolders(output_folder: Path, num_workers: int = 8) -> Tuple[int, int]:
    """
    Tar all numeric subfolders in the output folder using a ThreadPool.
    
    Args:
        output_folder: Base output folder containing numbered subfolders
        num_workers: Number of parallel workers for tarring
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Find all numeric subfolders (e.g., 0000, 0001, etc.)
    subfolders = sorted([
        d for d in output_folder.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    
    if not subfolders:
        logger.info("No subfolders to tar")
        return 0, 0
    
    logger.info(f"Tarring {len(subfolders)} subfolders with {num_workers} workers...")
    
    successful = 0
    failed = 0
    
    with ThreadPool(min(len(subfolders), num_workers)) as pool:
        results = pool.map(tar_folder, subfolders)
    
    for success, message in results:
        if success:
            successful += 1
            logger.info(message)
        else:
            failed += 1
            logger.error(message)
    
    return successful, failed


def cleanup_incomplete_tars(output_folder: Path) -> int:
    """
    Clean up incomplete tar files from interrupted processing.
    
    A tar is considered incomplete if:
    - The tar file exists AND the source folder still exists (interrupted mid-tar)
    - OR there's a .tar.tmp file (partial write)
    
    Args:
        output_folder: Base output folder containing numbered subfolders/tars
    
    Returns:
        Number of incomplete tars cleaned up
    """
    cleaned = 0
    
    # Find all tar files and check if their source folders still exist
    for tar_file in output_folder.glob("*.tar"):
        folder_name = tar_file.stem
        folder_path = output_folder / folder_name
        
        # If folder still exists, the tar is incomplete (was interrupted)
        if folder_path.exists() and folder_path.is_dir():
            try:
                tar_file.unlink()
                cleaned += 1
                logger.info(f"Removed incomplete tar: {tar_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove incomplete tar {tar_file.name}: {e}")
    
    # Also clean up any .tmp tar files
    for tmp_file in output_folder.glob("*.tar.tmp"):
        try:
            tmp_file.unlink()
            cleaned += 1
            logger.info(f"Removed temp tar file: {tmp_file.name}")
        except Exception as e:
            logger.warning(f"Failed to remove temp tar {tmp_file.name}: {e}")
    
    return cleaned


def cleanup_delete_folders(output_folder: Path) -> int:
    """
    Clean up folders marked for deletion (ending with '_DELETE').
    
    These folders were successfully tarred and renamed from their original name
    to 'foldername_DELETE'. They are safe to delete.
    
    Args:
        output_folder: Base output folder containing numbered subfolders/tars
    
    Returns:
        Number of folders deleted
    """
    deleted = 0
    
    for folder in output_folder.iterdir():
        if folder.is_dir() and folder.name.endswith('_DELETE'):
            try:
                shutil.rmtree(folder)
                deleted += 1
                logger.info(f"Deleted folder marked for removal: {folder.name}")
            except Exception as e:
                logger.warning(f"Failed to delete folder {folder.name}: {e}")
    
    return deleted


def tar_worker(output_folder: Path, stop_event: threading.Event, tar_queue: Queue):
    """
    Background worker that tars folders from the queue.
    
    Args:
        output_folder: Base output folder
        stop_event: Event to signal worker to stop
        tar_queue: Queue of folder names to tar
    """
    while not stop_event.is_set():
        try:
            # Wait for a folder to tar with timeout to check stop_event
            folder_name = tar_queue.get(timeout=5.0)
        except Empty:
            continue
        
        if folder_name is None:  # Poison pill to stop
            tar_queue.task_done()
            break
        
        folder_path = output_folder / folder_name
        if folder_path.exists() and folder_path.is_dir():
            success, message = tar_folder(folder_path)
            if success:
                logger.info(f"[Tar Worker] {message}")
            else:
                logger.error(f"[Tar Worker] {message}")
        
        tar_queue.task_done()


def start_tar_worker(output_folder: Path) -> Tuple[Queue, threading.Thread, threading.Event]:
    """
    Start the background tar worker thread.
    
    Args:
        output_folder: Base output folder
    
    Returns:
        Tuple of (queue, thread, stop_event)
    """
    tar_queue: Queue = Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=tar_worker,
        args=(output_folder, stop_event, tar_queue),
        daemon=True,
        name="TarWorker"
    )
    thread.start()
    logger.info("Started background tar worker thread")
    return tar_queue, thread, stop_event


def stop_tar_worker(tar_queue: Queue, thread: threading.Thread, stop_event: threading.Event):
    """
    Stop the background tar worker thread and wait for pending tars to complete.
    
    Args:
        tar_queue: The tar queue
        thread: The worker thread
        stop_event: The stop event
    """
    logger.info("Waiting for pending tar operations to complete...")
    # Wait for all queued items to be processed
    tar_queue.join()
    # Signal worker to stop
    stop_event.set()
    tar_queue.put(None)  # Poison pill
    thread.join(timeout=60*59)  # Wait up to 59 minutes
    if thread.is_alive():
        logger.warning("Tar worker thread did not stop cleanly")
    else:
        logger.info("Tar worker thread stopped")


def load_progress(output_folder: Path) -> Dict:
    """
    Load progress from processing_summary.json in the output folder.
    
    Returns:
        Dictionary with progress state, or empty dict if no summary exists.
        Keys:
        - 'processed_videos': set of base_names that have been fully processed
        - 'next_clip_id': the next clip ID to use (derived from total_clips)
        - 'results': list of all previous results
        - ... other summary fields
    """
    summary_file = output_folder / SUMMARY_FILE
    if not summary_file.exists():
        return {}
    
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        # Load processed_videos directly from the summary (stored as list, convert to set)
        data['processed_videos'] = set(data.get('processed_videos', []))
        # next_clip_id is total_clips (since clip IDs are 0-indexed)
        data['next_clip_id'] = data.get('total_clips', 0)
        return data
    except Exception as e:
        logger.warning(f"Failed to load progress from summary: {e}")
        return {}


def save_summary(
    output_folder: Path,
    input_folder: Path,
    source_folder_name: str,
    threshold: float,
    total_videos: int,
    total_video_files: int,
    all_results: List,
    processed_videos: set,
    interrupted: bool = False,
    tarring: Optional[Dict] = None
) -> bool:
    """
    Save processing summary to the output folder.
    
    This file serves both as a progress checkpoint and as the final summary.
    
    Args:
        output_folder: Output folder path
        input_folder: Input folder path
        source_folder_name: Name of the source folder
        threshold: Scene detection threshold used
        total_videos: Total number of videos to process
        total_video_files: Total number of video files found
        all_results: List of all processing results
        processed_videos: Set of base_names that have been fully processed
        interrupted: Whether processing was interrupted
        tarring: Tarring status dict (optional)
    
    Returns:
        True if successful, False otherwise
    """
    summary_file = output_folder / SUMMARY_FILE
    try:
        # Calculate statistics from all results
        total_clips = sum(len(r.get('clips', [])) for r in all_results)
        total_errors = sum(1 for r in all_results if 'error' in r)
        num_subfolders = (total_clips + MAX_CLIPS_PER_FOLDER - 1) // MAX_CLIPS_PER_FOLDER if total_clips > 0 else 0
        
        data = {
            'input_folder': str(input_folder),
            'output_base_folder': str(output_folder),
            'source_folder_name': source_folder_name,
            'max_clips_per_folder': MAX_CLIPS_PER_FOLDER,
            'num_subfolders': num_subfolders,
            'threshold': threshold,
            'total_videos': total_videos,
            'total_video_files': total_video_files,
            'videos_processed': len(all_results),
            'total_clips': total_clips,
            'total_errors': total_errors,
            'processed_videos': sorted(processed_videos),  # Store as sorted list for JSON serialization
            'interrupted': interrupted,
            'tarring': tarring or {'enabled': False, 'successful': 0, 'failed': 0},
            'results': all_results
        }
        
        # Write atomically by writing to temp file first
        temp_file = summary_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(summary_file)
        return True
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        return False


def cleanup_partial_clips(output_folder: Path, next_clip_id: int) -> int:
    """
    Clean up any partial clips from an interrupted video processing.
    
    Removes any clips with IDs >= next_clip_id, as these are from interrupted processing.
    
    Args:
        output_folder: Base output folder containing numbered subfolders
        next_clip_id: The next expected clip ID (clips with this ID or higher are partial)
    
    Returns:
        Number of files cleaned up
    """
    cleaned = 0
    
    # Check all subfolders that could contain partial clips
    start_folder = next_clip_id // MAX_CLIPS_PER_FOLDER
    
    # Check folders starting from where partial clips could be
    for folder_idx in range(start_folder, start_folder + 10):  # Check up to 10 folders ahead
        folder_path = output_folder / f"{folder_idx:04d}"
        if not folder_path.exists():
            break
        
        # Find all files that belong to clips >= next_clip_id
        for item in folder_path.iterdir():
            if not item.is_file():
                continue
            
            # Extract clip ID from filename (e.g., "000123.mp4" -> 123)
            name_parts = item.name.split('.')
            if not name_parts[0].isdigit():
                continue
            
            try:
                clip_id = int(name_parts[0])
                if clip_id >= next_clip_id:
                    item.unlink()
                    cleaned += 1
            except ValueError:
                continue
    
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} files from interrupted processing")
    
    return cleaned


def process_video(
    base_name: str,
    input_folder: str,
    output_base_folder: str,
    clip_id: int,
    threshold: float = 30.0,
    copy: bool = False,
    tar_queue: Optional[Queue] = None
) -> Tuple[Dict, int]:
    """
    Process a single video: detect scenes, split video/audio, and copy metadata.
    
    Uses scenedetect CLI for scene detection and video splitting, ffmpeg for audio splitting.
    
    Args:
        base_name: Base name of the video (e.g., '0000')
        input_folder: Input folder containing videos and metadata
        output_base_folder: Base output folder (will create subfolders as needed)
        clip_id: Starting clip ID for this video
        threshold: Scene detection threshold
        copy: If True, use stream copy for fast splitting without re-encoding
        tar_queue: Optional queue to send completed folder names for background tarring
    
    Returns:
        Tuple of (results dictionary, next available clip_id)
    """
    global _shutdown_requested
    
    # Find related files
    files = find_related_files(base_name, input_folder)
    
    if not files['video']:
        logger.warning(f"[{base_name}] No video file found, skipping")
        return {'error': f'No video file found for {base_name}', 'clips': []}, clip_id
    
    video_path = files['video']
    
    # Check video duration before processing
    if DURATION_FILTER:
        duration = get_video_duration(video_path)
        if duration is not None:
            if duration < MIN_VIDEO_DURATION:
                logger.warning(f"[{base_name}] Video too short ({duration:.2f}s < {MIN_VIDEO_DURATION}s), skipping")
                return {'error': f'Video too short ({duration:.2f}s) for {base_name}', 'clips': []}, clip_id
            if duration > MAX_VIDEO_DURATION:
                logger.warning(f"[{base_name}] Video too long ({duration:.2f}s > {MAX_VIDEO_DURATION/60:.0f}min), skipping")
                return {'error': f'Video too long ({duration:.2f}s) for {base_name}', 'clips': []}, clip_id
    
    # Log found related files
    has_audio = files['audio'] is not None
    num_vtt = len(files.get('vtt_files', []))
    num_metadata = sum(1 for k in ['description', 'id_txt', 'info_json', 'comments_json'] if files.get(k))
    logger.info(f"[{base_name}] Found: video={Path(video_path).name}, audio={'yes' if has_audio else 'no'}, VTT files={num_vtt}, metadata files={num_metadata}")
    
    # Create a temporary directory for scenedetect output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Detect scenes and split video using scenedetect CLI
        logger.info(f"[{base_name}] Starting scene detection and video splitting with scenedetect CLI...")
        try:
            scenes, split_video_files = detect_and_split_scenes_cli(video_path, temp_dir, threshold=threshold, copy=copy)
        except Exception as e:
            logger.error(f"[{base_name}] Scene detection/splitting failed: {e}")
            return {'error': f'Scene detection/splitting failed for {base_name}: {e}', 'clips': []}, clip_id
        
        logger.info(f"[{base_name}] Detected {len(scenes)} scenes, created {len(split_video_files)} video clips")
        
        if not scenes:
            return {'error': f'No scenes detected for {base_name}', 'clips': []}, clip_id
        
        # Process each scene
        clips = []
        current_clip_id = clip_id
        output_base_path = Path(output_base_folder)
        
        for scene_idx, scene in enumerate(scenes):
            
            # Calculate which subfolder this clip should be in
            folder_index = current_clip_id // MAX_CLIPS_PER_FOLDER
            output_folder = output_base_path / f"{folder_index:04d}"
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created new subfolder: {output_folder.name}/ (starting at clip {current_clip_id:0{CLIP_ID_DIGITS}d})")
                
                # Queue previous folder for tarring if it exists and is complete
                if tar_queue is not None and folder_index > 0:
                    prev_folder_name = f"{folder_index - 1:04d}"
                    prev_folder_path = output_base_path / prev_folder_name
                    prev_tar_path = output_base_path / f"{prev_folder_name}.tar"
                    # Only queue if folder exists and hasn't been tarred yet
                    if prev_folder_path.exists() and not prev_tar_path.exists():
                        logger.info(f"[{base_name}] Queuing folder {prev_folder_name} for background tarring")
                        tar_queue.put(prev_folder_name)
            
            # Format clip ID as 6-digit number (WebDataset format)
            clip_id_str = f"{current_clip_id:0{CLIP_ID_DIGITS}d}"
            
            scene_info = {
                'clip_id': current_clip_id,
                'clip_id_str': clip_id_str,
                'folder_index': folder_index,
                'original_base_name': base_name,
                'original_video': video_path,
                'scene_index': scene_idx,
                'start_time': scene.start_time,
                'end_time': scene.end_time,
                'start_frame': scene.start_frame,
                'end_frame': scene.end_frame,
                'duration': scene.end_time - scene.start_time,
                'frame_count': scene.end_frame - scene.start_frame,
                'files': {}
            }
            
            # Move the split video file to the output folder with the correct name
            video_output = output_folder / f"{clip_id_str}.mp4"
            if scene_idx < len(split_video_files):
                src_video = split_video_files[scene_idx]
                
                # Check if the video is static (no motion) - skip static videos
                try:
                    video_is_static = is_static_video(str(src_video))
                    scene_info['is_static'] = video_is_static
                    if video_is_static:
                        logger.info(f"[{base_name}] Skipping static scene {scene_idx + 1}/{len(scenes)} ({scene.start_time:.2f}s - {scene.end_time:.2f}s)")
                        continue
                except Exception as e:
                    logger.warning(f"[{base_name}] Failed to check if scene {scene_idx + 1} is static: {e}, proceeding with move")
                
                logger.info(f"[{base_name}] Moving scene {scene_idx + 1}/{len(scenes)} -> {clip_id_str}.mp4 ({scene.start_time:.2f}s - {scene.end_time:.2f}s, duration: {scene.end_time - scene.start_time:.2f}s)")
                try:
                    shutil.move(str(src_video), str(video_output))
                    scene_info['files']['video'] = str(video_output)
                except Exception as e:
                    logger.warning(f"[{base_name}] Failed to move video for scene {scene_idx + 1}: {e}")
            else:
                logger.warning(f"[{base_name}] No video file for scene {scene_idx + 1}")
            
            # Split audio using ffmpeg (scenedetect doesn't handle separate audio files)
            if files['audio']:
                audio_ext = Path(files['audio']).suffix
                audio_output = output_folder / f"{clip_id_str}{audio_ext}"
                logger.info(f"[{base_name}] Splitting audio -> {clip_id_str}{audio_ext}")
                audio_success, audio_error = split_audio(files['audio'], str(audio_output), scene.start_time, scene.end_time)
                if audio_success:
                    scene_info['files']['audio'] = str(audio_output)
                else:
                    logger.warning(f"[{base_name}] Failed to split audio for scene {scene_idx + 1}: {audio_error}")
            
            # Copy metadata files using ThreadPool for parallel copying
            metadata_mapping = {
                'description': 'description',
                'id_txt': 'id.txt',
                'info_json': 'info.json',
                'comments_json': 'comments.json',
            }
            
            # Collect all copy tasks: (src_path, dst_path, key, is_vtt)
            copy_tasks = []
            for key, suffix in metadata_mapping.items():
                if files[key]:
                    metadata_output = output_folder / f"{clip_id_str}.{suffix}"
                    copy_tasks.append((files[key], str(metadata_output), key, False))
            
            # Add VTT files to copy tasks
            vtt_files = files.get('vtt_files', [])
            for vtt_path in vtt_files:
                # Extract suffix like "en.vtt" or "de.vtt" from the original filename
                vtt_suffix = Path(vtt_path).name.replace(f"{base_name}.", "")
                vtt_output = output_folder / f"{clip_id_str}.{vtt_suffix}"
                copy_tasks.append((vtt_path, str(vtt_output), 'vtt', True))
            
            # Execute copy tasks in parallel using ThreadPool
            if copy_tasks:
                num_metadata = sum(1 for t in copy_tasks if not t[3])
                num_vtt = sum(1 for t in copy_tasks if t[3])
                logger.debug(f"[{base_name}] Copying {num_metadata} metadata files and {num_vtt} VTT files for {clip_id_str}")
                
                def do_copy(task):
                    src, dst, key, is_vtt = task
                    success = copy_metadata_file(src, dst)
                    return (success, dst, key, is_vtt)
                
                with ThreadPool(min(len(copy_tasks), 8)) as pool:
                    results = pool.map(do_copy, copy_tasks)
                
                # Process results
                scene_info['files']['vtt_files'] = []
                copied_count = 0
                for success, dst_path, key, is_vtt in results:
                    if success:
                        copied_count += 1
                        if is_vtt:
                            scene_info['files']['vtt_files'].append(dst_path)
                        else:
                            scene_info['files'][key] = dst_path
                    else:
                        logger.warning(f"[{base_name}] Failed to copy {key} file for {clip_id_str}")
            
            # Save scene metadata JSON
            metadata_json_output = output_folder / f"{clip_id_str}.metadata.json"
            with open(metadata_json_output, 'w') as f:
                json.dump({
                    'clip_id': current_clip_id,
                    'folder_index': folder_index,
                    'original_base_name': base_name,
                    'original_video': video_path,
                    'scene_index': scene_idx,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'start_frame': scene.start_frame,
                    'end_frame': scene.end_frame,
                    'duration': scene.end_time - scene.start_time,
                    'frame_count': scene.end_frame - scene.start_frame,
                }, f, indent=2)
            
            scene_info['files']['metadata_json'] = str(metadata_json_output)
            clips.append(scene_info)
            current_clip_id += 1
    
    # Log completion summary for this video
    total_duration = sum(c['duration'] for c in clips)
    logger.info(f"[{base_name}] Completed: {len(clips)} clips created (total duration: {total_duration:.2f}s)")
    
    results = {
        'base_name': base_name,
        'original_video': video_path,
        'num_scenes': len(scenes),
        'clips': clips
    }
    
    return results, current_clip_id


def main():
    global _shutdown_requested
    start_time = time.time()
    
    # Setup signal handlers for graceful shutdown
    # signal.signal(signal.SIGINT, _signal_handler)
    # signal.signal(signal.SIGTERM, _signal_handler)
    
    parser = argparse.ArgumentParser(
        description='Detect scenes in videos and split them into smaller clips with metadata'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing videos and metadata files'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder for split clips and metadata'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=30.0,
        help='Scene detection threshold (lower = more sensitive, default: 30.0)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='Pattern to match video files (default: *.mp4)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )
    # parser.add_argument(
    #     '--no-tar',
    #     action='store_true',
    #     help='Skip tarring subfolders at the end (default: tar subfolders)'
    # )
    # parser.add_argument(
    #     '--tar-workers',
    #     type=int,
    #     default=8,
    #     help='Number of parallel workers for tarring subfolders (default: 8)'
    # )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignoring any previous progress checkpoint'
    )
    parser.add_argument(
        '--cutoff-time',
        type=float,
        default=None,
        help='Cutoff time in seconds. If the processing is longer than this time, it will be stopped. Default: None (no cutoff)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Use stream copy for fast splitting without re-encoding'
    )
    parser.add_argument(
        '--tar',
        action='store_true',
        help='Enable background tarring of subfolders once they are full (8000 clips). '
             'Incomplete tars from interrupted runs will be cleaned up on restart.'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if args.threshold <= 0:
        logger.error(f"Threshold must be positive, got {args.threshold}")
        return
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        logger.error(f"Input folder '{input_folder}' does not exist")
        return
    
    # Get source folder name and create subfolder in output
    source_folder_name = input_folder.name
    output_subfolder = output_folder / source_folder_name
    
    # Create output subfolder
    output_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Initialize tar worker variables
    tar_queue: Optional[Queue] = None
    tar_thread: Optional[threading.Thread] = None
    tar_stop_event: Optional[threading.Event] = None
    
    # Clean up incomplete tars, delete folders marked for removal, and start tar worker if --tar is enabled
    if args.tar:
        cleaned_tars = cleanup_incomplete_tars(output_subfolder)
        if cleaned_tars > 0:
            logger.info(f"Cleaned up {cleaned_tars} incomplete tar files from previous run")
        deleted_folders = cleanup_delete_folders(output_subfolder)
        if deleted_folders > 0:
            logger.info(f"Deleted {deleted_folders} folders marked for removal from previous run")
        tar_queue, tar_thread, tar_stop_event = start_tar_worker(output_subfolder)
    
    # Find all video files and group by base name
    video_files = list(input_folder.glob(args.pattern))
    
    if not video_files:
        logger.error(f"No video files found matching pattern '{args.pattern}' in '{input_folder}'")
        return
    
    # Group video files by base name to avoid processing duplicates
    video_files_by_base = {}
    for video_file in video_files:
        base_name = get_base_name(str(video_file))
        if base_name not in video_files_by_base:
            video_files_by_base[base_name] = []
        video_files_by_base[base_name].append(str(video_file))
    
    logger.info(f"Found {len(video_files)} video files ({len(video_files_by_base)} unique base names)")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output base folder: {output_subfolder}")
    logger.info(f"Scene detection threshold: {args.threshold}")
    logger.info(f"Stream copy mode: [bold]{'ENABLED' if args.copy else 'DISABLED (re-encoding)'}[/bold]")
    logger.info(f"Background tarring: [bold]{'ENABLED' if args.tar else 'DISABLED'}[/bold]")
    logger.info(f"Max clips per folder: {MAX_CLIPS_PER_FOLDER}")
    logger.debug(f"COMPLETED FILE: {output_subfolder}.COMPLETED")
    
    # Load progress from previous run for resumability
    all_results: List[Dict] = []
    clip_id = 0  # Start with 0 for WebDataset format
    processed_videos: set = set()
    
    base_names = sorted(video_files_by_base.keys())
    total_videos = len(base_names)
    
    if not args.no_resume:
        progress = load_progress(output_subfolder)
        if progress:
            processed_videos = progress.get('processed_videos', set())
            clip_id = progress.get('next_clip_id', 0)
            all_results = progress.get('results', [])
            
            # Clean up any partial clips from interrupted processing
            cleanup_partial_clips(output_subfolder, clip_id)
        else:
            logger.info("No previous progress found, starting fresh")
    else:
        logger.info("Resume disabled (--no-resume), starting fresh")
    
    # Count videos to process (excluding already processed ones)
    videos_to_process = [bn for bn in base_names if bn not in processed_videos]
    already_done = len(processed_videos)
    if already_done > 0:
        logger.info(f"Resuming: {already_done} videos already processed, {len(videos_to_process)} remaining")
        logger.info(f"Continuing from clip ID {clip_id}")
    
    # Queue any untarred folders from previous runs for tarring (except the current one being written to)
    # This handles the case where folders were left untarred when resuming
    if tar_queue is not None and clip_id > 0:
        current_folder_index = clip_id // MAX_CLIPS_PER_FOLDER
        for folder_idx in range(current_folder_index):
            folder_name = f"{folder_idx:04d}"
            folder_path = output_subfolder / folder_name
            tar_path = output_subfolder / f"{folder_name}.tar"
            if folder_path.exists() and folder_path.is_dir() and not tar_path.exists():
                logger.info(f"Queuing untarred folder {folder_name} from previous run for tarring")
                tar_queue.put(folder_name)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing videos...", total=len(videos_to_process))
        
        for idx, base_name in enumerate(videos_to_process):
            # Check for shutdown request
            if _shutdown_requested:
                logger.warning("Shutdown requested, stopping processing...")
                break
            
            current_num = already_done + idx + 1
            logger.info(f"Processing video {current_num}/{total_videos}: {base_name}")
            
            result, clip_id = process_video(
                base_name,
                str(input_folder),
                str(output_subfolder),
                clip_id,
                threshold=args.threshold,
                copy=args.copy,
                tar_queue=tar_queue
            )
            all_results.append(result)
            processed_videos.add(base_name)
            
            # Save progress after each video completes
            save_summary(
                output_subfolder, input_folder, source_folder_name, args.threshold,
                total_videos, len(video_files), all_results, processed_videos
            )
            progress.update(task, advance=1)
            
            # Log running totals
            clips_so_far = sum(len(r.get('clips', [])) for r in all_results)
            errors_so_far = sum(1 for r in all_results if 'error' in r)
            logger.info(f"Progress: {current_num}/{total_videos} videos processed, {clips_so_far} total clips, {errors_so_far} errors")
            elapsed_time = time.time() - start_time
            if args.cutoff_time is not None and elapsed_time > args.cutoff_time:
                logger.warning(f"Cutoff time reached ({elapsed_time:.2f}s/{args.cutoff_time}s), stopping processing...")
                break
    
    # Calculate cumulative statistics from all results
    total_clips = sum(len(r.get('clips', [])) for r in all_results)
    total_errors = sum(1 for r in all_results if 'error' in r)
    
    # Queue the last folder for tarring (it wasn't queued since no new folder was created after it)
    if tar_queue is not None and clip_id > 0:
        last_folder_index = (clip_id - 1) // MAX_CLIPS_PER_FOLDER
        last_folder_name = f"{last_folder_index:04d}"
        last_folder_path = output_subfolder / last_folder_name
        last_tar_path = output_subfolder / f"{last_folder_name}.tar"
        # Only queue if folder exists and hasn't been tarred yet
        if last_folder_path.exists() and not last_tar_path.exists():
            logger.info(f"Queuing final folder {last_folder_name} for background tarring")
            tar_queue.put(last_folder_name)
    
    # Stop tar worker and wait for pending tars to complete
    if tar_queue is not None and tar_thread is not None and tar_stop_event is not None:
        stop_tar_worker(tar_queue, tar_thread, tar_stop_event)
    
    summary_path = output_subfolder / SUMMARY_FILE
    logger.info("Processing complete!")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Videos processed: {len(all_results)}/{total_videos}")
    logger.info(f"Total clips created: {total_clips}")
    # Touch file subfolder.completed
    if len(all_results) == total_videos:
        os.system(f"touch {output_subfolder}.COMPLETED")
    else:
        logger.warning(f"Processing interrupted before completion. Run the same command again to resume from where it left off.")
    if total_errors > 0:
        logger.warning(f"Errors encountered: {total_errors}")
    else:
        logger.info("No errors encountered")
    
    # if _shutdown_requested:
    #     logger.warning("Processing was interrupted before completion")
    #     logger.info("Run the same command again to resume from where it left off")


if __name__ == '__main__':
    main()
