import os
import re
import json
import shutil
import argparse
import subprocess
from glob import glob
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.console import Console
from typing import Optional, List, Dict

console = Console()


def _get_captions(video_file: str, captions: Optional[List[str]]) -> Dict[str, str]:
    if captions is None:
        return {}
    res = {}
    for c in captions:
        c_file = video_file.replace(".mp4", f".{c}.vtt")
        if os.path.exists(c_file):
            with open(c_file, "r") as f:
                res[c] = f.read()
    return res

def get_frame_info(stderr: str):
    """
    Parses the stderr output of ffmpeg to extract frame information.

    Parameters:
    stderr (str): The stderr output from ffmpeg.

    Returns:
    List[Dict[str, str]]: A list of dictionaries containing frame information.
    """
    frame_info, black_detect = [], []
    for line in stderr.split('\n'):
        if "showinfo" in line and "pts_time" in line:
            # Parse the line to extract frame number and timestamp
            pattern = re.compile(r'n:\s*(\d+)\s.*pts_time:\s*([\d.]+)')
            match = pattern.search(line)
            if match:
                frame_id = f'{int(match.group(1))+1:04d}'  # Contains the frame number
                pts_time = match.group(2)   # Contains the pts_time
                frame_info.append((frame_id, float(pts_time)))
        if "black_start" in line:
            # Parse the line to extract black start and end times
            pattern = re.compile(r'black_start:\s*([\d.]+)\s.*black_end:\s*([\d.]+)')
            match = pattern.search(line)
            if match:
                black_start = match.group(1)
                black_end = match.group(2)
                black_detect.append((float(black_start), float(black_end)))
    return frame_info, black_detect

def run_ffmpeg_filter(video_file: str, output_dir: str, black_filter='d=0.1:pix_th=0.1', scene_threshold=0.2, only_keyframes=True):
    select = f"eq(pict_type\\,I)*gt(scene,{scene_threshold})" if only_keyframes else f"gt(scene,{scene_threshold})"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_file,
        "-vf",
        f"blackdetect={black_filter}, select='{select}', showinfo",
        "-vsync",
        "vfr",
        os.path.join(output_dir, '%04d.png'),
    ]
    # console.log("RUN: " + " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def extract_frame_by_id(video_file: str, output_filename: str, frame_id: int):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_file,
        "-vf",
        f"select=eq(n\\,{frame_id}), showinfo",
        "-vsync",
        "vfr",
        output_filename
        # os.path.join(output_dir, f"{frame_id:04d}.png"),
    ]
    # console.log("RUN: " + " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

# def extract_frame_timestamps(video_file: str):
#     # Use ffprobe to get frame timestamps
#     command = [
#         "ffprobe",
#         "-show_frames",
#         "-select_streams", "v:0",
#         "-print_format", "json",
#         video_file
#     ]
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     ffprobe_output = json.loads(result.stdout)
#     return [(i, float(frame['pts_time'])) for i, frame in enumerate(ffprobe_output['frames'])]

def get_video_data(video_file: str, captions=None, fallback=True, blackdetect='d=0.1:pix_th=0.1', scene_threshold=0.2, only_keyframes=True):
    """
    Extracts video data and metadata from a given video file.

    Parameters:
    video_file (str): Path to the video file.
    transforms (callable, optional): A function/transform to apply to the video data.
    num_threads (int, optional): Number of threads to use for video reading (default is 4).
    fault_tol (int, optional): Fault tolerance level for video reading (default is -1).
    backend (str, optional): Backend to use for video reading, either "decord" or "torchvision" (default is "decord").

    Returns:
    tuple: A tuple containing:
        - video_data (Tensor): The video frames as a tensor of shape (num_frames, channels, height, width).
        - metadata (dict): Metadata associated with the video.
        - video_file (str): The path to the video file.
    """
    temp_dir = TemporaryDirectory()
    try:
        in_fallback = False
        frame0_pts_time = []
        res = run_ffmpeg_filter(video_file, temp_dir.name, scene_threshold=scene_threshold, black_filter=blackdetect, only_keyframes=only_keyframes)
        if len(os.listdir(temp_dir.name)) == 0 and fallback:
            # console.log("No frames extracted. Fallback to just black detect.")
            res = run_ffmpeg_filter(video_file, temp_dir.name, scene_threshold=0., black_filter=blackdetect, only_keyframes=only_keyframes)
            in_fallback = True
        else:
            # Get first frame of first scene
            res2 = extract_frame_by_id(video_file, os.path.join(temp_dir.name, "0000.png"), 0)
            frame0_pts_time = [("0000", 0.0)]#get_frame_info(res2.stderr)
        frames_pts_time, black_detect = get_frame_info(res.stderr)
        frames_pts_time = frame0_pts_time + frames_pts_time
        with open(video_file.replace(".mp4", ".info.json"), "r") as f:
            metadata = json.load(f)
        metadata["captions"] = _get_captions(
            video_file, ["de", "en", "fr", "it", "es", "ru", "zh-Hans", "ur", "pl"]
        )
        metadata["frames_pts_time"] = {frame_id: timestamp for frame_id, timestamp in frames_pts_time}
        metadata["black_detect"] = black_detect

    except Exception as e:
        temp_dir.cleanup()
        console.log(f"Error processing video {video_file}: {e}")
        return None

    return temp_dir, metadata, video_file, in_fallback


def filter_black_frames(
    video_data: List[str],
    frames_pts_time: dict[str, float],
    black_detect: List[tuple],
) -> List[str]:
    """
    Filters out frames that fall within black detection intervals.

    Parameters:
    video_data (List[str]): List of video frame filenames.
    frames_pts_time (List[tuple]): List of tuples containing frame IDs and their corresponding timestamps.
    black_detect (List[tuple]): List of tuples containing black start and end times.

    Returns:
    List[str]: List of frame filenames that are not within black detection intervals.
    """
    # frame_timestamps = {frame_id: timestamp for frame_id, timestamp in frames_pts_time}
    filtered_frames = []

    for frame in video_data:
        frame_id = frame.split(".")[0]
        timestamp = frames_pts_time.get(frame_id)
        if timestamp is None:
            continue

        # Check if the frame timestamp falls within any black detection interval
        if not any(black_start <= timestamp < black_end for black_start, black_end in black_detect):
            filtered_frames.append(frame)

    return filtered_frames

def extract_images_and_store(
    video_file,
    output_dir,
    file_name_start=0,
    blackdetect='d=0.1:pix_th=0.1',
    scene_threshold=0.2,
):
    res = get_video_data(video_file, blackdetect=blackdetect, scene_threshold=scene_threshold, only_keyframes=True)
    if res is None:
        return
    try:
        video_dir, metadata, video_file, in_fallback = res
        video_data: list[str] = sorted(os.listdir(video_dir.name))
        # remove black frames
        video_data = filter_black_frames(
            video_data,
            metadata.get("frames_pts_time", {}),
            metadata.get("black_detect", []),
        )
        if len(video_data) == 0:
            console.log(f"[red]No frames extracted from {video_file}.")
            console.log(f"{metadata.get('frames_pts_time', {}),}.")
            console.log(f"{metadata.get('black_detect', {}),}.")
            console.log(f"{sorted(os.listdir(video_dir.name))}.")
            return

            # import pdb; pdb.set_trace()
        if in_fallback:
            video_data = np.random.choice(video_data, 1).tolist()
        # if len(video_data) == 0:
        #     raise Exception("No video data found.")
        target_metadata = {
            "video_file": video_file,
            "num_frames": len(video_data),
            # "frame_ids": [int(frame.split(".")[0]) for frame in video_data],
            "frames_pts_time": [(i, metadata["frames_pts_time"][frame.split(".")[0]]) for i, frame in enumerate(video_data)],  # metadata.get("frames_pts_time", {}),
            # "black_detect": metadata.get("black_detect", []),
            "captions": metadata.get("captions", {}),
            # "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "title": metadata.get("fulltitle", ""),
            "height": metadata.get("height", -1),
            "width": metadata.get("width", -1),
            "duration": metadata.get("duration", -1),
            "url": metadata.get("url", ""),
            "webpage_url": metadata.get("webpage_url", ""),
        }
        # consistency check in case we just select one frame randomly
        # target_metadata["frame_info"] = [fi for fi in target_metadata["frame_info"] if int(fi[0]) in target_metadata["frame_ids"]]
        target_names = []
        for i, frame in enumerate(video_data):
            target_name = f"{file_name_start+i:06d}"
            target_names.append(target_name)
            try:
                shutil.move(
                    os.path.join(video_dir.name, frame),
                    f"{output_dir}/{target_name}.png",
                )
            except OSError as e:
                if e.errno == 18:  # cross device link
                    shutil.copy2(
                        os.path.join(video_dir.name, frame),
                        f"{output_dir}/{target_name}.png",
                    )
                else:
                    raise e
            frame_target_metadata = target_metadata.copy()
            frame_target_metadata["frame_id"] = i  # int(frame.split(".")[0]) - instead of actual file name just enumerate through selected frames
            frame_target_metadata["frame_pts_time"] = metadata["frames_pts_time"][frame.split(".")[0]]
            with open(f"{output_dir}/{target_name}.info.json", "w") as f:
                json.dump(frame_target_metadata, f)
        return target_names, target_metadata
    except Exception as e:
        console.log(e)
    finally:
        video_dir.cleanup()


def process_shard(source_dir, parquet_dir, source_shard, target_dir, blackdetect='d=0.1:pix_th=0.1', scene_threshold=0.2):
    df_rows = []
    target_shard_dir = os.path.join(target_dir, parquet_dir, source_shard)
    video_files = sorted(glob(os.path.join(source_dir, parquet_dir, source_shard, "*.mp4")))
    if os.path.exists(target_shard_dir):
        if os.path.exists(f"{target_shard_dir}.parquet"):
            console.log(f"[green]Shard {source_shard} already processed.")
            image_files = glob(os.path.join(target_shard_dir, "*.png"))
            return source_shard, len(image_files), len(video_files)
        else:
            console.log(f"[red]Removing incomplete shard {target_shard_dir}.")
            shutil.rmtree(target_shard_dir)
    os.makedirs(target_shard_dir, exist_ok=True)
    total_files, total_videos = 0, 0
    for video_file in video_files:
        res = extract_images_and_store(
            # os.path.join(source_dir, parquet_dir, source_shard, video_file),
            video_file,
            target_shard_dir,
            file_name_start=total_files,
            blackdetect=blackdetect,
            scene_threshold=scene_threshold,
        )
        if res is None:
            continue
        target_names, target_metadata = res
        for name, (frame_id, frame_pts_time) in zip(target_names, target_metadata["frames_pts_time"]):
            df_rows.append({
                "key": name,
                "video_file": target_metadata["video_file"],
                "frame_id": frame_id,
                "frame_pts_time": frame_pts_time,
                "title": target_metadata["title"],
                "height": target_metadata["height"],
                "width": target_metadata["width"],
                "duration": target_metadata["duration"],
                "url": target_metadata["url"],
                "webpage_url": target_metadata["webpage_url"],
                "blackdetect": blackdetect,
                "scene_threshold": scene_threshold,
            })
        total_files += target_metadata['num_frames']
        total_videos += 1
    pd.DataFrame(df_rows).to_parquet(f"{target_shard_dir}.parquet", index=False)
    return source_shard, total_files, total_videos

def starprocess_shard(args):
    return process_shard(*args)

def run_parquet(source_dir, parquet_dir, target_dir, workers=4, blackdetect='d=0.1:pix_th=0.1', scene_threshold=0.2):
    total_files, total_videos = 0, 0
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(), console=console) as progress:
        shards = os.listdir(os.path.join(source_dir, parquet_dir))
        shards = sorted(list(filter(lambda x: os.path.isdir(os.path.join(source_dir, parquet_dir, x)), shards)))
        shard_task = progress.add_task(f"[green]Shards", total=len(shards))
        os.makedirs(os.path.join(target_dir, parquet_dir), exist_ok=True)
        with Pool(workers) as p:
            for r in p.imap_unordered(starprocess_shard, [(source_dir, parquet_dir, source_shard, target_dir, blackdetect, scene_threshold) for source_shard in shards]):
                source_shard, files, videos = r
                console.log(f"Shard: {source_shard}, Files: {files}, Videos: {videos}")
                total_files += files
                total_videos += videos
                console.log(f"Total files: {total_files}, Total videos: {total_videos}")
                progress.update(shard_task, advance=1)

        progress.remove_task(shard_task)
        return total_files, total_videos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from videos and store them.")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing the videos.")
    parser.add_argument("--parquet_name", type=str, required=True, help="Name of the parquet file.")
    parser.add_argument("--target", type=str, required=True, help="Target directory to store the extracted images.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use.")
    parser.add_argument("--blackdetect", default="d=0.1:pix_th=0.1", type=str, help="Blackdetect filter parameters.")
    parser.add_argument("--scene_threshold", default=0.1, type=float, help="Scene threshold for ffmpeg.")

    args = parser.parse_args()

    source = args.source
    parquet_name = args.parquet_name
    target = args.target
    workers = args.workers

    console.log(f'Start extracting images from {parquet_name}')
    console.log(f'{source} -> {target}')
    total_files, total_videos = run_parquet(source, parquet_name, target, workers, blackdetect=args.blackdetect, scene_threshold=args.scene_threshold)
    console.log(f"Total files: {total_files}, Total videos: {total_videos}")
    console.log(f'Done {parquet_name}')
