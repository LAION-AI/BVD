#!/usr/bin/env python3
"""Video captioning script using vLLM with Qwen3-VL on webdataset format."""

import argparse
import json
from pathlib import Path
import re
import io

import torch
import webdataset as wds
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from transformers import AutoProcessor
import math

# from torchcodec.decoders import VideoDecoder
from typing import Optional, Union, List, Any, Tuple
import torchvision.transforms

console = Console()
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = object()
    SamplingParams = object()
    console.log(
        "[yellow]Warning: vLLM not installed, using dummy model and sampling parameters[/yellow]"
    )
from qwen_vl_utils import process_vision_info

try:
    import ray

    @ray.remote
    class LLMActor:
        """Ray actor for distributed vLLM inference."""

        def __init__(
            self,
            model: str,
            tensor_parallel_size: int,
            gpu_memory_utilization: float,
            max_model_len: int,
            limit_mm_per_prompt: dict,
            mm_processor_kwargs: dict,
        ):
            """Initialize the LLM actor with vLLM model.

            Args:
                model: Model path or name
                tensor_parallel_size: Tensor parallel size
                gpu_memory_utilization: GPU memory utilization
                max_model_len: Maximum model length
                limit_mm_per_prompt: Multimodal limits
                mm_processor_kwargs: Multimodal processor kwargs
            """
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                limit_mm_per_prompt=limit_mm_per_prompt,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        def generate(self, inputs: list[dict], sampling_params: dict) -> list:
            """Generate captions for a batch of inputs.

            Args:
                inputs: List of input dictionaries
                sampling_params: Sampling parameters as dict

            Returns:
                List of generation outputs
            """
            sampling_params_obj = SamplingParams(**sampling_params)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params_obj)
            # Convert outputs to serializable format
            return [
                {
                    "text": output.outputs[0].text,
                    "prompt": output.prompt,
                }
                for output in outputs
            ]

except ImportError:
    ray = None
    LLMActor = None
    console.log(
        "[yellow]Warning: Ray not installed, data parallelism will not be available[/yellow]"
    )

try:
    import decord

    decord.bridge.set_bridge("torch")
except ImportError:
    decord = object()
    console.log(
        "[yellow]Warning: decord not installed, using dummy video decoder[/yellow]"
    )

# Default prompt for video captioning
DEFAULT_PROMPT = """Describe this video in detail. Include:
1. The main subjects and their actions
2. The setting and environment
3. Any notable visual elements, colors, or movements
4. The overall mood or atmosphere

Provide a comprehensive but concise description."""
NUM_THREADS = 1
NUM_FRAMES = 32
ALLOWED_VIDEO_EXTENSIONS = [
    "360p.mp4",
    ".360p.mp4",
    "mp4",
    ".mp4",
]

# FROM: AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct").image_processor.patch_size
IMAGE_PATCH_SIZE = 16
# FROM: https://github.com/QwenLM/Qwen3-VL/blob/e5c7e5c26af6a8bd65aec9388f3642cf6ea9d75c/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24C1-L38C37
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8

MODEL_SEQ_LEN = 32768


class KeyFilter:
    """Filters the dataset based on the key"""

    def __init__(self, enforce_keys: Union[str, List[str]] = ["mp4", "txt"]):
        if isinstance(enforce_keys, str):
            enforce_keys = [enforce_keys]
        self.enforce_keys = enforce_keys

    def __call__(self, sample):
        # print('KeyFilter', sample.keys())
        try:
            for key in self.enforce_keys:
                if key not in sample:
                    return False
            return True
        except Exception:  # pylint: disable=broad-except
            return False


class NoneFilter:
    """Filters the dataset based on the key"""

    def __init__(self, enforce_keys: Optional[Union[str, List[str]]] = ["mp4", "txt"]):
        if isinstance(enforce_keys, str):
            enforce_keys = [enforce_keys]
        self.enforce_keys = enforce_keys

    def __call__(self, sample):
        # print('KeyFilter', sample.keys())
        try:
            for key in self.enforce_keys:
                if sample[key] is None or sample[key] == "":
                    # print(f"NoneFilter: {sample['__key__']}")
                    return False
            return True
        except KeyError as e:
            console.log(f"KeyError: {e} in NoneFilter")
            return False
        except Exception as e:  # pylint: disable=broad-except
            console.log(f"Exception {type(e).__name__} in NoneFilter")
            return False


class ExistingIdsFilter:
    """Filters the dataset based on the existing IDs"""

    def __init__(self, existing_ids: List[str]):
        self.existing_ids = existing_ids

    def __call__(self, sample):
        return sample["__key__"] not in self.existing_ids


class Qwen3VideoDecoder:
    def __init__(
        self,
        image_patch_size: int = 16,
        spatial_merge_size: int = 4,
        backend: str = "decord",
        frame_factor: int = 2,
        num_frames: int = 32,
    ):
        self.image_patch_size = image_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.backend = backend
        self.video_min_token_num = VIDEO_MIN_TOKEN_NUM
        self.video_max_token_num = VIDEO_MAX_TOKEN_NUM
        self.frame_factor = frame_factor
        self.num_frames = num_frames

    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    # Based on https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        max_pixels = (
            max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor**2)
        )
        min_pixels = (
            min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor**2)
        )
        assert (
            max_pixels >= min_pixels
        ), "The max_pixels of image must be greater than or equal to min_pixels."
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def __call__(self, key, data: bytes):
        """Decode video bytes to frames using torchcodec.

        Args:
            video_bytes: Raw video bytes
            num_frames: Number of frames to sample uniformly

        Returns:
            Tensor of shape (num_frames, C, H, W) in uint8 format
        """
        extension = re.sub(r".*[.]", "", key)
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            return None

        # Based on Qwen3-VL/qwen-vl-utils: https://github.com/QwenLM/Qwen3-VL/blob/e5c7e5c26af6a8bd65aec9388f3642cf6ea9d75c/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L405C5-L407C79
        image_factor = self.image_patch_size * self.spatial_merge_size
        video_frame_min_pixels = self.video_min_token_num * image_factor * image_factor
        video_frame_max_pixels = self.video_max_token_num * image_factor * image_factor

        try:
            if self.backend == "decord":
                reader = decord.VideoReader(io.BytesIO(data))
                total_frames, video_fps = len(reader), reader.get_avg_fps()
                nframes = self.round_by_factor(self.num_frames, self.frame_factor)
                if total_frames <= nframes:
                    indices = list(range(total_frames))
                else:
                    indices = (
                        torch.linspace(0, total_frames - 1, nframes)
                        .round()
                        .long()
                        .tolist()
                    )
                frames = reader.get_batch(indices)
                frames = frames.permute(0, 3, 1, 2)  # Convert to TCHW format
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

            nframes, _, height, width = frames.shape
            min_pixels = video_frame_min_pixels
            total_pixels = MODEL_SEQ_LEN * image_factor * image_factor * 0.9
            max_pixels = max(
                min(video_frame_max_pixels, total_pixels / nframes * self.frame_factor),
                int(min_pixels * 1.05),
            )

            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            frames = torchvision.transforms.functional.resize(
                frames,
                [resized_height, resized_width],
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            metadata = {
                "fps": video_fps,
                "total_num_frames": total_frames,
                "video_backend": self.backend,
                "frames_indices": indices,
            }
            sample_fps = len(frames) / max(total_frames, 1e-6) * video_fps
            video_kwargs = {
                "do_sample_frames": False,  # sampling already happened here
                # "sample_fps": sample_fps,
            }

            return frames, metadata, video_kwargs
        except Exception as e:
            console.log(f"[yellow]Warning: Failed to decode video {key}: {e}[/yellow]")
            return ""


def create_webdataset_pipeline(
    tar_path: Path,
    existing_ids: Optional[List[str]] = None,
    enforce_keys: Union[str, List[str]] = ["mp4"],
    num_frames: int = NUM_FRAMES,
    frame_factor: int = FRAME_FACTOR,
    image_patch_size: int = IMAGE_PATCH_SIZE,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
    backend: str = "decord",
):
    """Create a webdataset pipeline to load videos from tar files.

    Args:
        tar_path: Path to directory containing .tar files or a single .tar file
        batch_size: Number of samples per batch
        existing_ids: List of existing sample IDs to skip

    Yields:
        Batches of (sample_id, video_bytes) tuples
    """
    tar_pattern = str(tar_path)
    if tar_path.is_dir():
        tar_files = list(tar_path.glob("*.tar"))
        if len(tar_files) == 1:
            tar_pattern = str(tar_files[0])
        else:
            tar_files = sorted(tar_files)
            tar_pattern = (
                str(tar_path)
                + "/{"
                + f"{tar_files[0].name.split('.')[0]}..{tar_files[-1].name.split('.')[0]}"
                + "}.tar"
            )

    console.log(f"Loading tars from: [bold]{tar_pattern}[/bold]")

    dataset = wds.WebDataset(tar_pattern, shardshuffle=False)
    dataset = dataset.select(KeyFilter(enforce_keys))
    if existing_ids:
        dataset = dataset.select(ExistingIdsFilter(existing_ids))

    # Decode video
    video_decoder = Qwen3VideoDecoder(
        image_patch_size=image_patch_size,
        spatial_merge_size=spatial_merge_size,
        backend=backend,
        frame_factor=frame_factor,
        num_frames=num_frames,
    )
    dataset = dataset.decode(video_decoder)
    dataset = dataset.select(NoneFilter(enforce_keys))
    # dataset = dataset.map(lambda x: {**x, "frames": x.pop("mp4")})  # rename key
    # .map(video_transformer)  # resizing and normalization happens in the Qwen3-VL model

    # dataset = dataset.batched(batch_size)

    return dataset


def create_dataloader(
    dataset: wds.WebDataset,
    batch_size: Optional[int] = None,
    num_workers: int = NUM_THREADS,
):
    """Create a dataloader from a webdataset pipeline.

    Args:
        dataset: WebDataset pipeline
        batch_size: Number of samples per batch

    Returns:
        DataLoader
    """
    return wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        persistent_workers=num_workers > 0,
    )


def prepare_vllm_inputs(
    processor: AutoProcessor,
    video_samples: list[tuple[str, tuple[torch.Tensor, dict, dict]]],
    prompt: str,
) -> tuple[list[str], list[dict]]:
    """Prepare inputs for vLLM inference.

    Args:
        processor: HuggingFace processor
        video_samples: List of (sample_id, video_data) tuples
        prompt: Prompt for video captioning

    Returns:
        Tuple of (sample_ids, inputs)
    """
    inputs = []
    sample_ids = []

    for sample_id, video in video_samples:
        sample_ids.append(sample_id)

        # Format message for Qwen3-VL
        frames, metadata, video_kwargs = video
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ""},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        llm_inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {
                "video": (frames, metadata),
            },
            "mm_processor_kwargs": video_kwargs,
        }

        inputs.append(llm_inputs)

    return sample_ids, inputs


def process_videos_with_vllm(
    llm: LLM,
    processor: AutoProcessor,
    video_samples: list[tuple[str, tuple[torch.Tensor, dict, dict]]],
    prompt: str,
    sampling_params: SamplingParams,
) -> list[tuple[str, str]]:
    """Process a batch of videos with vLLM.

    Args:
        llm: vLLM LLM instance
        video_samples: List of (sample_id, video_bytes) tuples
        prompt: Prompt for video captioning
        sampling_params: vLLM sampling parameters
        num_frames: Number of frames to sample from each video

    Returns:
        List of (sample_id, caption) tuples
    """
    sample_ids, inputs = prepare_vllm_inputs(processor, video_samples, prompt)

    # Generate captions
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Extract captions
    results = []
    for sample_id, output in zip(sample_ids, outputs):
        caption = output.outputs[0].text
        results.append((sample_id, caption))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate video captions using vLLM with Qwen3-VL on webdataset format"
    )
    parser.add_argument(
        "tar_path",
        type=Path,
        help="Path to directory containing .tar files or a single .tar file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file for captions (default: captions.json in tar_path directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model to use for captioning (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of videos to process per device in each batch. "
        "Total batch size = batch_size * data_parallel_size (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for generated captions (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt for video captioning",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU inference (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--data-parallel-size",
        "-dp",
        type=int,
        default=1,
        help="Data parallel size for multi-GPU inference. Each replica processes different data. "
        "Total GPUs used = tensor_parallel_size * data_parallel_size (default: 1)",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=32,
        help="Maximum number of frames to sample from each video (default: 32)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already captioned videos",
    )
    args = parser.parse_args()

    tar_path = args.tar_path.resolve()
    if not tar_path.exists():
        raise ValueError(f"Path does not exist: {tar_path}")

    # Set output path
    if args.output is None:
        if tar_path.is_dir():
            output_path = tar_path / "captions.json"
        else:
            output_path = tar_path.parent / "captions.json"
    else:
        output_path = args.output.resolve()

    # Load existing captions if resuming
    existing_captions = {}
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            existing_captions: dict[str, Any] = json.load(f)
        console.log(
            f"[blue]Resuming with {len(existing_captions)} existing captions[/blue]"
        )

    # Initialize Ray if data parallelism is requested
    use_ray_actors = args.data_parallel_size > 1
    llm_actors = None

    if use_ray_actors:
        if ray is None:
            console.log(
                "[red]Error: Ray is required for data parallelism but not installed.[/red]"
            )
            console.log("[yellow]Install with: pip install ray[/yellow]")
            console.log("[yellow]Falling back to data_parallel_size=1[/yellow]")
            args.data_parallel_size = 1
            use_ray_actors = False
        else:
            if not ray.is_initialized():
                console.log(
                    f"[bold blue]Initializing Ray for data parallelism (dp={args.data_parallel_size})...[/bold blue]"
                )
                # Initialize Ray without dashboard and metrics to avoid warnings in HPC environments
                ray.init(
                    include_dashboard=False,
                    _metrics_export_port=None,
                    _system_config={
                        "metrics_report_interval_ms": 0,
                    },
                )
            else:
                console.log("[blue]Ray already initialized[/blue]")

    try:
        # Initialize vLLM with Qwen3-VL
        console.log(f"[bold blue]Loading model: {args.model}[/bold blue]")

        # Common model arguments
        model_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": MODEL_SEQ_LEN,
            "limit_mm_per_prompt": {"video": 1},
            "mm_processor_kwargs": {
                "max_pixels": IMAGE_MAX_TOKEN_NUM
                * SPATIAL_MERGE_SIZE
                * SPATIAL_MERGE_SIZE,
            },
        }

        if use_ray_actors:
            # Create Ray actors for data parallelism
            console.log(
                f"[bold blue]Creating {args.data_parallel_size} Ray actors...[/bold blue]"
            )
            # Calculate GPUs per actor
            gpus_per_actor = args.tensor_parallel_size
            llm_actors = [
                LLMActor.options(num_gpus=gpus_per_actor).remote(**model_kwargs)
                for _ in range(args.data_parallel_size)
            ]
            llm = None
            console.log("[bold green]Ray actors created successfully.[/bold green]")
        else:
            # Single process mode
            llm = LLM(**model_kwargs)
            console.log("[bold green]Model loaded successfully.[/bold green]")

        sampling_params_dict = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": 0.95,
        }
        processor = AutoProcessor.from_pretrained(args.model)
        console.log(f"[bold blue]Processor loaded[/bold blue]")

        # Calculate effective batch size
        # When using Ray actors, load batch_size * num_actors samples
        # so each actor gets batch_size samples
        effective_batch_size = (
            args.batch_size * args.data_parallel_size if use_ray_actors else args.batch_size
        )
        console.log(
            f"[bold blue]Batch size per device: {args.batch_size}, "
            f"Effective batch size: {effective_batch_size}[/bold blue]"
        )

        # Create webdataset pipeline
        dataset = create_webdataset_pipeline(
            tar_path, existing_ids=list(existing_captions.keys())
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=NUM_THREADS,
            collate_fn=lambda x: x,
            prefetch_factor=10,
        )

        # Process videos
        captions = existing_captions.copy()
        processed_count = 0
        skipped_count = 0

        with Progress(
            # SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            # BarColumn(),
            # TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing videos...", total=None)

            for samples in dataloader:
                video_samples = [(sample["__key__"], sample["mp4"]) for sample in samples]

                if use_ray_actors:
                    # Distribute batch across Ray actors
                    sample_ids, inputs = prepare_vllm_inputs(
                        processor, video_samples, args.prompt
                    )

                    # Split inputs across actors
                    actor_count = len(llm_actors)
                    batch_size = len(inputs)
                    items_per_actor = (batch_size + actor_count - 1) // actor_count

                    # Create tasks for each actor
                    futures = []
                    actor_sample_ids = []
                    for i, actor in enumerate(llm_actors):
                        start_idx = i * items_per_actor
                        end_idx = min((i + 1) * items_per_actor, batch_size)
                        if start_idx < batch_size:
                            actor_inputs = inputs[start_idx:end_idx]
                            actor_ids = sample_ids[start_idx:end_idx]
                            futures.append(
                                actor.generate.remote(actor_inputs, sampling_params_dict)
                            )
                            actor_sample_ids.append(actor_ids)

                    # Collect results from all actors
                    actor_outputs = ray.get(futures)

                    # Combine results
                    results = []
                    for ids, outputs in zip(actor_sample_ids, actor_outputs):
                        for sample_id, output in zip(ids, outputs):
                            results.append((sample_id, output["text"]))
                else:
                    sampling_params = SamplingParams(
                        temperature=sampling_params_dict['temperature'],
                        max_tokens=sampling_params_dict['max_tokens'],
                        top_p=sampling_params_dict["top_p"],
                    )
                    # Single process mode
                    results = process_videos_with_vllm(
                        llm, processor, video_samples, args.prompt, sampling_params
                    )

                # console.log(results)

                for sample_id, caption in results:
                    captions[sample_id] = caption
                    processed_count += 1

                # # Save intermediate results
                # with open(output_path, "w") as f:
                #     json.dump(captions, f, indent=2, ensure_ascii=False)

                progress.update(
                    task,
                    description=f"[cyan]Processed {processed_count} video",
                )

        # Save final results
        with open(output_path, "w") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

        console.log(f"\n[bold green]✓ Captioning complete![/bold green]")
        console.log(f"  Processed: [bold]{processed_count}[/bold] videos")
        console.log(f"  Skipped: [bold]{skipped_count}[/bold] videos")
        console.log(f"  Total captions: [bold]{len(captions)}[/bold]")
        console.log(f"  Output saved to: [bold]{output_path}[/bold]")

    finally:
        # Cleanup Ray if it was initialized
        if args.data_parallel_size > 1 and ray is not None and ray.is_initialized():
            console.log("[blue]Shutting down Ray...[/blue]")
            ray.shutdown()


if __name__ == "__main__":
    main()
