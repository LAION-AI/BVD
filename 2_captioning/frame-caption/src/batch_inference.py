from vllm import LLM, SamplingParams
from open_clip_models import OpenClipModel
import fire
from visualization import plot_grid
import pyarrow as pa
import ray
import shutil
import os
import glob
import torch
from tqdm import tqdm
from clip_score_utils import ClipScoreProcessor
from prompts import PROMPT_MAP
from extra_args import EXTRA_ARGS_MAP
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
import gc
import contextlib
from data_routines.data import get_data_loader
from data_routines.writer import ParquetSampleWriter
import logging
import time
import re

logging.basicConfig(level=logging.INFO)


def cleanup_vllm(llm):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_done_shards(out_dir):
    done_shards = []
    for f in os.listdir(out_dir):
        if f.endswith(".parquet"):
            done_shards.append(f.split(".")[0])
    return done_shards


def get_clipscores(
    data_path,
    parquet_path,
    clip_model="openai/clip-vit-base-patch32",
    meta_key="info.json",
    img_key="image",
    output_dir="output",
    shard_id=0,
    batch_size=32,
):
    logging.info(f"Computing CLIP scores with model: {clip_model}")

    clip_score = ClipScoreProcessor(clip_model)
    dataloader = get_data_loader(
        data_path, batch_size=batch_size, num_workers=1, meta_key=meta_key
    )

    metadata_pq = pa.parquet.read_table(parquet_path).to_pandas()

    images, captions = [], []

    for batch_id, (imgs, _, _, _) in tqdm(enumerate(dataloader)):
        if batch_id > 0:
            break

        for i, img in enumerate(imgs):
            row = metadata_pq.iloc[i]
            row = row.to_dict()
            caption = row["txt"]
            captions.append(caption)
            images.append(img)

    n_col = 4
    n_row = (
        len(images) // n_col if len(images) % n_col == 0 else len(images) // n_col + 1
    )
    try:
        clip_scores = clip_score(captions, images)
        captions = [f"{c} [{score:.2f}]" for c, score in zip(captions, clip_scores)]
        plot_grid(
            images,
            n_col,
            n_row,
            captions,
            orig_captions=None,
            fig_path=os.path.join(output_dir, f"{shard_id}.png"),
        )
        # for i, score in enumerate(clip_scores):
        #     metadata_pq.loc[i, "clip_score"] = score

        avg_score = sum(clip_scores) / len(clip_scores)
        logging.info(f"Average CLIP score: {avg_score:.2f}")

    except Exception as e:
        logging.error(f"Error computing CLIP scores: {e}")
        clip_scores = [0] * len(images)
    # save the metadata with clip scores
    # pq = pa.Table.from_pandas(metadata_pq)
    # pq.write_table(os.path.join(output_dir, f"{shard_id}.parquet"))
    return clip_scores


def clean_caption(caption):
    # both closing and opening tags
    html_tags = re.compile(r"<.*?>")
    caption = re.sub(html_tags, "", caption)
    # closing tags
    closing_tags = re.compile(r"</.*?>")
    caption = re.sub(closing_tags, "", caption)
    # opening tags
    caption = caption.replace("<", "").replace(">", "")
    return caption





@ray.remote
class ShardProcessor:
    def __init__(
        self,
        model_path,
        output_dir,
        tp,
        backend,
        compute_clip_score=False,
        clip_model="openai/clip-vit-base-patch32",
        language_only=False,
        language_only_key="txt",
        debug=False,
        max_model_len=4096,
    ):
        cuda_devices = torch.cuda.device_count()
        print(f"Found {cuda_devices} cuda devices")
        self.model_path = model_path
        self.output_dir = output_dir
        self.tp = tp
        self.backend = backend
        self.max_model_len = max_model_len
        self.model = get_model(model_path, 0, backend, tp, max_model_len=self.max_model_len)
        self.compute_clip_score = compute_clip_score
        self.clip_model = clip_model
        self.language_only = language_only
        self.language_only_key = language_only_key
        self.debug = debug

        assert self.backend in ["vllm", "open_clip"], "Unknown backend, must be vllm or open_clip"
        if self.language_only:
            assert self.language_only and backend == "vllm", "Language only is only supported for VLLM"
        assert self.language_only is False or self.language_only_key is not None, "Language only key must be provided if language only is True"

        self.tokenizer = self.model.get_tokenizer() if self.backend == "vllm" else None


    def format_prompt_language_only(self, prompt, txt_data):
        prompt = prompt.format(txt=txt_data)
        # prompt = self.tokenizer.apply_chat_template([
        #     {
        #         "role": "user",
        #         "content": prompt,
        #     }
        # ]
        # )
        return prompt

    def process_shard(
        self,
        shard_id,
        data_path,
        prompt,
        sampling_params,
        batch_size,
        num_workers,
        img_key="image",
        meta_key="info.json",
    ):
        dataloader = get_data_loader(
            data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            meta_key=meta_key,
        )
        schema = pa.schema(
            [
                pa.field("txt", pa.string()),
                pa.field("prompt", pa.string()),
                pa.field("url", pa.string()),
                pa.field("key", pa.string()),
            ]
        )

        base_name = os.path.basename(data_path).replace(".tar", "")
        print(f"Processing shard: {base_name}")
        writer = ParquetSampleWriter(
            shard_id=base_name,
            output_folder=self.output_dir,
            save_caption=True,
            oom_shard_count=0,
            schema=schema,
            encode_format="jpg",
        )
        batch_id = 0

        for batch_id, (imgs, metas, urls, keys) in tqdm(enumerate(dataloader)):
            if self.debug and batch_id > 0:
                break
            prompt = prompt.format(txt=metas[i][self.language_only_key]) if self.language_only_key else prompt
            prompt = prompt[:self.max_model_len]
            inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img,
                    },
                } if not self.language_only else self.format_prompt_language_only(prompt, metas[i][self.language_only_key])
                for i, img in enumerate(imgs)
            ]
            captions = self.model.generate(
                inputs,
                sampling_params=sampling_params,
            )

            captions = [c.outputs[0].text for c in captions]
            captions = [clean_caption(c) for c in captions]
            for i in range(len(imgs)):
                meta_data = metas[i]
                if "key" in meta_data:
                    meta_data["key"] = keys[i]
                meta_data["url"] = urls[i]
                meta_data["prompt"] = prompt
                writer.write(
                    key=keys[i],
                    caption=captions[i],
                    meta=meta_data,
                )

        writer.close()

        # if self.backend == "vllm":
        #     cleanup_vllm(self.model)

        if self.compute_clip_score:
            get_clipscores(
                data_path,
                writer.pq_file,
                meta_key=meta_key,
                img_key=img_key,
                output_dir=self.output_dir,
                shard_id=shard_id,
                clip_model=self.clip_model,
            )

        logging.info(f"Processed {batch_id + 1} batches")
        logging.info(os.path.join(self.output_dir, f"{shard_id}.png"))

        # tmp_dir = os.path.join(output_dir, "_tmp")

        # with open(os.path.join(tmp_dir, f"{shard_id}.feather"), "w") as f:
        #     f.write("done")


@ray.remote
def process_shard(
    shard_id,
    data_path,
    model_path,
    output_dir,
    tp,
    rank=0,
    img_key="image",
    meta_key="info.json",
    prompt="",
    sampling_params=None,
    batch_size=32,
    num_workers=4,
    backend="vllm",
    compute_clip_score=False,
    clip_model="openai/clip-vit-base-patch32",
    debug=False,
):
    model = get_model(model_path, rank, backend, tp)
    dataloader = get_data_loader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        meta_key=meta_key,
    )
    schema = pa.schema(
        [
            pa.field("txt", pa.string()),
            pa.field("prompt", pa.string()),
            pa.field("url", pa.string()),
            pa.field("key", pa.string()),
        ]
    )

    base_name = os.path.basename(data_path).replace(".tar", "")
    print(f"Processing shard: {base_name}")
    writer = ParquetSampleWriter(
        shard_id=base_name,
        output_folder=output_dir,
        save_caption=True,
        oom_shard_count=0,
        schema=schema,
        encode_format="jpg",
    )
    batch_id = 0

    for batch_id, (imgs, metas, urls, keys) in tqdm(enumerate(dataloader)):
        if debug and batch_id > 0:
            break

        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": img,
                },
            }
            for img in imgs
        ]
        for i in range(3):
            try:
                captions = model.generate(
                    inputs,
                    sampling_params=sampling_params,
                )
                break
            except Exception as e:
                logging.error(
                    f"Got error: {e} in shard: {shard_id}, {i + 1}/3 retrying..."
                )
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)

                if i == 2:
                    logging.error(f"Failed to generate captions for shard: {shard_id}")
                    return

        torch.cuda.empty_cache()
        captions = [c.outputs[0].text for c in captions]
        captions = [clean_caption(c) for c in captions]
        for i in range(len(imgs)):
            meta_data = metas[i]
            if "key" in meta_data:
                meta_data["key"] = keys[i]
            meta_data["url"] = urls[i]
            meta_data["prompt"] = prompt
            writer.write(
                key=keys[i],
                caption=captions[i],
                meta=meta_data,
            )

    writer.close()

    if backend == "vllm":
        cleanup_vllm(model)

    logging.info(f"Computing CLIP scores for shard: {shard_id}")

    if compute_clip_score:
        get_clipscores(
            data_path,
            writer.pq_file,
            meta_key=meta_key,
            img_key=img_key,
            output_dir=output_dir,
            shard_id=shard_id,
            clip_model=clip_model,
        )

    logging.info(f"Processed {batch_id + 1} batches")
    logging.info(os.path.join(output_dir, f"{shard_id}.png"))

    # tmp_dir = os.path.join(output_dir, "_tmp")

    # with open(os.path.join(tmp_dir, f"{shard_id}.feather"), "w") as f:
    #     f.write("done")


def get_model(model_path, rank, backend="vllm", tensor_parallel_size=1, max_model_len=4096):
    """
    Get the model based on the backend

    Args:
        model_path (str): Path to the model
        rank (int): Rank of the process
        backend (str): Backend to use (vllm or open_clip)
    """

    if backend == "vllm":
        extra_args = EXTRA_ARGS_MAP.get("/".join(model_path.split("/")[-2:]), {})
        logging.warning(f"Extra args: {extra_args}")
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_num_seqs=32,
            dtype="bfloat16",
            **extra_args,
        )
    elif backend == "open_clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OpenClipModel(model_path, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return model


def batch_inference(
    model_path: str,
    data_path: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.6,
    tp: int = 1,
    dp: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = "output",
    img_key: str = "image",
    meta_key: str = "info.json",
    backend: str = "vllm",
    compute_clip_score: bool = False,
    clip_model: str = "openai/clip-vit-base-patch32",
    debug: bool = False,
    limit: int = None,
    shuffle_shards: bool = False,
    language_only: bool = False,
    language_only_key: str = None,
):
    """
    Run batch inference on a dataset

    Args:

        model_path (str): Path to the model
        data_path (str): Path to the dataset
        prompt (str): Prompt to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        tp (int): Tensor parallel size
        dp (int): Data parallel size
        batch_size (int): Batch size
        num_workers (int): Number of webdataset loader workers
        output_dir (str): Output directory
        img_key (str): Key for the image data
        meta_key (str): Key for the metadata
        backend (str): Backend to use (vllm or open_clip)
        compute_clip_score (bool): Whether to compute CLIP scores
        clip_model (str): Model name or path to CLIP model to use for scoring
        debug (bool): Debug mode
        limit (int): Limit the number of shards to process
    """

    logging.info(f"Running batch inference with model: {model_path}")

    s = time.time()

    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if backend == "vllm":
        sampling_params = SamplingParams(**sampling_params)
        prompt_func = PROMPT_MAP.get("/".join(model_path.split("/")[-2:]), lambda x: x)
        prompt = prompt_func(prompt)

    shards = glob.glob(f"{data_path}/*.tar")
    logging.info(f"Found {len(shards)} shards")
    cuda_devices = torch.cuda.device_count()
    logging.info(f"Found {cuda_devices} cuda devices")
    num_gpus_per_task = cuda_devices // dp
    logging.info(f"Using {num_gpus_per_task} GPUs per task")
    shards = shards[:limit] if limit is not None else shards
    logging.info(f"Limiting to {len(shards)} shard(s)")

    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_shards = get_done_shards(output_dir)
    print(done_shards)

    try:
        ray.init(address="auto", include_dashboard=False)
    except ConnectionError:
        ray.init(include_dashboard=False)

    print("ray resources", ray.available_resources())
    ray_num_total_gpus = ray.available_resources().get("GPU", 0)
    print("ray num total gpus", ray_num_total_gpus)

    num_actors = int(ray_num_total_gpus // num_gpus_per_task)
    print("num actors", num_actors)

    actors = []

    for i in range(num_actors):
        shard_processor = ShardProcessor.options(
            num_gpus=num_gpus_per_task, num_cpus=8
        ).remote(
            model_path=model_path,
            output_dir=output_dir,
            tp=tp,
            backend=backend,
            compute_clip_score=compute_clip_score,
            clip_model=clip_model,
            language_only=language_only,
            language_only_key=language_only_key
        )
        actors.append(shard_processor)

    # futures = [
    #     process_shard.options(num_gpus=num_gpus_per_task, num_cpus=8).remote(
    #         shard_id=shard_id,
    #         data_path=shards[shard_id],
    #         model_path=model_path,
    #         output_dir=output_dir,
    #         tp=tp,
    #         prompt=prompt,
    #         img_key=img_key,
    #         meta_key=meta_key,
    #         sampling_params=sampling_params,
    #         backend=backend,
    #         compute_clip_score=compute_clip_score,
    #         clip_model=clip_model,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         debug=debug,
    #     )
    #     for shard_id in range(len(shards))
    #     if os.path.basename(shards[shard_id]).replace(".tar", "") not in done_shards
    # ]

    futures = []
    for shard_id in range(len(shards)):
        if os.path.basename(shards[shard_id]).replace(".tar", "") not in done_shards:
            actor_id = shard_id % len(actors)
            shard_processor = actors[actor_id]
            futures.append(
                shard_processor.process_shard.remote(
                    shard_id=shard_id,
                    data_path=shards[shard_id],
                    prompt=prompt,
                    sampling_params=sampling_params,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    meta_key=meta_key,
                    img_key=img_key,
                )
            )
    ray.get(futures)

    ray.shutdown()
    shutil.rmtree(tmp_dir)

    e = time.time()
    logging.info(f"Total time: {e - s:.2f}s")


if __name__ == "__main__":
    fire.Fire(batch_inference)
