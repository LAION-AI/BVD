import ray
import yaml
import fire
import dataclasses
import glob
import pyarrow as pa
import os
from typing import List
from .data import get_data_loader
from .filters import FILTERS
from .mappers import MAPPERS
from .writer import ParquetSampleWriter, TarWriter
from visualization import plot_grid
import logging


@dataclasses.dataclass
class DataConfig:
    data_path: str
    batch_size: int
    num_workers: int
    meta_key: str
    img_key: str
    filter_functions: List[dict] = None
    mapper_functions: List[dict] = None
    output_dir: str = "output"
    save_tar: bool = False
    txt_key: str = None
    make_visualization: bool = False
    convert_to_tuples: bool = True


@ray.remote
def process_shard(
    shard_id: int,
    shard_path: str,
    output_dir: str,
    config: DataConfig,
    debug: bool = False,
):
    schema = [
        ("key", pa.string()),
    ]

    print(config)

    schema = pa.schema(schema)
    txt_key = config.txt_key
    img_key = config.img_key
    meta_key = config.meta_key
    writer = ParquetSampleWriter(
        shard_id=shard_id,
        output_folder=output_dir,
        save_caption=True,
        oom_shard_count=0,
        schema=schema,
        encode_format="jpg",
    )

    filters = []
    if config.filter_functions is None:
        config.filter_functions = []
    for ff in config.filter_functions:
        filter_type = ff["type"]
        filter = FILTERS[filter_type]
        filter_config = filter["config"](**ff)
        filter_fn = filter["filter"](filter_config)
        filters.append(filter_fn)

    mappers = []
    if config.mapper_functions is None:
        config.mapper_functions = []

    for mf in config.mapper_functions:
        mapper_type = mf["type"]
        mapper = MAPPERS[mapper_type]
        mapper_config = mapper["config"](**mf)
        mapper_config.data_path = shard_path
        mapper_fn = mapper["mapper"](mapper_config)
        mappers.append(mapper_fn)

    dataloader = get_data_loader(
        data_path=shard_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        meta_key=config.meta_key,
        filter_functions=filters,
        mapper_functions=mappers,
        convert_to_tuples=config.convert_to_tuples,
    )

    if config.save_tar:
        output_tar = os.path.join(output_dir, f"{shard_id}.tar")
        print(f"Saving tar to {output_tar}")
        tar_writer = TarWriter(output_tar)
    else:
        tar_writer = None

    print("tar_writer", tar_writer)

    try:
        for batch_id, batch in enumerate(dataloader):
            if isinstance(batch, tuple):
                imgs, metas, _, keys = batch[0], batch[1], batch[2], batch[3]
            else:
                imgs, metas, _, keys = (
                    batch["image"],
                    batch["json"],
                    batch["__url__"],
                    batch["__key__"],
                )
            if debug and batch_id > 0:
                break
            print(f"Processing batch: {batch_id}")
            if config.make_visualization and batch_id == 0:
                # imgs = [b[0] for b in batch]
                rows = len(imgs) // 4
                cols = 4
                captions = keys

                fig_path = os.path.join(output_dir, f"shard_{shard_id}.png")
                plot_grid(
                    imgs,
                    rows,
                    cols,
                    captions,
                    orig_captions=None,
                    figsize=(20, 12),
                    fig_path=fig_path,
                )
                print(f"Saved visualization to {fig_path}")

            for key, meta in zip(keys, metas):
                writer.write(key, meta[txt_key] if txt_key else None, meta)

            if tar_writer is not None:
                print(f"Writing to tar {output_tar}")

                if isinstance(batch, dict):
                    batch_list = [dict(zip(batch, t)) for t in zip(*batch.values())]
                    for sample in batch_list:
                        sample[img_key] = sample.pop("image")
                        sample[meta_key] = sample.pop("json")
                        tar_writer.write(sample)

    except Exception:
        print(f"Error processing shard {shard_id}")
        # remove tar file if it exists
        if config.save_tar:
            os.remove(output_tar)
        # remove parquet file if it exists
        os.remove(writer.parquet_path)
        return

    writer.close()
    if tar_writer is not None:
        tar_writer.close()

    print(f"Finished processing shard {shard_id}")
    print(f"Output written to {output_dir}")


def main(
    config_path: str = None,
    debug: bool = False,
    limit: int = None,
):
    assert config_path is not None, "Please provide a config file"
    config = yaml.safe_load(open(config_path))
    data_config = DataConfig(**config)

    shards = glob.glob(os.path.join(data_config.data_path, "*.tar"))
    if limit:
        shards = shards[:limit]
    print(f"Processing {len(shards)} shards")

    output_dir = data_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    done_shards = glob.glob(os.path.join(output_dir, "*.parquet")) + glob.glob(
        os.path.join(output_dir, "*.tar")
    )

    done_shards = [os.path.basename(shard).split(".")[0] for shard in done_shards]
    done_shards = set(done_shards)
    shards = [
        shard
        for shard in shards
        if os.path.basename(shard).split(".")[0] not in done_shards
    ]

    logging.info(f"Processing {len(shards)} shards")
    try:
        ray.init(address="auto", include_dashboard=False)
    except ConnectionError:
        ray.init(include_dashboard=False)
    ret = []
    for shard_id, shard_path in enumerate(shards):
        base_name = os.path.basename(shard_path).split(".")[0]
        ret.append(
            process_shard.remote(
                base_name, shard_path, data_config.output_dir, data_config, debug
            )
        )

    ray.get(ret)


if __name__ == "__main__":
    fire.Fire(main)
