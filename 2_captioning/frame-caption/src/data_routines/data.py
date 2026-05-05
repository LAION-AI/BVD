import webdataset as wds
import logging
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
import traceback


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def filter_no_image(sample):
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    logging.warning(traceback.format_exc())
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample or "data" not in filesample:
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def get_data_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    meta_key: str = "info.json",
    filter_functions: list = None,
    mapper_functions: list = None,
    convert_to_tuples: bool = True,
):
    pipeline = [wds.SimpleShardList(data_path)]
    pipeline.extend(
        [
            tarfile_to_samples_nothrow,
            wds.decode("pilrgb", handler=log_and_continue),
            wds.select(filter_no_image),
            wds.rename(
                image="jpg;png;jpeg;webp", json=meta_key, handler=log_and_continue
            ),
        ]
    )

    if filter_functions is not None:
        for filter_function in filter_functions:
            pipeline.append(wds.select(filter_function, handler=log_and_continue))

    extra_keys = set()
    if mapper_functions is not None:
        for mapper_function in mapper_functions:
            output_key = mapper_function.config.output_key
            extra_keys.add(output_key)
            pipeline.append(wds.map(mapper_function, handler=log_and_continue))

    if convert_to_tuples:
        tuple_keys = ["image", "json", "__url__", "__key__"]
        tuple_keys.extend(extra_keys)
        pipeline.extend(
            [
                wds.to_tuple(*tuple_keys),
            ]
        )

    pipeline.append(
        wds.batched(
            batch_size,
            partial=False,
        )
    )

    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        persistent_workers=num_workers > 0,
    )
    return dataloader
