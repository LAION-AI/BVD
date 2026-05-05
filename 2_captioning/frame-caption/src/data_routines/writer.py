import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
import webdataset as wds


class BufferedParquetWriter:
    """Write samples to parquet files incrementally with a buffer"""

    def __init__(self, output_file, schema, buffer_size=100):
        self.buffer_size = buffer_size
        self.schema = schema
        self._initiatlize_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)
        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)

    def _initiatlize_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def write(self, sample):
        if self.current_buffer_size >= self.buffer_size:
            self.flush()
        self._add_sample_to_buffer(sample)

    def flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._initiatlize_buffer()

    def close(self):
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class ParquetSampleWriter:
    """ParquetSampleWriter is a image+caption writer to parquet"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        shard_name = shard_id
        output_file = f"{output_folder}/{shard_name}.parquet"
        self.pq_file = output_file
        self.buffered_parquet_writer = BufferedParquetWriter(output_file, schema, 100)
        self.save_caption = save_caption

    def write(self, key, caption, meta):
        sample = {"key": key}

        sample["txt"] = str(caption) if caption is not None else ""
        sample.update(meta)
        self.buffered_parquet_writer.write(sample)

    def close(self):
        self.buffered_parquet_writer.close()


class TarWriter:
    def __init__(self, pattern):
        self.pattern = pattern
        self.sink = wds.TarWriter(pattern)

    def write(self, sample):
        self.sink.write(sample)

    def close(self):
        self.sink.close()
