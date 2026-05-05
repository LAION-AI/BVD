import dataclasses
import pyarrow.parquet as pq
import os
from typing import Literal


@dataclasses.dataclass
class AddFieldsMapperConfig:
    type: Literal["add_fields"]
    input_key: str
    output_key: str
    metadata_path: str
    data_path: str
    metadata_key: str = "key"


class AddFieldsMapper:
    def __init__(self, config: AddFieldsMapperConfig):
        self.config = config
        self.data_path = self.config.data_path
        base_name = os.path.basename(self.data_path).replace(".tar", "")
        self.metadata_path = os.path.join(
            self.config.metadata_path, f"{base_name}.parquet"
        )
        self.metadata_key = self.config.metadata_key
        self.metadata = self.load_metadata()

    def load_metadata(self) -> dict:
        metadata = {}
        table = pq.read_table(self.metadata_path)
        metadata_pandas = table.to_pandas()
        for _, row in metadata_pandas.iterrows():
            metadata[row[self.metadata_key]] = row.to_dict()
        return metadata

    def __call__(self, data: dict) -> dict:
        key = data["__key__"]
        metadata = self.metadata[key]
        data[self.config.output_key] = metadata[self.config.input_key]
        return data
