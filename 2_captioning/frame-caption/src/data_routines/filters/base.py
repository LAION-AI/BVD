import dataclasses
from typing import Literal


@dataclasses.dataclass
class FilterConfig:
    type: Literal["mono", "blur", "noise"]
