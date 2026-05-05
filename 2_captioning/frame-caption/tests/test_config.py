import os
import yaml
from data_routines.dist_data_processing import DataConfig
from data_routines.filters import FILTERS
import pytest


@pytest.fixture
def test_data_path():
    return os.path.join(os.path.dirname(__file__), "test_data")


def test_data_config(test_data_path):
    config_path = os.path.join(test_data_path, "test.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = DataConfig(**config)
    filters = []
    for ff in data_config.filter_functions:
        filter_type = ff["type"]
        filter = FILTERS[filter_type]
        filter_config = filter["config"](**ff)
        filter_fn = filter["filter"](filter_config)
        filters.append(filter_fn)
    assert data_config.batch_size == 32
    assert data_config.num_workers == 4
    assert data_config.meta_key == "info.json"
    assert len(filters) == 1
    assert filters[0].config.threshold == 0.8
    assert filters[0].config.type == "mono"
