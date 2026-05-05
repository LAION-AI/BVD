from .base import FilterConfig  # noqa
from .filter_mono import FilterMonoConfig, FilterMono  # noqa
# from .filter_imagenet import FilterImagenetConfig, FilterImagenet  # noqa

FILTERS = {
    "mono": {
        "config": FilterMonoConfig,
        "filter": FilterMono,
    },
    # "imagenet": {
    #     "config": FilterImagenetConfig,
    #     "filter": FilterImagenet,
    # },
}
