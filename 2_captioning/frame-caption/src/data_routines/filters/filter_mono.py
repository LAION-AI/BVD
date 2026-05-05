import cv2
import numpy as np
import dataclasses
from typing import Literal
from .base import FilterConfig


@dataclasses.dataclass
class FilterMonoConfig(FilterConfig):
    type: Literal["mono"]
    threshold: float = 0.8
    is_monochromatic: bool = True


def is_not_monochromatic(sample: dict, threshold: float = 0.8) -> bool:
    """
    Check if an image is monochromatic by checking if the highest bin in any channel is significant.
    Args:
        image: The image to check.
        threshold: The threshold to consider a channel significant.
    Returns:
        True if the image is not monochromatic, False otherwise.
    """

    image = sample["image"]
    # convert from PIL image to numpy array
    image = np.array(image)
    # Convert image to HSV or just use BGR histogram
    histSize = 256
    # Calculate histogram per channel
    hist_b = cv2.calcHist([image], [0], None, [histSize], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [histSize], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [histSize], [0, 256])
    total_pixels = np.prod(image.shape[:2])

    # Find the highest bin in each channel
    max_b = np.max(hist_b)
    max_g = np.max(hist_g)
    max_r = np.max(hist_r)

    if (
        max_b / total_pixels > threshold
        or max_g / total_pixels > threshold
        or max_r / total_pixels > threshold
    ):
        return False
    return True


def is_monochromatic(sample: dict, threshold: float = 0.95) -> bool:
    """
    Check if an image is monochromatic by checking if the highest bin in any channel is significant.
    Args:
        image: The image to check.
        threshold: The threshold to consider a channel significant.
    Returns:
        True if the image is monochromatic, False otherwise.
    """

    return not is_not_monochromatic(sample, threshold)


class FilterMono:
    def __init__(self, config: FilterMonoConfig):
        self.config = config
        self.threshold = config.threshold
        self.is_monochromatic = config.is_monochromatic

    def __call__(self, sample: dict) -> bool:
        return is_monochromatic(sample, self.threshold) == self.is_monochromatic
