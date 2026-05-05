import os
from data_routines.filters.filter_mono import is_monochromatic
import cv2
import pytest


@pytest.fixture
def test_data_path():
    return os.path.join(os.path.dirname(__file__), "test_data")


def test_is_monochromatic(test_data_path):
    # Load a monochromatic image
    image_path = os.path.join(test_data_path, "solid_blue.jpg")
    image = cv2.imread(image_path)
    assert is_monochromatic({"image": image})

    # Load a non-monochromatic image
    image_path = os.path.join(test_data_path, "dog_img.jpg")
    image = cv2.imread(image_path)
    assert not is_monochromatic({"image": image})

    image_path = os.path.join(test_data_path, "mono_blue.jpg")
    image = cv2.imread(image_path)
    assert not is_monochromatic({"image": image}, threshold=0.95)

    image_path = os.path.join(test_data_path, "mono_blue.jpg")
    image = cv2.imread(image_path)
    assert is_monochromatic({"image": image}, threshold=0.6)
