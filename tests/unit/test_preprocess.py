from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image

from seefood.config import ClassifierConfig
from seefood.exceptions import ImageLoadError
from seefood.preprocess import (
    center_crop,
    load_image,
    preprocess,
    resize_shorter_side,
    to_tensor,
)


def test_load_image_grayscale_file_returns_rgb(tmp_path: Path):
    path = tmp_path / "gray.png"
    Image.new(mode="L", size=(10, 10), color=128).save(fp=path)

    assert load_image(path).mode == "RGB"


def test_load_image_exif_rotation_returns_upright_image(tmp_path: Path):
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    Image.new(mode="RGB", size=(40, 20)).save(fp=path, exif=exif)

    assert load_image(path).size == (20, 40)


def test_load_image_missing_file_raises_image_load_error(tmp_path: Path):
    with pytest.raises(ImageLoadError) as exc_info:
        load_image(path=tmp_path / "missing.jpg")

    assert exc_info.value.path == tmp_path / "missing.jpg"


def test_load_image_non_image_file_raises_image_load_error(tmp_path: Path):
    path = tmp_path / "notes.jpg"
    path.write_text("definitely not a hotdog")

    with pytest.raises(ImageLoadError):
        load_image(path)


@pytest.mark.parametrize(
    "size,expected",
    [((400, 300), (384, 288)), ((300, 400), (288, 384)), ((100, 100), (288, 288))],
    ids=["landscape", "portrait", "square-upscale"],
)
def test_resize_shorter_side_keeps_aspect_ratio(
    make_image: Callable[..., Image.Image], size: tuple[int, int], expected: tuple[int, int]
):
    width, height = size

    resized = resize_shorter_side(image=make_image(width=width, height=height), size=288)

    assert resized.size == expected


def test_center_crop_takes_the_middle_square():
    image = Image.new(mode="RGB", size=(5, 3), color=(0, 0, 0))
    image.putpixel(xy=(2, 1), value=(255, 255, 255))

    cropped = center_crop(image, size=1)

    assert cropped.getpixel(xy=(0, 0)) == (255, 255, 255)


def test_to_tensor_normalizes_with_mean_and_std(make_image: Callable[..., Image.Image]):
    image = make_image(width=2, height=2, color=(255, 0, 51))

    tensor = to_tensor(image, mean=(0.5, 0.0, 0.1), std=(0.5, 1.0, 0.1))

    np.testing.assert_allclose(actual=tensor[0, :, 0, 0], desired=[1.0, 0.0, 1.0], atol=1e-6)


def test_preprocess_returns_float32_nchw_batch(
    make_image: Callable[..., Image.Image], config: ClassifierConfig
):
    tensor = preprocess(image=make_image(width=640, height=480), config=config)

    assert tensor.shape == (1, 3, config.image_size, config.image_size)
    assert tensor.dtype == np.float32
