"""Image loading and preprocessing, matching torchvision's ImageNet evaluation transforms."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from seefood.config import ClassifierConfig
from seefood.exceptions import ImageLoadError


def load_image(path: Path) -> Image.Image:
    """Open an image file, apply its EXIF orientation and convert it to RGB.

    Parameters
    ----------
    path : Path
        Image file to open.

    Returns
    -------
    Image.Image
        The decoded image in RGB mode, upright as the camera intended.

    Raises
    ------
    ImageLoadError
        If the file is missing, unreadable, not an image, or suspiciously large.
    """
    try:
        with Image.open(path) as image:
            # Phone photos are stored sideways with an EXIF rotation tag; browsers apply it too.
            return ImageOps.exif_transpose(image).convert(mode="RGB")
    except (OSError, Image.DecompressionBombError) as err:
        raise ImageLoadError(path, reason=str(err)) from err


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    """Resize so that the shorter side equals ``size``, keeping the aspect ratio.

    Parameters
    ----------
    image : Image.Image
        Input image.
    size : int
        Target length of the shorter side.

    Returns
    -------
    Image.Image
        The resized image, using bicubic interpolation.
    """
    width, height = image.size
    # Same rounding as torchvision.transforms.Resize, so ONNX and PyTorch see identical pixels.
    if width <= height:
        new_size = (size, int(size * height / width))
    else:
        new_size = (int(size * width / height), size)
    return image.resize(size=new_size, resample=Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop a ``size`` x ``size`` square from the center of the image.

    Parameters
    ----------
    image : Image.Image
        Input image, with both sides at least ``size`` pixels.
    size : int
        Side of the square crop.

    Returns
    -------
    Image.Image
        The cropped image.
    """
    width, height = image.size
    left = round((width - size) / 2)
    top = round((height - size) / 2)
    return image.crop(box=(left, top, left + size, top + size))


def to_tensor(
    image: Image.Image,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """Convert an RGB image to a normalized NCHW float32 tensor with batch size 1.

    Parameters
    ----------
    image : Image.Image
        RGB image.
    mean : tuple[float, float, float]
        Per-channel mean subtracted after scaling pixels to [0, 1].
    std : tuple[float, float, float]
        Per-channel standard deviation used to divide.

    Returns
    -------
    np.ndarray
        Array of shape ``(1, 3, height, width)`` and dtype float32.
    """
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return normalized.transpose(2, 0, 1)[np.newaxis]


def preprocess(image: Image.Image, config: ClassifierConfig) -> np.ndarray:
    """Apply the full evaluation pipeline: resize, center crop, normalize.

    Parameters
    ----------
    image : Image.Image
        RGB image of any size.
    config : ClassifierConfig
        Provides the image size and normalization statistics.

    Returns
    -------
    np.ndarray
        Model input of shape ``(1, 3, image_size, image_size)`` and dtype float32.
    """
    resized = resize_shorter_side(image, size=config.image_size)
    cropped = center_crop(image=resized, size=config.image_size)
    return to_tensor(image=cropped, mean=config.mean, std=config.std)
