"""Configuration of the model, preprocessing and hotdog decision rule."""

from dataclasses import dataclass
from pathlib import Path

from seefood.exceptions import ConfigError

DEFAULT_MODEL_PATH = Path(__file__).parent / "assets" / "efficientnet_b2.onnx"
IMAGENET_HOTDOG_CLASS = 934
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassifierConfig:
    """Settings for preprocessing and the hotdog decision rule.

    Attributes
    ----------
    model_path : Path
        ONNX model that outputs softmax probabilities over the ImageNet classes.
    threshold : float
        Minimum probability of the hotdog class to answer "Hotdog!".
    hotdog_class_index : int
        Index of ``hotdog, hot dog, red hot`` among the ImageNet-1k classes.
    image_size : int
        Shorter-side resize target, which is also the center-crop size (EfficientNet-B2 uses 288).
    mean : tuple[float, float, float]
        Per-channel normalization mean, in RGB order.
    std : tuple[float, float, float]
        Per-channel normalization standard deviation, in RGB order.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    threshold: float = 0.025
    hotdog_class_index: int = IMAGENET_HOTDOG_CLASS
    image_size: int = 288
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        """Validate value ranges.

        Raises
        ------
        ConfigError
            If ``threshold`` is outside [0, 1] or ``image_size`` is not positive.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigError(field_name="threshold", value=self.threshold, reason="not in [0, 1]")
        if self.image_size <= 0:
            raise ConfigError(field_name="image_size", value=self.image_size, reason="must be > 0")
