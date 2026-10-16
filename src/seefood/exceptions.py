"""Exception hierarchy for SeeFood."""

from pathlib import Path


class SeeFoodError(Exception):
    """Base exception for SeeFood."""


class ConfigError(SeeFoodError, ValueError):
    """Raised when a configuration value is out of its valid range.

    Attributes
    ----------
    field_name : str
        Name of the invalid configuration field.
    value : object
        The rejected value.
    """

    def __init__(self, field_name: str, value: object, reason: str) -> None:
        super().__init__(f"Invalid {field_name}={value!r}: {reason}")
        self.field_name = field_name
        self.value = value


class ImageLoadError(SeeFoodError):
    """Raised when an image file cannot be opened or decoded.

    Attributes
    ----------
    path : Path
        Path of the image that failed to load.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Cannot load image '{path}': {reason}")
        self.path = path


class ModelLoadError(SeeFoodError):
    """Raised when the ONNX model cannot be loaded.

    Attributes
    ----------
    path : Path
        Path of the model that failed to load.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Cannot load model '{path}': {reason}")
        self.path = path
