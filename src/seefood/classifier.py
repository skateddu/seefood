"""Hotdog / not hotdog classification on top of an ImageNet ONNX model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from seefood.config import ClassifierConfig
from seefood.exceptions import ModelLoadError
from seefood.preprocess import load_image, preprocess

logger = logging.getLogger(__name__)


class InferenceSession(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` used by the classifier."""

    def get_inputs(self) -> Sequence[Any]:
        """Return the model input descriptors."""
        ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Run inference and return the model outputs."""
        ...


@dataclass(frozen=True)
class Prediction:
    """Outcome of classifying one image.

    Attributes
    ----------
    is_hotdog : bool
        Whether the hotdog probability reached the configured threshold.
    hotdog_probability : float
        Probability assigned by the model to the ImageNet hotdog class.
    """

    is_hotdog: bool
    hotdog_probability: float


def load_session(model_path: Path) -> InferenceSession:
    """Create an ONNX Runtime CPU session for the given model.

    Parameters
    ----------
    model_path : Path
        ONNX model file.

    Returns
    -------
    InferenceSession
        A ready-to-use ONNX Runtime session.

    Raises
    ------
    ModelLoadError
        If the file does not exist or ONNX Runtime cannot load it.
    """
    if not model_path.is_file():
        raise ModelLoadError(path=model_path, reason="file not found")
    # Imported lazily: onnxruntime is the slowest import and tests inject fake sessions.
    import onnxruntime
    from onnxruntime.capi import onnxruntime_pybind11_state as ort_errors

    load_errors = (
        ort_errors.Fail,
        ort_errors.InvalidArgument,
        ort_errors.InvalidGraph,
        ort_errors.InvalidProtobuf,
        ort_errors.NoSuchFile,
        ort_errors.NotImplemented,
    )
    try:
        session = onnxruntime.InferenceSession(
            path_or_bytes=str(model_path), providers=["CPUExecutionProvider"]
        )
    except load_errors as err:
        raise ModelLoadError(path=model_path, reason=str(err)) from err
    logger.debug("Loaded ONNX model %s", model_path, extra={"model_path": str(model_path)})
    return session


class HotdogClassifier:
    """Decides whether images contain a hotdog.

    Parameters
    ----------
    session : InferenceSession
        Session for a model mapping ``(N, 3, H, W)`` images to ``(N, 1000)`` probabilities.
    config : ClassifierConfig
        Preprocessing settings and decision threshold.
    """

    def __init__(self, session: InferenceSession, config: ClassifierConfig) -> None:
        self._session = session
        self._config = config
        self._input_name = session.get_inputs()[0].name

    @classmethod
    def from_config(cls, config: ClassifierConfig) -> HotdogClassifier:
        """Build a classifier, loading the ONNX model referenced by the config.

        Parameters
        ----------
        config : ClassifierConfig
            Configuration whose ``model_path`` points to the ONNX model.

        Returns
        -------
        HotdogClassifier
            A classifier backed by an ONNX Runtime session.

        Raises
        ------
        ModelLoadError
            If the model cannot be loaded.
        """
        return cls(session=load_session(config.model_path), config=config)

    def hotdog_probabilities(self, batch: np.ndarray) -> np.ndarray:
        """Return the hotdog probability for each preprocessed image in a batch.

        Parameters
        ----------
        batch : np.ndarray
            Preprocessed images of shape ``(N, 3, H, W)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with probabilities in [0, 1].
        """
        outputs = self._session.run(output_names=None, input_feed={self._input_name: batch})
        return np.asarray(outputs[0])[:, self._config.hotdog_class_index]

    def predict_image(self, image: Image.Image) -> Prediction:
        """Classify an already loaded RGB image.

        Parameters
        ----------
        image : Image.Image
            RGB image of any size.

        Returns
        -------
        Prediction
            The hotdog verdict and probability.
        """
        batch = preprocess(image, config=self._config)
        probability = float(self.hotdog_probabilities(batch)[0])
        return Prediction(
            is_hotdog=probability >= self._config.threshold,
            hotdog_probability=probability,
        )

    def predict(self, path: Path) -> Prediction:
        """Load an image file and classify it.

        Parameters
        ----------
        path : Path
            Image file.

        Returns
        -------
        Prediction
            The hotdog verdict and probability.

        Raises
        ------
        ImageLoadError
            If the image cannot be loaded.
        """
        return self.predict_image(image=load_image(path))
