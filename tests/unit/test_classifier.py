from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seefood.classifier import HotdogClassifier, load_session
from seefood.config import ClassifierConfig
from seefood.exceptions import ImageLoadError, ModelLoadError
from tests.conftest import SessionFactory


@pytest.mark.parametrize(
    "probability,expected",
    [(0.9, True), (0.025, True), (0.0249, False), (0.0, False)],
    ids=["confident", "at-threshold", "just-below", "zero"],
)
def test_predict_image_applies_threshold(
    make_session: SessionFactory,
    make_image: Callable[..., Image.Image],
    config: ClassifierConfig,
    probability: float,
    expected: bool,
):
    classifier = HotdogClassifier(
        session=make_session(hotdog_probability=probability), config=config
    )

    prediction = classifier.predict_image(image=make_image())

    assert prediction.is_hotdog is expected
    assert prediction.hotdog_probability == pytest.approx(probability)


def test_predict_image_feeds_preprocessed_tensor_to_session(
    make_session: SessionFactory,
    make_image: Callable[..., Image.Image],
    config: ClassifierConfig,
):
    session = make_session(hotdog_probability=0.5)
    classifier = HotdogClassifier(session, config)

    classifier.predict_image(image=make_image())

    assert session.received[0]["image"].shape == (1, 3, config.image_size, config.image_size)


def test_hotdog_probabilities_returns_one_value_per_image(
    make_session: SessionFactory, config: ClassifierConfig
):
    classifier = HotdogClassifier(session=make_session(hotdog_probability=0.3), config=config)
    batch = np.zeros(shape=(4, 3, config.image_size, config.image_size), dtype=np.float32)

    probabilities = classifier.hotdog_probabilities(batch)

    np.testing.assert_allclose(actual=probabilities, desired=[0.3] * 4)


def test_predict_reads_image_from_disk(
    make_session: SessionFactory, config: ClassifierConfig, image_file: Path
):
    classifier = HotdogClassifier(session=make_session(hotdog_probability=0.8), config=config)

    assert classifier.predict(path=image_file).is_hotdog


def test_predict_missing_image_raises_image_load_error(
    make_session: SessionFactory, config: ClassifierConfig, tmp_path: Path
):
    classifier = HotdogClassifier(session=make_session(hotdog_probability=0.8), config=config)

    with pytest.raises(ImageLoadError):
        classifier.predict(path=tmp_path / "missing.jpg")


def test_load_session_missing_model_raises_model_load_error(tmp_path: Path):
    with pytest.raises(ModelLoadError, match="file not found"):
        load_session(model_path=tmp_path / "missing.onnx")


def test_load_session_corrupted_model_raises_model_load_error(tmp_path: Path):
    model_path = tmp_path / "broken.onnx"
    model_path.write_bytes(b"not a protobuf")

    with pytest.raises(ModelLoadError) as exc_info:
        load_session(model_path)

    assert exc_info.value.path == model_path


def test_from_config_missing_model_raises_model_load_error(tmp_path: Path):
    config = ClassifierConfig(model_path=tmp_path / "missing.onnx")

    with pytest.raises(ModelLoadError):
        HotdogClassifier.from_config(config)
