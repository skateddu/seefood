from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from seefood.classifier import HotdogClassifier, load_session
from seefood.config import ClassifierConfig

pytestmark = pytest.mark.integration


def test_onnx_model_outputs_probability_distribution(config: ClassifierConfig):
    session = load_session(config.model_path)
    batch = np.random.default_rng(seed=0).standard_normal(
        size=(2, 3, config.image_size, config.image_size), dtype=np.float32
    )

    probabilities = session.run(output_names=None, input_feed={"image": batch})[0]

    assert probabilities.shape == (2, 1000)
    np.testing.assert_allclose(actual=probabilities.sum(axis=1), desired=[1.0, 1.0], atol=1e-4)


def test_classifier_plain_image_is_not_hotdog(
    config: ClassifierConfig, make_image: Callable[..., Image.Image]
):
    classifier = HotdogClassifier.from_config(config)

    prediction = classifier.predict_image(
        image=make_image(width=400, height=300, color=(40, 90, 200))
    )

    assert not prediction.is_hotdog
