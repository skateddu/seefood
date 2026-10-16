import pytest

from seefood.config import ClassifierConfig
from seefood.exceptions import ConfigError


@pytest.mark.parametrize("threshold", [0.0, 0.025, 1.0], ids=["zero", "default", "one"])
def test_config_valid_threshold_is_accepted(threshold: float):
    assert ClassifierConfig(threshold=threshold).threshold == threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.5], ids=["negative", "above-one"])
def test_config_threshold_out_of_range_raises_config_error(threshold: float):
    with pytest.raises(ConfigError, match="threshold"):
        ClassifierConfig(threshold=threshold)


def test_config_non_positive_image_size_raises_config_error():
    with pytest.raises(ConfigError, match="image_size"):
        ClassifierConfig(image_size=0)


def test_config_error_is_a_value_error():
    with pytest.raises(ValueError):
        ClassifierConfig(threshold=2.0)
