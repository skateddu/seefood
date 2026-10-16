from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from PIL import Image

from seefood.config import ClassifierConfig


@dataclass(frozen=True)
class FakeInput:
    name: str


class FakeSession:
    """Stands in for onnxruntime.InferenceSession, returning a fixed hotdog probability."""

    def __init__(self, hotdog_probability: float, hotdog_class_index: int) -> None:
        self.hotdog_probability = hotdog_probability
        self.hotdog_class_index = hotdog_class_index
        self.received: list[dict[str, Any]] = []

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(name="image")]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        self.received.append(input_feed)
        batch_size = input_feed["image"].shape[0]
        probabilities = np.zeros(shape=(batch_size, 1000), dtype=np.float32)
        probabilities[:, self.hotdog_class_index] = self.hotdog_probability
        return [probabilities]


class SessionFactory(Protocol):
    def __call__(self, hotdog_probability: float) -> FakeSession: ...


@pytest.fixture
def config() -> ClassifierConfig:
    return ClassifierConfig()


@pytest.fixture
def make_session(config: ClassifierConfig) -> SessionFactory:
    def factory(hotdog_probability: float) -> FakeSession:
        return FakeSession(hotdog_probability, config.hotdog_class_index)

    return factory


@pytest.fixture
def make_image() -> Callable[..., Image.Image]:
    def factory(width: int = 64, height: int = 48, color: tuple[int, int, int] = (200, 30, 30)):
        return Image.new(mode="RGB", size=(width, height), color=color)

    return factory


@pytest.fixture
def image_file(tmp_path: Path, make_image: Callable[..., Image.Image]) -> Path:
    path = tmp_path / "food.png"
    make_image().save(fp=path)
    return path
