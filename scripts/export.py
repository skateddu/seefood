"""Export torchvision's ImageNet EfficientNet-B2 to ONNX, with softmax baked into the graph.

Requires the ``export`` dependency group: ``uv sync --group export``.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from seefood.config import DEFAULT_MODEL_PATH, ClassifierConfig
from seefood.preprocess import preprocess

logger = logging.getLogger(__name__)

WEIGHTS = EfficientNet_B2_Weights.IMAGENET1K_V1
# Lowest opset supported by the torch.export-based exporter; onnxruntime-web runs it fine.
OPSET_VERSION = 18
TOLERANCE = 1e-4


def build_model() -> nn.Module:
    """Return EfficientNet-B2 with ImageNet weights followed by a softmax, in eval mode."""
    return nn.Sequential(efficientnet_b2(weights=WEIGHTS), nn.Softmax(dim=1)).eval()


def export_onnx(model: nn.Module, output: Path, image_size: int) -> None:
    """Export the model to ONNX with a dynamic batch dimension.

    Parameters
    ----------
    model : nn.Module
        Model to export, in eval mode.
    output : Path
        Destination ONNX file.
    image_size : int
        Height and width of the model input.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(2, 3, image_size, image_size)
    torch.onnx.export(
        model,
        args=(dummy,),
        f=str(output),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    logger.info("Exported %s (%.1f MB)", output, output.stat().st_size / 1e6)


def check_model_parity(model: nn.Module, output: Path, image_size: int) -> None:
    """Check that ONNX Runtime reproduces PyTorch outputs on random inputs.

    Parameters
    ----------
    model : nn.Module
        Reference PyTorch model.
    output : Path
        Exported ONNX file.
    image_size : int
        Height and width of the model input.

    Raises
    ------
    AssertionError
        If outputs differ by more than ``TOLERANCE``.
    """
    inputs = torch.randn(2, 3, image_size, image_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(inputs).numpy()
    session = onnxruntime.InferenceSession(
        path_or_bytes=str(output), providers=["CPUExecutionProvider"]
    )
    actual = session.run(output_names=None, input_feed={"image": inputs.numpy()})[0]
    np.testing.assert_allclose(actual, desired=expected, atol=TOLERANCE)
    logger.info("Model parity OK (max abs diff %.2e)", np.abs(actual - expected).max())


def check_preprocessing_parity(config: ClassifierConfig) -> None:
    """Check that the numpy preprocessing matches torchvision's transforms for these weights.

    Parameters
    ----------
    config : ClassifierConfig
        Configuration used by the runtime preprocessing.

    Raises
    ------
    AssertionError
        If the two pipelines produce tensors differing by more than ``TOLERANCE``.
    """
    rng = np.random.default_rng(seed=0)
    pixels = rng.integers(low=0, high=256, size=(375, 500, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    expected = WEIGHTS.transforms()(image).unsqueeze(dim=0).numpy()
    actual = preprocess(image, config)
    np.testing.assert_allclose(actual, desired=expected, atol=TOLERANCE)
    logger.info("Preprocessing parity OK (max abs diff %.2e)", np.abs(actual - expected).max())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="ONNX file path")
    return parser.parse_args()


def main() -> None:
    """Export the model and verify both model and preprocessing parity."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    args = parse_args()
    config = ClassifierConfig()
    model = build_model()
    export_onnx(model, args.output, config.image_size)
    check_model_parity(model, args.output, config.image_size)
    check_preprocessing_parity(config)


if __name__ == "__main__":
    main()
