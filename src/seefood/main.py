"""Command-line interface: tells you whether each image is a hotdog or not a hotdog."""

import argparse
import io
import json
import logging
import sys
from collections.abc import Callable, Sequence
from enum import IntEnum
from pathlib import Path

from seefood import __version__
from seefood.classifier import HotdogClassifier, Prediction
from seefood.config import DEFAULT_MODEL_PATH, ClassifierConfig
from seefood.exceptions import ImageLoadError, SeeFoodError

logger = logging.getLogger(__name__)

HOTDOG_LABEL = "🌭 Hotdog!"
NOT_HOTDOG_LABEL = "❌ Not hotdog!"

ClassifierFactory = Callable[[ClassifierConfig], HotdogClassifier]


class ExitCode(IntEnum):
    """Process exit codes (argparse itself exits with 2 on usage errors)."""

    OK = 0
    ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the ``seefood`` command.
    """
    parser = argparse.ArgumentParser(
        prog="seefood",
        description="Tells you if you have a hotdog. Or not a hotdog.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="image files to classify")
    parser.add_argument(
        "--threshold",
        type=float,
        default=ClassifierConfig.threshold,
        help="minimum hotdog probability to answer 'Hotdog!' (default: %(default)s)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="ONNX model file")
    parser.add_argument("--json", action="store_true", help="print one JSON object per image")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def format_prediction(path: Path, prediction: Prediction, as_json: bool) -> str:
    """Render the verdict for one image.

    Parameters
    ----------
    path : Path
        Classified image.
    prediction : Prediction
        Verdict and probability for the image.
    as_json : bool
        Whether to render a JSON object instead of human-readable text.

    Returns
    -------
    str
        One line of output.
    """
    if as_json:
        record = {
            "path": str(path),
            "is_hotdog": prediction.is_hotdog,
            "hotdog_probability": round(prediction.hotdog_probability, 6),
        }
        return json.dumps(record, ensure_ascii=False)
    label = HOTDOG_LABEL if prediction.is_hotdog else NOT_HOTDOG_LABEL
    return f"{label} {path} (p = {prediction.hotdog_probability:.3f})"


def classify_images(classifier: HotdogClassifier, paths: Sequence[Path], as_json: bool) -> bool:
    """Classify each image and print its verdict, logging images that cannot be loaded.

    Parameters
    ----------
    classifier : HotdogClassifier
        Classifier used for every image.
    paths : Sequence[Path]
        Image files to classify.
    as_json : bool
        Whether to print JSON objects instead of text.

    Returns
    -------
    bool
        True if every image was classified.
    """
    all_classified = True
    for path in paths:
        try:
            prediction = classifier.predict(path)
        except ImageLoadError as err:
            logger.error("%s", err)
            all_classified = False
            continue
        print(format_prediction(path, prediction, as_json))
    return all_classified


def configure_output(verbose: bool) -> None:
    """Set up logging on stderr and UTF-8 stdout, so the emoji survive Windows pipes.

    Parameters
    ----------
    verbose : bool
        Whether to log SeeFood's own messages at DEBUG level instead of WARNING.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("seefood").setLevel(logging.DEBUG if verbose else logging.WARNING)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    classifier_factory: ClassifierFactory = HotdogClassifier.from_config,
) -> int:
    """Run the ``seefood`` command.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, without the program name; ``None`` reads ``sys.argv``.
    classifier_factory : ClassifierFactory
        Builds the classifier from the configuration; replaced in tests.

    Returns
    -------
    int
        Exit code: 0 if every image was classified, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    configure_output(args.verbose)
    try:
        config = ClassifierConfig(model_path=args.model, threshold=args.threshold)
        classifier = classifier_factory(config)
    except SeeFoodError as err:
        logger.error("%s", err)
        return ExitCode.ERROR
    all_classified = classify_images(classifier, paths=args.images, as_json=args.json)
    return ExitCode.OK if all_classified else ExitCode.ERROR
