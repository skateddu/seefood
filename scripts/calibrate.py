"""Calibrate the hotdog threshold and report test metrics.

Uses the "Hotdog or not a hotdog" dataset (Apache-2.0) from the Hugging Face Hub:
https://huggingface.co/datasets/antareepdey/Hotdog-or-not-a-hotdog

The threshold maximizing F1 on the train split is evaluated on the test split.
Train images that also appear in the test split are dropped to avoid leakage.
"""

import argparse
import hashlib
import logging
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seefood.classifier import HotdogClassifier
from seefood.config import ClassifierConfig
from seefood.preprocess import load_image, preprocess

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://huggingface.co/datasets/antareepdey/Hotdog-or-not-a-hotdog/resolve/main/data.zip"
)
DEFAULT_DATA_DIR = Path("data/raw/hotdog-or-not-a-hotdog")
LABEL_DIRS = {"hotdog": True, "others": False}
NAIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Sample:
    """A labeled image file."""

    path: Path
    true_label: bool


@dataclass(frozen=True)
class Metrics:
    """Binary classification metrics, with hotdog as the positive class."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def download_dataset(data_dir: Path) -> Path:
    """Download and extract the dataset unless it is already present.

    Parameters
    ----------
    data_dir : Path
        Path to the directory that holds the archive and its extracted content.

    Returns
    -------
    Path
        Root of the extracted dataset, containing ``train/`` and ``test/``.
    """
    root = data_dir / "data"
    if root.is_dir():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "data.zip"
    logger.info("Downloading %s", DATASET_URL)
    urllib.request.urlretrieve(url=DATASET_URL, filename=archive)
    with zipfile.ZipFile(file=archive) as zip_file:
        zip_file.extractall(path=data_dir)
    return root


def list_samples(split_path: Path) -> list[Sample]:
    """List the labeled images of a dataset split.

    Parameters
    ----------
    split_path : Path
        Path to the split directory with one sub-directory per label.

    Returns
    -------
    list[Sample]
        Samples sorted by path.
    """
    return [
        Sample(path, true_label=is_hotdog)
        for label_dir, is_hotdog in LABEL_DIRS.items()
        for path in sorted((split_path / label_dir).glob(pattern="*.jpg"))
    ]


def file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's content.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        Hex digest of the file's content.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_leaked_samples(train: Sequence[Sample], test: Sequence[Sample]) -> list[Sample]:
    """Remove train samples whose file content also appears in the test split.

    Parameters
    ----------
    train : Sequence[Sample]
        Train samples.
    test : Sequence[Sample]
        Test samples.

    Returns
    -------
    list[Sample]
        Train samples that do not leak into the test split.
    """
    test_digests = {file_digest(sample.path) for sample in test}
    kept = [sample for sample in train if file_digest(sample.path) not in test_digests]
    logger.info("Dropped %d train images duplicated in test", len(train) - len(kept))
    return kept


def chunks(items: Sequence[Sample], size: int) -> Iterator[Sequence[Sample]]:
    """Yield consecutive slices of ``items`` with at most ``size`` elements.

    Parameters
    ----------
    items : Sequence[Sample]
        Items to slice.
    size : int
        Maximum number of items per slice.

    Yields
    ------
    Sequence[Sample]
        Consecutive slices of ``items``.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def score_samples(
    classifier: HotdogClassifier,
    samples: Sequence[Sample],
    config: ClassifierConfig,
    batch_size: int,
) -> np.ndarray:
    """Compute the hotdog probability of every sample.

    Parameters
    ----------
    classifier : HotdogClassifier
        Classifier to evaluate.
    samples : Sequence[Sample]
        Images to score.
    config : ClassifierConfig
        Preprocessing configuration.
    batch_size : int
        Number of images per inference call.

    Returns
    -------
    np.ndarray
        Probabilities aligned with ``samples``.
    """
    scores = []
    for batch in chunks(items=samples, size=batch_size):
        tensors = [preprocess(image=load_image(sample.path), config=config) for sample in batch]
        scores.append(classifier.hotdog_probabilities(batch=np.concatenate(tensors)))
    return np.concatenate(scores)


def compute_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    """Compute accuracy, precision, recall and F1 at a given threshold.

    Parameters
    ----------
    probabilities : np.ndarray
        Hotdog probabilities.
    labels : np.ndarray
        Boolean ground truth, True for hotdog.
    threshold : float
        Minimum probability to predict hotdog.

    Returns
    -------
    Metrics
        Metrics with hotdog as the positive class.
    """
    predicted = probabilities >= threshold
    true_positives = int(np.sum(predicted & labels))
    precision = true_positives / max(int(predicted.sum()), 1)
    recall = true_positives / max(int(labels.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(predicted == labels))
    return Metrics(threshold, accuracy, precision, recall, f1)


def best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return the threshold, among observed probabilities, that maximizes F1.

    Parameters
    ----------
    probabilities : np.ndarray
        Hotdog probabilities.
    labels : np.ndarray
        Boolean ground truth, True for hotdog.

    Returns
    -------
    float
        The F1-optimal threshold.
    """
    thresholds = np.unique(probabilities)
    f1_scores = [compute_metrics(probabilities, labels, threshold).f1 for threshold in thresholds]
    return float(thresholds[int(np.argmax(f1_scores))])


def format_report(rows: dict[str, Metrics]) -> str:
    """Render metrics as a Markdown table.

    Parameters
    ----------
    rows : dict[str, Metrics]
        Metrics keyed by row label.

    Returns
    -------
    str
        A Markdown table.
    """
    lines = [
        "| | Threshold | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {name} | {metrics.threshold:.4f} | {metrics.accuracy:.1%} | {metrics.precision:.1%} "
        f"| {metrics.recall:.1%} | {metrics.f1:.3f} |"
        for name, metrics in rows.items()
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Calibrate on the train split and print a Markdown report for the test split."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    args = parse_args()
    root = download_dataset(args.data_dir)
    train = list_samples(split_path=root / "train")
    test = list_samples(split_path=root / "test")
    train = remove_leaked_samples(train, test)
    logger.info("Train: %d images, test: %d images", len(train), len(test))

    config = ClassifierConfig()
    classifier = HotdogClassifier.from_config(config)
    splits = {"train": train, "test": test}
    scores = {
        name: score_samples(classifier, samples, config, args.batch_size)
        for name, samples in splits.items()
    }
    labels = {
        name: np.array([sample.true_label for sample in samples])
        for name, samples in splits.items()
    }

    threshold = best_threshold(probabilities=scores["train"], labels=labels["train"])
    rows = {
        f"Naive ({NAIVE_THRESHOLD})": compute_metrics(
            probabilities=scores["test"], labels=labels["test"], threshold=NAIVE_THRESHOLD
        ),
        "Calibrated": compute_metrics(
            probabilities=scores["test"], labels=labels["test"], threshold=threshold
        ),
    }
    print(format_report(rows))


if __name__ == "__main__":
    main()
