import json
import logging
import runpy
import sys
from pathlib import Path

import pytest

from seefood import __version__
from seefood.classifier import HotdogClassifier
from seefood.config import ClassifierConfig
from seefood.exceptions import ModelLoadError
from seefood.main import HOTDOG_LABEL, NOT_HOTDOG_LABEL, ExitCode, main
from tests.conftest import SessionFactory


class RecordingFactory:
    """Builds classifiers on a fake session and records the configs it receives."""

    def __init__(self, make_session: SessionFactory, hotdog_probability: float) -> None:
        self.make_session = make_session
        self.hotdog_probability = hotdog_probability
        self.configs: list[ClassifierConfig] = []

    def __call__(self, config: ClassifierConfig) -> HotdogClassifier:
        self.configs.append(config)
        return HotdogClassifier(self.make_session(self.hotdog_probability), config)


@pytest.fixture
def hotdog_factory(make_session: SessionFactory) -> RecordingFactory:
    return RecordingFactory(make_session, hotdog_probability=0.7)


@pytest.fixture
def not_hotdog_factory(make_session: SessionFactory) -> RecordingFactory:
    return RecordingFactory(make_session, hotdog_probability=0.001)


def test_main_hotdog_image_prints_hotdog_label(
    hotdog_factory: RecordingFactory, image_file: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(argv=[str(image_file)], classifier_factory=hotdog_factory)

    assert exit_code == ExitCode.OK
    assert capsys.readouterr().out.startswith(HOTDOG_LABEL)


def test_main_not_hotdog_image_prints_not_hotdog_label(
    not_hotdog_factory: RecordingFactory, image_file: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(argv=[str(image_file)], classifier_factory=not_hotdog_factory)

    assert exit_code == ExitCode.OK
    assert capsys.readouterr().out.startswith(NOT_HOTDOG_LABEL)


def test_main_json_flag_prints_one_object_per_image(
    hotdog_factory: RecordingFactory, image_file: Path, capsys: pytest.CaptureFixture[str]
):
    main(argv=["--json", str(image_file), str(image_file)], classifier_factory=hotdog_factory)

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [
        {"path": str(image_file), "is_hotdog": True, "hotdog_probability": 0.7},
        {"path": str(image_file), "is_hotdog": True, "hotdog_probability": 0.7},
    ]


def test_main_missing_image_logs_error_and_classifies_the_others(
    hotdog_factory: RecordingFactory,
    image_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
):
    missing = tmp_path / "missing.jpg"

    exit_code = main(argv=[str(missing), str(image_file)], classifier_factory=hotdog_factory)

    assert exit_code == ExitCode.ERROR
    assert "missing.jpg" in caplog.text
    assert str(image_file) in capsys.readouterr().out


def test_main_threshold_option_reaches_config(hotdog_factory: RecordingFactory, image_file: Path):
    main(argv=["--threshold", "0.9", str(image_file)], classifier_factory=hotdog_factory)

    assert hotdog_factory.configs[0].threshold == 0.9


def test_main_threshold_above_probability_answers_not_hotdog(
    hotdog_factory: RecordingFactory, image_file: Path, capsys: pytest.CaptureFixture[str]
):
    main(argv=["--threshold", "0.9", str(image_file)], classifier_factory=hotdog_factory)

    assert capsys.readouterr().out.startswith(NOT_HOTDOG_LABEL)


def test_main_invalid_threshold_logs_error_and_returns_error(
    hotdog_factory: RecordingFactory, image_file: Path, caplog: pytest.LogCaptureFixture
):
    exit_code = main(argv=["--threshold", "2", str(image_file)], classifier_factory=hotdog_factory)

    assert exit_code == ExitCode.ERROR
    assert "threshold" in caplog.text


def test_main_model_load_error_logs_error_and_returns_error(
    image_file: Path, caplog: pytest.LogCaptureFixture
):
    def failing_factory(config: ClassifierConfig) -> HotdogClassifier:
        raise ModelLoadError(path=config.model_path, reason="file not found")

    exit_code = main(argv=[str(image_file)], classifier_factory=failing_factory)

    assert exit_code == ExitCode.ERROR
    assert "file not found" in caplog.text


def test_main_without_images_exits_with_usage_error(hotdog_factory: RecordingFactory):
    with pytest.raises(SystemExit) as exc_info:
        main(argv=[], classifier_factory=hotdog_factory)

    assert exc_info.value.code == 2


def test_main_version_flag_prints_version(
    hotdog_factory: RecordingFactory, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit):
        main(argv=["--version"], classifier_factory=hotdog_factory)

    assert capsys.readouterr().out.strip() == f"seefood {__version__}"


def test_main_verbose_flag_enables_package_debug_logging(
    hotdog_factory: RecordingFactory, image_file: Path
):
    main(argv=["-v", str(image_file)], classifier_factory=hotdog_factory)

    assert logging.getLogger("seefood").level == logging.DEBUG


def test_python_m_seefood_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["seefood", "--version"])

    with pytest.raises(SystemExit):
        runpy.run_module("seefood", run_name="__main__")

    assert __version__ in capsys.readouterr().out
