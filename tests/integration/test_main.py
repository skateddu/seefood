from pathlib import Path

import pytest

from seefood.main import NOT_HOTDOG_LABEL, ExitCode, main

pytestmark = pytest.mark.integration


def test_main_real_model_plain_image_is_not_hotdog(
    image_file: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(argv=[str(image_file)])

    assert exit_code == ExitCode.OK
    assert capsys.readouterr().out.startswith(NOT_HOTDOG_LABEL)
