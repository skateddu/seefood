from importlib.metadata import version

import seefood


def test_version_matches_installed_metadata():
    assert seefood.__version__ == version("seefood")
