"""Smoke tests for the package skeleton.

These exist so that continuous integration has something real to run from the
first commit onward. They are replaced by feature tests as modules land.
"""

import re

import geoai3d


def test_package_imports() -> None:
    """The package imports cleanly on a bare interpreter."""
    assert geoai3d is not None


def test_version_is_pep440_compatible() -> None:
    """__version__ is present and parseable as a simple release version."""
    pattern = r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev)\d+)?"
    assert re.fullmatch(pattern, geoai3d.__version__)
