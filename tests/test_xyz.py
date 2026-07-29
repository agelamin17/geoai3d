"""Tests for XYZ text reading and writing."""

from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import pytest
from numpy.typing import NDArray

from geoai3d import PointCloud, read_xyz, to_xyz


def test_round_trip_xyz_only(tmp_path: Path) -> None:
    cloud = PointCloud(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), crs=28992)
    path = tmp_path / "pts.xyz"
    to_xyz(cloud, path)
    loaded = read_xyz(path)
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded.xyz, cloud.xyz)
    crs = loaded.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_writes_prj_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "pts.xyz"
    to_xyz(PointCloud(np.zeros((2, 3)), crs=28992), path)
    prj = path.with_suffix(".prj")
    assert prj.exists()
    text = prj.read_text(encoding="utf-8")
    assert "PROJCRS" in text or "PROJCS" in text


def test_round_trip_with_attribute(tmp_path: Path) -> None:
    attributes: dict[str, NDArray[Any]] = {"planarity": np.array([0.1, 0.5])}
    cloud = PointCloud(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        attributes=attributes,
        crs=28992,
    )
    path = tmp_path / "pts.xyz"
    to_xyz(cloud, path)
    loaded = read_xyz(path, columns=("x", "y", "z", "planarity"))
    np.testing.assert_allclose(
        loaded.attribute("planarity"), cloud.attribute("planarity")
    )


def test_read_with_explicit_crs(tmp_path: Path) -> None:
    path = tmp_path / "pts.xyz"
    to_xyz(PointCloud(np.zeros((2, 3)), crs=28992), path, write_prj=False)
    crs = read_xyz(path, crs=4326).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4326


def test_read_without_crs_or_prj_raises(tmp_path: Path) -> None:
    path = tmp_path / "pts.xyz"
    to_xyz(PointCloud(np.zeros((2, 3)), crs=28992), path, write_prj=False)
    with pytest.raises(ValueError, match="no CRS"):
        read_xyz(path)


def test_columns_missing_axis_raises(tmp_path: Path) -> None:
    path = tmp_path / "pts.xyz"
    to_xyz(PointCloud(np.zeros((2, 3)), crs=28992), path)
    with pytest.raises(ValueError, match="must include 'z'"):
        read_xyz(path, columns=("x", "y"))


def test_column_count_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "pts.xyz"
    to_xyz(PointCloud(np.zeros((2, 3)), crs=28992), path)
    with pytest.raises(ValueError, match="column names were given"):
        read_xyz(path, columns=("x", "y", "z", "extra"), crs=28992)
