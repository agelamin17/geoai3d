"""Tests for read_lidar, the extension-dispatching reader."""

from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import pytest
from numpy.typing import NDArray

from geoai3d import PointCloud, read_lidar, to_las, to_xyz


def _make_cloud(crs: object = 28992) -> PointCloud:
    xyz = np.array([[100.0, 200.0, 30.0], [101.5, 200.5, 31.25], [102.0, 201.0, 32.0]])
    attributes: dict[str, NDArray[Any]] = {
        "intensity": np.array([10, 20, 30], dtype=np.uint16),
    }
    return PointCloud(xyz, attributes=attributes, crs=crs)


def test_reads_las_by_extension(tmp_path: Path) -> None:
    path = tmp_path / "scan.las"
    to_las(_make_cloud(), path)
    loaded = read_lidar(path)
    assert len(loaded) == 3
    crs = loaded.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_reads_uppercase_extension(tmp_path: Path) -> None:
    # Dispatch is case-insensitive: an upper-cased suffix still routes to LAS.
    path = tmp_path / "SCAN.LAS"
    to_las(_make_cloud(), path)
    assert len(read_lidar(path)) == 3


def test_reads_xyz_by_extension(tmp_path: Path) -> None:
    path = tmp_path / "cloud.xyz"
    # read_lidar reads text with read_xyz defaults (x, y, z), so use a
    # coordinates-only cloud: XYZ is a lossy exchange format and extra columns
    # are not carried through the high-level reader.
    cloud = PointCloud(np.zeros((3, 3)), crs=28992)
    to_xyz(cloud, path)  # writes a sibling .prj carrying the CRS
    loaded = read_lidar(path)
    assert len(loaded) == 3
    crs = loaded.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_txt_suffix_routes_to_xyz(tmp_path: Path) -> None:
    path = tmp_path / "cloud.txt"
    to_xyz(PointCloud(np.zeros((3, 3)), crs=28992), path)
    assert len(read_lidar(path)) == 3


def test_crs_argument_overrides_file(tmp_path: Path) -> None:
    path = tmp_path / "scan.las"
    to_las(_make_cloud(crs=28992), path)
    loaded = read_lidar(path, crs=4326)
    crs = loaded.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4326


def test_parquet_extension_is_redirected() -> None:
    # Parquet is the serialization format, not an acquisition format.
    with pytest.raises(ValueError, match="read_parquet"):
        read_lidar("cloud.parquet")


def test_unknown_extension_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported point-cloud format"):
        read_lidar("mesh.ply")
