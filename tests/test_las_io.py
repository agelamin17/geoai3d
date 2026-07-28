"""Tests for LAS/LAZ reading and writing."""

from pathlib import Path
from typing import Any

import laspy
import numpy as np
import pyproj
import pytest
from numpy.typing import NDArray

from geoai3d import PointCloud, read_las, to_las


def _make_cloud(crs: object = 28992) -> PointCloud:
    xyz = np.array(
        [[100.0, 200.0, 30.0], [101.5, 200.5, 31.25], [102.0, 201.0, 32.0]]
    )
    attributes: dict[str, NDArray[Any]] = {
        "intensity": np.array([10, 20, 30], dtype=np.uint16),
        "classification": np.array([2, 2, 6], dtype=np.uint8),
        "gps_time": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "planarity": np.array([0.1, 0.5, 0.9], dtype=np.float64),
    }
    return PointCloud(xyz, attributes=attributes, crs=crs)


def _write_plain_las_without_crs(path: Path) -> None:
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.zeros(3)
    header.scales = np.full(3, 0.001)
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0])
    las.y = np.array([1.0, 2.0])
    las.z = np.array([1.0, 2.0])
    las.write(str(path))


def test_round_trip_preserves_attributes_and_crs(tmp_path: Path) -> None:
    cloud = _make_cloud()
    path = tmp_path / "rt.las"
    to_las(cloud, path)
    loaded = read_las(path)

    assert len(loaded) == 3
    np.testing.assert_allclose(loaded.xyz, cloud.xyz, atol=1e-3)
    np.testing.assert_array_equal(
        loaded.attribute("intensity"), cloud.attribute("intensity")
    )
    np.testing.assert_array_equal(
        loaded.attribute("classification"), cloud.attribute("classification")
    )
    np.testing.assert_allclose(
        loaded.attribute("gps_time"), cloud.attribute("gps_time")
    )


def test_round_trip_preserves_extra_dimension(tmp_path: Path) -> None:
    cloud = _make_cloud()
    path = tmp_path / "extra.las"
    to_las(cloud, path)
    loaded = read_las(path)

    np.testing.assert_allclose(
        loaded.attribute("planarity"), cloud.attribute("planarity")
    )
    assert loaded.attribute("planarity").dtype == np.float64


def test_round_trip_preserves_crs(tmp_path: Path) -> None:
    path = tmp_path / "crs.las"
    to_las(_make_cloud(crs=28992), path)
    crs = read_las(path).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_read_records_provenance(tmp_path: Path) -> None:
    path = tmp_path / "prov.las"
    to_las(_make_cloud(), path)
    provenance = read_las(path).provenance

    assert provenance is not None
    assert provenance.steps[0].description == "read LAS/LAZ"
    assert provenance.steps[0].parameters["point_format"] == 6


def test_reads_externally_authored_file(tmp_path: Path) -> None:
    # A file written by laspy directly, not by to_las, to prove we can read
    # clouds we did not produce.
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.zeros(3)
    header.scales = np.full(3, 0.01)
    header.add_crs(pyproj.CRS.from_epsg(28992))
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([4.0, 5.0, 6.0])
    las.z = np.array([7.0, 8.0, 9.0])
    las.intensity = np.array([5, 6, 7], dtype=np.uint16)
    path = tmp_path / "external.las"
    las.write(str(path))

    loaded = read_las(path)
    np.testing.assert_array_equal(
        loaded.attribute("intensity"), np.array([5, 6, 7], dtype=np.uint16)
    )
    crs = loaded.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_read_without_crs_raises(tmp_path: Path) -> None:
    path = tmp_path / "nocrs.las"
    _write_plain_las_without_crs(path)
    with pytest.raises(ValueError, match="no coordinate reference system"):
        read_las(path)


def test_read_with_crs_override(tmp_path: Path) -> None:
    path = tmp_path / "nocrs.las"
    _write_plain_las_without_crs(path)
    crs = read_las(path, crs=28992).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_write_without_crs_raises(tmp_path: Path) -> None:
    cloud = PointCloud(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="without a coordinate reference"):
        to_las(cloud, tmp_path / "x.las")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_las(tmp_path / "does_not_exist.las")


def test_laz_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lazrs")
    cloud = _make_cloud()
    path = tmp_path / "rt.laz"
    to_las(cloud, path)
    loaded = read_las(path)

    np.testing.assert_allclose(loaded.xyz, cloud.xyz, atol=1e-3)
    np.testing.assert_array_equal(
        loaded.attribute("intensity"), cloud.attribute("intensity")
    )
