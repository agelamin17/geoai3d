"""Tests for Parquet at-rest reading and writing."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pytest
from numpy.typing import NDArray

from geoai3d import PointCloud, Provenance, read_parquet, to_parquet


def _make_cloud() -> PointCloud:
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    attributes: dict[str, NDArray[Any]] = {
        "intensity": np.array([10, 20, 30], dtype=np.uint16),
        "planarity": np.array([0.1, 0.5, 0.9], dtype=np.float64),
    }
    provenance = Provenance(source="scan.laz")
    provenance.add_step("voxel subsampling", {"voxel_size": 0.5})
    return PointCloud(xyz, attributes=attributes, crs=28992, provenance=provenance)


def test_round_trip_preserves_coordinates_exactly(tmp_path: Path) -> None:
    cloud = _make_cloud()
    path = tmp_path / "rt.parquet"
    to_parquet(cloud, path)
    loaded = read_parquet(path)
    # Parquet stores float64 losslessly, so coordinates are exact.
    np.testing.assert_array_equal(loaded.xyz, cloud.xyz)


def test_round_trip_preserves_attributes_and_dtypes(tmp_path: Path) -> None:
    cloud = _make_cloud()
    path = tmp_path / "attrs.parquet"
    to_parquet(cloud, path)
    loaded = read_parquet(path)
    np.testing.assert_array_equal(
        loaded.attribute("intensity"), cloud.attribute("intensity")
    )
    assert loaded.attribute("intensity").dtype == np.uint16
    np.testing.assert_array_equal(
        loaded.attribute("planarity"), cloud.attribute("planarity")
    )


def test_round_trip_preserves_crs(tmp_path: Path) -> None:
    path = tmp_path / "crs.parquet"
    to_parquet(_make_cloud(), path)
    crs = read_parquet(path).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_round_trip_preserves_provenance(tmp_path: Path) -> None:
    path = tmp_path / "prov.parquet"
    to_parquet(_make_cloud(), path)
    provenance = read_parquet(path).provenance
    assert provenance is not None
    assert provenance.source == "scan.laz"
    assert provenance.steps[0].description == "voxel subsampling"
    assert provenance.steps[0].parameters == {"voxel_size": 0.5}


def test_write_without_crs_raises(tmp_path: Path) -> None:
    cloud = PointCloud(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="without a coordinate reference"):
        to_parquet(cloud, tmp_path / "x.parquet")


def test_reserved_attribute_name_raises(tmp_path: Path) -> None:
    cloud = PointCloud(np.zeros((3, 3)), attributes={"x": np.ones(3)}, crs=28992)
    with pytest.raises(ValueError, match="reserved coordinate column"):
        to_parquet(cloud, tmp_path / "x.parquet")


def test_read_non_geoai3d_parquet_raises(tmp_path: Path) -> None:
    path = tmp_path / "foreign.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), str(path))
    with pytest.raises(ValueError, match="missing coordinate columns"):
        read_parquet(path)


def test_read_with_crs_override(tmp_path: Path) -> None:
    path = tmp_path / "rt.parquet"
    to_parquet(_make_cloud(), path)
    crs = read_parquet(path, crs=4326).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4326
