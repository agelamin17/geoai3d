"""Tests for the streaming, disk-based out-of-core feature engine."""

from pathlib import Path

import laspy
import numpy as np
import pyproj
import pytest

from geoai3d import (
    geometric_features,
    geometric_features_stream,
    read_las,
    read_parquet,
)

_FEATURES = (
    "linearity",
    "planarity",
    "sphericity",
    "anisotropy",
    "omnivariance",
    "eigenentropy",
    "surface_variation",
    "verticality",
    "sum_eigenvalues",
)


def _scene(n: int = 2000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 10.0, (n, 3))
    xyz[:, 2] *= 0.3
    return xyz


def _write_las(path: Path, xyz: np.ndarray, *, with_crs: bool = True) -> None:
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = np.full(3, 0.001)
    if with_crs:
        header.add_crs(pyproj.CRS.from_epsg(28992))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(str(path))


def _write_las_with_attributes(
    path: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
    classification: np.ndarray,
) -> None:
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = np.full(3, 0.001)
    header.add_crs(pyproj.CRS.from_epsg(28992))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = intensity
    las.classification = classification
    las.write(str(path))


def test_stream_matches_in_memory_bitwise(tmp_path: Path) -> None:
    xyz = _scene()
    las_path = tmp_path / "in.las"
    _write_las(las_path, xyz)
    out_path = tmp_path / "out.parquet"
    radius = 0.6

    geometric_features_stream(las_path, out_path, radius=radius, tile_size=2.5)
    streamed = read_parquet(out_path)
    reference = geometric_features(read_las(las_path), radius=radius)

    order = np.argsort(streamed.attribute("index"))
    assert len(streamed) == len(reference)
    for name in _FEATURES:
        np.testing.assert_array_equal(
            streamed.attribute(name)[order],
            reference.attribute(name),
            err_msg=f"stream mismatch in {name!r}",
        )


def test_stream_parallel_matches_serial(tmp_path: Path) -> None:
    xyz = _scene()
    las_path = tmp_path / "in.las"
    _write_las(las_path, xyz)
    serial = tmp_path / "serial.parquet"
    parallel = tmp_path / "parallel.parquet"

    geometric_features_stream(las_path, serial, radius=0.6, tile_size=2.5, workers=1)
    geometric_features_stream(las_path, parallel, radius=0.6, tile_size=2.5, workers=2)

    a = read_parquet(serial)
    b = read_parquet(parallel)
    order_a = np.argsort(a.attribute("index"))
    order_b = np.argsort(b.attribute("index"))
    np.testing.assert_array_equal(
        a.attribute("planarity")[order_a], b.attribute("planarity")[order_b]
    )


def test_stream_output_carries_crs(tmp_path: Path) -> None:
    las_path = tmp_path / "in.las"
    _write_las(las_path, _scene(500))
    out_path = tmp_path / "out.parquet"
    geometric_features_stream(las_path, out_path, radius=0.6, tile_size=2.5)
    crs = read_parquet(out_path).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992


def test_stream_without_crs_raises(tmp_path: Path) -> None:
    las_path = tmp_path / "nocrs.las"
    _write_las(las_path, _scene(300), with_crs=False)
    with pytest.raises(ValueError, match="no coordinate reference"):
        geometric_features_stream(
            las_path, tmp_path / "o.parquet", radius=0.6, tile_size=2.5
        )


def test_stream_tile_size_below_radius_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tile_size must be at least radius"):
        geometric_features_stream(
            tmp_path / "x.las", tmp_path / "o.parquet", radius=1.0, tile_size=0.5
        )


def test_stream_non_positive_radius_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="radius must be"):
        geometric_features_stream(
            tmp_path / "x.las", tmp_path / "o.parquet", radius=0.0, tile_size=2.5
        )


def test_stream_carries_source_attributes(tmp_path: Path) -> None:
    xyz = _scene(800)
    rng = np.random.default_rng(0)
    intensity = rng.integers(0, 1000, len(xyz)).astype(np.uint16)
    classification = rng.integers(0, 10, len(xyz)).astype(np.uint8)
    las_path = tmp_path / "attr.las"
    _write_las_with_attributes(las_path, xyz, intensity, classification)
    out_path = tmp_path / "out.parquet"

    geometric_features_stream(las_path, out_path, radius=0.6, tile_size=2.5)
    streamed = read_parquet(out_path)
    order = np.argsort(streamed.attribute("index"))

    # Attributes are carried through unchanged, in source order, with dtypes.
    np.testing.assert_array_equal(streamed.attribute("intensity")[order], intensity)
    np.testing.assert_array_equal(
        streamed.attribute("classification")[order], classification
    )
    assert streamed.attribute("intensity").dtype == np.uint16
    assert streamed.attribute("classification").dtype == np.uint8


def test_stream_parallel_carries_attributes(tmp_path: Path) -> None:
    xyz = _scene(800)
    rng = np.random.default_rng(1)
    intensity = rng.integers(0, 1000, len(xyz)).astype(np.uint16)
    classification = rng.integers(0, 10, len(xyz)).astype(np.uint8)
    las_path = tmp_path / "attr.las"
    _write_las_with_attributes(las_path, xyz, intensity, classification)
    out_path = tmp_path / "out.parquet"

    geometric_features_stream(las_path, out_path, radius=0.6, tile_size=2.5, workers=2)
    streamed = read_parquet(out_path)
    order = np.argsort(streamed.attribute("index"))
    np.testing.assert_array_equal(streamed.attribute("intensity")[order], intensity)


def test_stream_attribute_subset_selects_requested(tmp_path: Path) -> None:
    xyz = _scene(500)
    rng = np.random.default_rng(2)
    intensity = rng.integers(0, 1000, len(xyz)).astype(np.uint16)
    classification = rng.integers(0, 10, len(xyz)).astype(np.uint8)
    las_path = tmp_path / "attr.las"
    _write_las_with_attributes(las_path, xyz, intensity, classification)
    out_path = tmp_path / "out.parquet"

    geometric_features_stream(
        las_path, out_path, radius=0.6, tile_size=2.5, attributes=["intensity"]
    )
    names = read_parquet(out_path).attribute_names
    assert "intensity" in names
    assert "classification" not in names


def test_stream_unknown_attribute_raises(tmp_path: Path) -> None:
    las_path = tmp_path / "attr.las"
    _write_las_with_attributes(
        las_path,
        _scene(200),
        np.zeros(200, dtype=np.uint16),
        np.zeros(200, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="are not dimensions of the source"):
        geometric_features_stream(
            las_path,
            tmp_path / "o.parquet",
            radius=0.6,
            tile_size=2.5,
            attributes=["not_a_real_dimension"],
        )
