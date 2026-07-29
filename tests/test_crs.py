"""Tests for horizontal and 3D reprojection."""

import numpy as np
import pyproj
import pytest

from geoai3d import PointCloud, reproject, reproject_3d


def test_reproject_changes_crs_and_coordinates() -> None:
    cloud = PointCloud(np.array([[155000.0, 463000.0, 10.0]]), crs=28992)
    out = reproject(cloud, 4326)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4326
    # RD New (155000, 463000) is roughly 5.4 deg E, 52.15 deg N.
    assert 3.0 < out.xyz[0, 0] < 8.0
    assert 50.0 < out.xyz[0, 1] < 54.0


def test_reproject_preserves_height() -> None:
    cloud = PointCloud(np.array([[155000.0, 463000.0, 42.5]]), crs=28992)
    out = reproject(cloud, 4326)
    assert out.xyz[0, 2] == 42.5


def test_reproject_preserves_attributes() -> None:
    cloud = PointCloud(
        np.array([[155000.0, 463000.0, 0.0]]),
        attributes={"intensity": np.array([7], dtype=np.uint16)},
        crs=28992,
    )
    out = reproject(cloud, 4326)
    np.testing.assert_array_equal(
        out.attribute("intensity"), np.array([7], dtype=np.uint16)
    )


def test_reproject_appends_provenance() -> None:
    cloud = PointCloud(np.array([[155000.0, 463000.0, 0.0]]), crs=28992)
    provenance = reproject(cloud, 4326).provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "reproject (horizontal)"


def test_reproject_without_crs_raises() -> None:
    with pytest.raises(ValueError, match="no CRS"):
        reproject(PointCloud(np.zeros((2, 3))), 4326)


def test_reproject_3d_transforms_all_axes() -> None:
    # WGS84 3D geographic -> geocentric ECEF is a pure computation (no grid),
    # so it exercises the 3D path offline.
    cloud = PointCloud(np.array([[5.0, 52.0, 43.0]]), crs=4979)
    out = reproject_3d(cloud, 4978)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4978
    assert np.all(np.isfinite(out.xyz))
    # Geocentric coordinates are on the order of millions of metres.
    assert abs(out.xyz[0, 0]) > 1e6


def test_reproject_3d_requires_vertical_axis() -> None:
    cloud = PointCloud(np.array([[155000.0, 463000.0, 0.0]]), crs=28992)
    with pytest.raises(ValueError, match="no vertical axis"):
        reproject_3d(cloud, 4979)


def test_reproject_3d_without_crs_raises() -> None:
    with pytest.raises(ValueError, match="no CRS"):
        reproject_3d(PointCloud(np.zeros((2, 3))), 4979)
