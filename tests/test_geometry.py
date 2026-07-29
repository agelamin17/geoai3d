"""Tests for normal estimation and outlier removal."""

import numpy as np
import pytest

from geoai3d import PointCloud, estimate_normals, remove_statistical_outliers


def _flat_plane(n: int = 300) -> PointCloud:
    rng = np.random.default_rng(0)
    xy = rng.random((n, 2))
    xyz = np.column_stack([xy, np.zeros(n)])
    return PointCloud(xyz, crs=28992)


def test_normals_of_flat_plane_point_up() -> None:
    cloud = estimate_normals(_flat_plane(), k=8)
    # A horizontal plane has a vertical normal; orient_upward makes it +z.
    np.testing.assert_allclose(cloud.attribute("nz"), 1.0, atol=1e-6)
    np.testing.assert_allclose(cloud.attribute("nx"), 0.0, atol=1e-6)


def test_normals_add_three_columns() -> None:
    cloud = estimate_normals(_flat_plane(), k=8)
    for name in ("nx", "ny", "nz"):
        assert name in cloud.attribute_names


def test_estimate_normals_small_k_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        estimate_normals(_flat_plane(), k=2)


def test_outlier_removal_drops_far_point() -> None:
    rng = np.random.default_rng(0)
    pts = rng.random((500, 3))
    pts[0] = [100.0, 100.0, 100.0]
    cloud = PointCloud(pts, crs=28992)
    cleaned = remove_statistical_outliers(cloud, k=16, std_ratio=2.0)
    assert len(cleaned) < 500
    assert float(cleaned.xyz.max()) < 100.0


def test_outlier_removal_records_provenance() -> None:
    cleaned = remove_statistical_outliers(_flat_plane(), k=8)
    assert cleaned.provenance is not None
    assert cleaned.provenance.steps[-1].description == "remove_statistical_outliers"
