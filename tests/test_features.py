"""Tests for eigenvalue-based geometric features."""

import numpy as np
import pytest

from geoai3d import PointCloud, geometric_features, multiscale_features


def _grid_plane() -> PointCloud:
    grid = np.linspace(-1.0, 1.0, 21)
    gx, gy = np.meshgrid(grid, grid)
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    return PointCloud(xyz, crs=28992)


def _vertical_grid_plane() -> PointCloud:
    grid = np.linspace(-1.0, 1.0, 21)
    gx, gz = np.meshgrid(grid, grid)
    xyz = np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()])
    return PointCloud(xyz, crs=28992)


def _line() -> PointCloud:
    t = np.linspace(-5.0, 5.0, 200)
    xyz = np.column_stack([t, np.zeros(200), np.zeros(200)])
    return PointCloud(xyz, crs=28992)


def test_all_feature_columns_are_added() -> None:
    cloud = geometric_features(_grid_plane(), k=13)
    for name in (
        "linearity",
        "planarity",
        "sphericity",
        "anisotropy",
        "omnivariance",
        "eigenentropy",
        "surface_variation",
        "verticality",
        "sum_eigenvalues",
    ):
        assert name in cloud.attribute_names


def test_plane_is_planar_and_flat() -> None:
    cloud = geometric_features(_grid_plane(), k=13)
    assert float(np.nanmedian(cloud.attribute("planarity"))) > 0.9
    assert float(np.nanmedian(cloud.attribute("sphericity"))) < 0.1
    # A horizontal plane has a vertical normal, so verticality is near zero.
    assert float(np.nanmedian(cloud.attribute("verticality"))) < 0.1


def test_vertical_plane_has_high_verticality() -> None:
    cloud = geometric_features(_vertical_grid_plane(), k=13)
    assert float(np.nanmedian(cloud.attribute("verticality"))) > 0.9


def test_line_is_linear() -> None:
    cloud = geometric_features(_line(), k=7)
    assert float(np.nanmedian(cloud.attribute("linearity"))) > 0.9


def test_radius_neighbourhood_works() -> None:
    cloud = geometric_features(_grid_plane(), radius=0.25)
    assert float(np.nanmedian(cloud.attribute("planarity"))) > 0.9


def test_both_k_and_radius_raises() -> None:
    with pytest.raises(ValueError, match="not both"):
        geometric_features(_grid_plane(), k=10, radius=0.5)


def test_small_k_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        geometric_features(_grid_plane(), k=2)


def test_non_positive_radius_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        geometric_features(_grid_plane(), radius=0.0)


def test_multiscale_adds_optimal_scale() -> None:
    cloud = multiscale_features(_grid_plane(), ks=(8, 16, 24))
    assert "optimal_k" in cloud.attribute_names
    assert "planarity" in cloud.attribute_names
    assert set(np.unique(cloud.attribute("optimal_k")).tolist()).issubset({8, 16, 24})


def test_multiscale_small_k_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        multiscale_features(_grid_plane(), ks=(2, 10))
