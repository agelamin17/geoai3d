"""Tests for RANSAC primitive fitting."""

import numpy as np
import pytest

from geoai3d import PointCloud, fit_plane


def _tilted_plane_cloud() -> tuple[PointCloud, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0.0, 10.0, n)
    y = rng.uniform(0.0, 10.0, n)
    z = 0.5 * x + 0.2 * y + 1.0 + rng.normal(0.0, 0.01, n)
    inliers = np.column_stack([x, y, z])
    outliers = rng.uniform(0.0, 10.0, (400, 3))
    xyz = np.vstack([inliers, outliers])
    truth = np.concatenate([np.ones(n, dtype=bool), np.zeros(400, dtype=bool)])
    return PointCloud(xyz, crs=28992), truth


def test_recovers_tilted_plane_normal_and_inliers() -> None:
    cloud, truth = _tilted_plane_cloud()
    plane = fit_plane(cloud, distance_threshold=0.05)
    true_normal = np.array([-0.5, -0.2, 1.0])
    true_normal /= np.linalg.norm(true_normal)
    assert abs(np.array(plane.normal) @ true_normal) > 0.999
    recall = np.count_nonzero(plane.inliers & truth) / np.count_nonzero(truth)
    assert recall > 0.98
    assert plane.num_inliers == int(np.count_nonzero(plane.inliers))


def test_recovers_horizontal_plane() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0.0, 10.0, (500, 2))
    xyz = np.column_stack([xy, np.zeros(len(xy))])
    plane = fit_plane(PointCloud(xyz, crs=28992), distance_threshold=0.01)
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-6


def test_signed_distance_near_zero_on_plane() -> None:
    cloud, _ = _tilted_plane_cloud()
    plane = fit_plane(cloud, distance_threshold=0.05)
    distances = plane.signed_distance(cloud.xyz[plane.inliers])
    assert np.all(np.abs(distances) < 0.05)


def test_fit_is_deterministic_with_seed() -> None:
    cloud, _ = _tilted_plane_cloud()
    first = fit_plane(cloud, distance_threshold=0.05, seed=7)
    second = fit_plane(cloud, distance_threshold=0.05, seed=7)
    assert first.normal == second.normal
    assert first.offset == second.offset


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least three points"):
        fit_plane(PointCloud(np.zeros((2, 3)), crs=28992))


def test_non_positive_threshold_raises() -> None:
    cloud, _ = _tilted_plane_cloud()
    with pytest.raises(ValueError, match="distance_threshold must be"):
        fit_plane(cloud, distance_threshold=0.0)


def test_non_positive_iterations_raises() -> None:
    cloud, _ = _tilted_plane_cloud()
    with pytest.raises(ValueError, match="max_iterations must be"):
        fit_plane(cloud, max_iterations=0)
