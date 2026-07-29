"""Tests for the spatial index."""

import numpy as np

from geoai3d import PointCloud, build_kdtree


def test_build_kdtree_queries_nearest() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    tree = build_kdtree(PointCloud(xyz, crs=28992))
    distances, indices = tree.query([0.0, 0.0, 0.0], k=2)
    assert list(indices) == [0, 1]
    assert distances[0] == 0.0


def test_build_kdtree_size_matches_cloud() -> None:
    cloud = PointCloud(np.random.default_rng(0).random((50, 3)), crs=28992)
    assert build_kdtree(cloud).n == 50
