"""Spatial indexing for point clouds.

A KD-tree (short for k-dimensional tree) is a lookup structure that makes
nearest-neighbour queries fast: given a point, it finds the closest points
without comparing against every other point in the cloud. GEOAI_3D builds one
on demand for feature computation, normal estimation, and subsampling.

The base implementation uses SciPy's ``cKDTree``, which needs no compiler.
Power users can call :func:`build_kdtree` and work with the returned tree
directly (radius queries, batched k-nearest queries, and so on).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud


def build_kdtree(cloud: PointCloud) -> cKDTree:
    """Build a KD-tree over a cloud's coordinates.

    Args:
        cloud: The cloud to index.

    Returns:
        A ``scipy.spatial.cKDTree`` over the ``(n_points, 3)`` coordinates.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, build_kdtree
        >>> cloud = PointCloud(np.arange(300.0).reshape(100, 3), crs=28992)
        >>> tree = build_kdtree(cloud)
        >>> distances, indices = tree.query([0.0, 0.0, 0.0], k=3)
        >>> indices.shape
        (3,)
    """
    return cKDTree(cloud.xyz)
