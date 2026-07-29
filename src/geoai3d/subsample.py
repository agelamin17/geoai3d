"""Subsampling point clouds: random, voxel, and farthest-point.

All three methods return a subset of the original points, so attributes and the
CRS are carried through unchanged (no coordinates are averaged or moved). The
choice of method trades speed against how evenly the result covers the cloud:
random is fastest, voxel gives a roughly uniform spatial density, and
farthest-point gives the most even coverage at higher cost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geoai3d.core._derive import select
from geoai3d.core.pointcloud import PointCloud

_METHODS = ("random", "voxel", "farthest_point")


def _random_indices(
    n_points: int, count: int | None, seed: int | None
) -> NDArray[np.intp]:
    """Return sorted indices of ``count`` points chosen uniformly at random."""
    if count is None:
        msg = "method='random' requires count= (the number of points to keep)."
        raise ValueError(msg)
    if count > n_points:
        msg = f"count={count} exceeds the {n_points} points in the cloud."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_points, size=count, replace=False)
    indices.sort()
    return indices


def _voxel_indices(
    xyz: NDArray[np.float64], voxel_size: float | None
) -> NDArray[np.intp]:
    """Return indices of the first point falling in each occupied voxel."""
    if voxel_size is None or voxel_size <= 0:
        msg = "method='voxel' requires voxel_size= to be a positive number."
        raise ValueError(msg)
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel_size).astype(np.int64)
    _, first_occurrence = np.unique(keys, axis=0, return_index=True)
    return np.sort(first_occurrence)


def _farthest_point_indices(
    xyz: NDArray[np.float64], count: int | None, seed: int | None
) -> NDArray[np.intp]:
    """Return indices of ``count`` points chosen by farthest-point sampling."""
    if count is None:
        msg = "method='farthest_point' requires count=."
        raise ValueError(msg)
    n_points = len(xyz)
    count = min(count, n_points)
    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.intp)
    selected[0] = int(rng.integers(n_points))
    distance = np.full(n_points, np.inf)
    for i in range(1, count):
        squared = np.sum((xyz - xyz[selected[i - 1]]) ** 2, axis=1)
        distance = np.minimum(distance, squared)
        selected[i] = int(np.argmax(distance))
    return np.sort(selected)


def subsample(
    cloud: PointCloud,
    *,
    method: str = "voxel",
    voxel_size: float | None = None,
    count: int | None = None,
    seed: int | None = None,
) -> PointCloud:
    """Return a subset of a cloud's points.

    Args:
        cloud: The cloud to subsample.
        method: One of ``"random"``, ``"voxel"``, or ``"farthest_point"``.
        voxel_size: Edge length of the voxel grid, in the cloud's CRS units.
            Required for ``method="voxel"``.
        count: Number of points to keep. Required for ``"random"`` and
            ``"farthest_point"``.
        seed: Seed for the random generator, for reproducible results.

    Returns:
        A new :class:`PointCloud` with the kept points, their attributes, the
        CRS, and a provenance step.

    Raises:
        ValueError: If ``method`` is unknown or a required argument is missing.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, subsample
        >>> cloud = PointCloud(np.random.default_rng(0).random((1000, 3)), crs=28992)
        >>> len(subsample(cloud, method="random", count=100, seed=0))
        100
    """
    if method not in _METHODS:
        msg = f"Unknown method {method!r}; choose one of {list(_METHODS)}."
        raise ValueError(msg)
    if method == "random":
        indices = _random_indices(len(cloud), count, seed)
    elif method == "voxel":
        indices = _voxel_indices(cloud.xyz, voxel_size)
    else:
        indices = _farthest_point_indices(cloud.xyz, count, seed)

    return select(
        cloud,
        indices,
        "subsample",
        {"method": method, "voxel_size": voxel_size, "count": count, "seed": seed},
    )
