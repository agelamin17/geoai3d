"""Out-of-core, tile-by-tile geometric feature computation.

A real airborne survey has far more points than fit in memory, so features must
be computed one spatial tile at a time. The catch is that a point near a tile
edge has neighbours on the next tile, so processing a tile in isolation gives it
the wrong features. :func:`geometric_features_tiled` avoids that by reading each
tile together with a **halo** -- a border of neighbouring points one radius wide
-- computing features on the tile-plus-halo, then keeping only the core-tile
points. Because the halo is exactly one radius wide, every core point sees all
of its true neighbours, and its features are bit-for-bit identical to the
whole-cloud result. This is the seam contract, and it holds only for a fixed
radius, because the halo width has to be known in advance.

This module tiles a cloud that is already in memory, which validates the tiling
and halo logic; feeding tiles from disk so the whole cloud never has to be
resident is a follow-up increment that reuses this exact per-tile computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from geoai3d.core._derive import attach_attributes
from geoai3d.core._neighborhood import FEATURE_NAMES, features_from_eigen, radius_eigen
from geoai3d.core.pointcloud import PointCloud


def estimate_radius(
    cloud: PointCloud,
    *,
    target_neighbors: int = 20,
    sample_size: int = 10000,
    seed: int = 0,
) -> float:
    """Estimate a feature radius that captures about ``target_neighbors`` points.

    Measures, over a random sample of points, the distance to the
    ``target_neighbors``-th nearest neighbour and returns the median. That is a
    radius which, on average, encloses roughly that many neighbours given the
    cloud's own density -- so the caller does not have to guess a number.

    Args:
        cloud: The cloud to measure.
        target_neighbors: Desired average neighbour count within the radius.
        sample_size: Number of points to sample for the estimate.
        seed: Seed for the random sample.

    Returns:
        The estimated radius in the cloud's CRS units.

    Raises:
        ValueError: If the cloud has fewer than two points, or
            ``target_neighbors`` is below 1.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, estimate_radius
        >>> cloud = PointCloud(np.random.default_rng(0).random((2000, 3)), crs=28992)
        >>> radius = estimate_radius(cloud, target_neighbors=15)
        >>> radius > 0
        True
    """
    if target_neighbors < 1:
        msg = "target_neighbors must be at least 1."
        raise ValueError(msg)
    n_points = len(cloud)
    if n_points < 2:
        msg = "estimate_radius needs at least two points."
        raise ValueError(msg)
    xyz = cloud.xyz
    tree = cKDTree(xyz)
    k = min(target_neighbors + 1, n_points)
    rng = np.random.default_rng(seed)
    if n_points > sample_size:
        sample = xyz[rng.choice(n_points, size=sample_size, replace=False)]
    else:
        sample = xyz
    distances, _ = tree.query(sample, k=k, workers=-1)
    return float(np.median(distances[:, -1]))


def geometric_features_tiled(
    cloud: PointCloud,
    *,
    radius: float,
    tile_size: float,
) -> PointCloud:
    """Compute fixed-radius geometric features tile by tile with a halo.

    The result is identical to :func:`geometric_features` with the same
    ``radius`` on the whole cloud -- bit for bit -- but each tile is processed
    with a bounded working set (the tile plus a one-radius halo), so memory
    stays low for clouds much larger than RAM.

    Args:
        cloud: The cloud to describe.
        radius: Neighbourhood radius. Points with fewer than three neighbours
            within it get NaN features. Also the halo width.
        tile_size: Edge length of the square (x, y) tiles, in CRS units.

    Returns:
        A new :class:`PointCloud` with the geometric feature columns added.

    Raises:
        ValueError: If ``radius`` or ``tile_size`` is not positive.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, geometric_features_tiled
        >>> pts = np.random.default_rng(0).uniform(0, 10, (2000, 3))
        >>> cloud = geometric_features_tiled(PointCloud(pts, crs=28992),
        ...                                  radius=0.6, tile_size=2.5)
        >>> "planarity" in cloud.attribute_names
        True
    """
    if radius <= 0:
        msg = "radius must be a positive number."
        raise ValueError(msg)
    if tile_size <= 0:
        msg = "tile_size must be a positive number."
        raise ValueError(msg)

    xyz = cloud.xyz
    n_points = len(cloud)
    global_index = np.arange(n_points, dtype=np.intp)
    origin = xyz[:, :2].min(axis=0)
    tile_ij = np.floor((xyz[:, :2] - origin) / tile_size).astype(np.int64)

    results: dict[str, NDArray[Any]] = {
        name: np.full(n_points, np.nan) for name in FEATURE_NAMES
    }
    for tile_i, tile_j in np.unique(tile_ij, axis=0):
        core_mask = (tile_ij[:, 0] == tile_i) & (tile_ij[:, 1] == tile_j)
        x_min = origin[0] + tile_i * tile_size
        x_max = x_min + tile_size
        y_min = origin[1] + tile_j * tile_size
        y_max = y_min + tile_size
        pool_mask = (
            (xyz[:, 0] >= x_min - radius)
            & (xyz[:, 0] < x_max + radius)
            & (xyz[:, 1] >= y_min - radius)
            & (xyz[:, 1] < y_max + radius)
        )
        pool_positions = np.flatnonzero(pool_mask)
        pool_xyz = xyz[pool_positions]
        pool_global = global_index[pool_positions]
        tree = cKDTree(pool_xyz)

        core_global = np.flatnonzero(core_mask)
        core_in_pool = np.searchsorted(pool_positions, core_global)
        eigenvalues, normals = radius_eigen(
            pool_xyz, pool_global, tree, core_in_pool, radius
        )
        tile_features = features_from_eigen(eigenvalues, normals)
        for name in FEATURE_NAMES:
            results[name][core_global] = tile_features[name]

    return attach_attributes(
        cloud,
        results,
        "geometric_features_tiled",
        {"radius": radius, "tile_size": tile_size},
    )
