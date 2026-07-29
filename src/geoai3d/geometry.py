"""Local geometry: surface normals and statistical outlier removal.

Both operations fit a small plane to each point's nearest neighbours. Normal
estimation takes the direction of least spread (the surface normal); outlier
removal flags points whose neighbours are unusually far away. Both use a
SciPy KD-tree and plain NumPy, so they need no GPU and no compiler.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from geoai3d.core._derive import attach_attributes, select
from geoai3d.core.pointcloud import PointCloud


def estimate_normals(
    cloud: PointCloud,
    *,
    k: int = 16,
    orient_upward: bool = True,
) -> PointCloud:
    """Estimate a surface normal at each point from its neighbours.

    The normal is the direction of least variance of the ``k`` nearest
    neighbours (the smallest principal component of the local covariance).

    Args:
        cloud: The cloud to add normals to.
        k: Number of nearest neighbours used to fit each local plane. Must be
            at least 3.
        orient_upward: If true, flip every normal to point into the upper
            hemisphere (positive z), giving a consistent sign.

    Returns:
        A new :class:`PointCloud` with ``nx``, ``ny``, ``nz`` attributes added.

    Raises:
        ValueError: If ``k`` is less than 3.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, estimate_normals
        >>> flat = np.random.default_rng(0).random((200, 3))
        >>> flat[:, 2] = 0.0
        >>> cloud = estimate_normals(PointCloud(flat, crs=28992), k=8)
        >>> bool(np.allclose(np.abs(cloud.attribute("nz")), 1.0))
        True
    """
    if k < 3:
        msg = "k must be at least 3 to fit a local plane."
        raise ValueError(msg)
    xyz = cloud.xyz
    k_effective = min(k, len(cloud))
    tree = cKDTree(xyz)
    _, neighbor_indices = tree.query(xyz, k=k_effective, workers=-1)
    neighbors = xyz[neighbor_indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / k_effective
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    if orient_upward:
        downward = normals[:, 2] < 0
        normals[downward] = -normals[downward]
    return attach_attributes(
        cloud,
        {
            "nx": np.ascontiguousarray(normals[:, 0]),
            "ny": np.ascontiguousarray(normals[:, 1]),
            "nz": np.ascontiguousarray(normals[:, 2]),
        },
        "estimate_normals",
        {"k": k, "orient_upward": orient_upward},
    )


def remove_statistical_outliers(
    cloud: PointCloud,
    *,
    k: int = 16,
    std_ratio: float = 2.0,
) -> PointCloud:
    """Remove points whose neighbours are unusually far away.

    For each point, the mean distance to its ``k`` nearest neighbours is
    computed. Points whose mean distance exceeds the global mean by more than
    ``std_ratio`` standard deviations are dropped.

    Args:
        cloud: The cloud to filter.
        k: Number of nearest neighbours per point.
        std_ratio: Multiplier of the global standard deviation above which a
            point is treated as an outlier. Smaller values remove more points.

    Returns:
        A new :class:`PointCloud` with the outliers removed.

    Raises:
        ValueError: If ``k`` is less than 1.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, remove_statistical_outliers
        >>> pts = np.random.default_rng(0).random((500, 3))
        >>> pts[0] = [100.0, 100.0, 100.0]  # a clear outlier
        >>> cleaned = remove_statistical_outliers(PointCloud(pts, crs=28992))
        >>> len(cleaned) < 500
        True
    """
    if k < 1:
        msg = "k must be at least 1."
        raise ValueError(msg)
    xyz = cloud.xyz
    k_effective = min(k, len(cloud) - 1)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k_effective + 1, workers=-1)
    # Column 0 is the point itself (distance 0); average the rest.
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    keep = np.flatnonzero(mean_distance <= threshold)
    return select(
        cloud, keep, "remove_statistical_outliers", {"k": k, "std_ratio": std_ratio}
    )
