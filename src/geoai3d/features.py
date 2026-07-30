"""Eigenvalue-based geometric feature descriptors.

For each point, the covariance of its local neighbourhood has three eigenvalues
lambda1 >= lambda2 >= lambda3 >= 0 that describe the local shape: one large
eigenvalue means a line, two means a plane, three roughly equal means a
volumetric blob. From the normalised eigenvalues this module derives the
standard descriptors of Weinmann et al. (2015) and Demantke et al. (2011) --
linearity, planarity, sphericity, anisotropy, omnivariance, eigenentropy,
surface variation, and verticality -- plus a multi-scale variant that picks,
per point, the neighbourhood size that minimises eigenentropy (the
dimensionality-based optimal scale of Demantke et al.).

The per-point math lives in ``core._neighborhood`` so the out-of-core tiled
engine reuses it verbatim. Everything here is plain NumPy over a SciPy KD-tree:
CPU-only, no compiler.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from geoai3d.core._derive import attach_attributes
from geoai3d.core._neighborhood import (
    FEATURE_NAMES,
    features_from_eigen,
    knn_eigen,
    radius_eigen,
)
from geoai3d.core.pointcloud import PointCloud

_DEFAULT_K = 20


def geometric_features(
    cloud: PointCloud,
    *,
    k: int | None = None,
    radius: float | None = None,
) -> PointCloud:
    """Add eigenvalue-based geometric feature columns to a cloud.

    Provide exactly one of ``k`` (a fixed number of neighbours) or ``radius``
    (a fixed spatial scale). If neither is given, ``k=20`` is used. Only the
    ``radius`` form carries the bit-identical out-of-core seam guarantee.

    Args:
        cloud: The cloud to describe.
        k: Number of nearest neighbours per point. Must be at least 3.
        radius: Neighbourhood radius in the cloud's CRS units. Points with
            fewer than three neighbours within it get NaN features.

    Returns:
        A new :class:`PointCloud` with the feature columns (``linearity``,
        ``planarity``, ``sphericity``, ``anisotropy``, ``omnivariance``,
        ``eigenentropy``, ``surface_variation``, ``verticality``,
        ``sum_eigenvalues``) added.

    Raises:
        ValueError: If both ``k`` and ``radius`` are given, or ``k`` is below 3,
            or ``radius`` is not positive.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, geometric_features
        >>> grid = np.linspace(-1, 1, 21)
        >>> gx, gy = np.meshgrid(grid, grid)
        >>> flat = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        >>> cloud = geometric_features(PointCloud(flat, crs=28992), k=13)
        >>> bool(np.nanmedian(cloud.attribute("planarity")) > 0.9)
        True
    """
    if k is not None and radius is not None:
        msg = "Give either k= or radius=, not both."
        raise ValueError(msg)
    xyz = cloud.xyz
    tree = cKDTree(xyz)
    if radius is not None:
        if radius <= 0:
            msg = "radius must be a positive number."
            raise ValueError(msg)
        positions = np.arange(len(cloud), dtype=np.intp)
        eigenvalues, normals = radius_eigen(xyz, positions, tree, positions, radius)
    else:
        neighbors = _DEFAULT_K if k is None else k
        if neighbors < 3:
            msg = "k must be at least 3 to fit a local neighbourhood."
            raise ValueError(msg)
        _, neighbor_indices = tree.query(xyz, k=min(neighbors, len(cloud)), workers=-1)
        eigenvalues, normals = knn_eigen(xyz[neighbor_indices])

    features = features_from_eigen(eigenvalues, normals)
    return attach_attributes(
        cloud, features, "geometric_features", {"k": k, "radius": radius}
    )


def multiscale_features(
    cloud: PointCloud,
    *,
    ks: tuple[int, ...] = (10, 20, 30, 40, 50),
) -> PointCloud:
    """Add geometric features computed at each point's optimal neighbourhood.

    Features are computed at several neighbourhood sizes and, per point, the
    size that minimises eigenentropy is chosen (the dimensionality-based scale
    selection of Demantke et al.). The chosen size is stored as ``optimal_k``.

    Args:
        cloud: The cloud to describe.
        ks: Candidate neighbour counts to try. Each must be at least 3.

    Returns:
        A new :class:`PointCloud` with the geometric feature columns at the
        per-point optimal scale, plus an ``optimal_k`` column.

    Raises:
        ValueError: If any candidate in ``ks`` is below 3, or ``ks`` is empty.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, multiscale_features
        >>> pts = np.random.default_rng(0).random((400, 3))
        >>> cloud = multiscale_features(PointCloud(pts, crs=28992), ks=(8, 16, 24))
        >>> "optimal_k" in cloud.attribute_names
        True
    """
    candidates = tuple(sorted({int(value) for value in ks}))
    if not candidates:
        msg = "ks must contain at least one neighbour count."
        raise ValueError(msg)
    if candidates[0] < 3:
        msg = "every candidate in ks must be at least 3."
        raise ValueError(msg)

    xyz = cloud.xyz
    n_points = len(cloud)
    tree = cKDTree(xyz)
    per_scale: list[dict[str, NDArray[Any]]] = []
    entropies = np.empty((n_points, len(candidates)))
    for column, k in enumerate(candidates):
        _, neighbor_indices = tree.query(xyz, k=min(k, n_points), workers=-1)
        scale_features = features_from_eigen(*knn_eigen(xyz[neighbor_indices]))
        per_scale.append(scale_features)
        entropies[:, column] = scale_features["eigenentropy"]

    filled = np.where(np.isnan(entropies), np.inf, entropies)
    best = np.argmin(filled, axis=1)

    chosen: dict[str, NDArray[Any]] = {
        name: np.empty(n_points) for name in FEATURE_NAMES
    }
    for column in range(len(candidates)):
        mask = best == column
        for name in FEATURE_NAMES:
            chosen[name][mask] = per_scale[column][name][mask]
    chosen["optimal_k"] = np.array([candidates[b] for b in best], dtype=np.float64)

    return attach_attributes(
        cloud, chosen, "multiscale_features", {"ks": list(candidates)}
    )
