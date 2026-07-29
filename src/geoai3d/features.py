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

Everything here is plain NumPy over a SciPy KD-tree: CPU-only, no compiler.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from geoai3d.core._derive import attach_attributes
from geoai3d.core.pointcloud import PointCloud

_FEATURE_NAMES = (
    "linearity",
    "planarity",
    "sphericity",
    "anisotropy",
    "omnivariance",
    "eigenentropy",
    "surface_variation",
    "verticality",
    "sum_eigenvalues",
)
_DEFAULT_K = 20


def _xlogx(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``x * log(x)``, defined as 0 where ``x <= 0``."""
    return np.where(values > 0, values * np.log(values), 0.0)


def _eigen_from_neighbors(
    neighbors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return descending eigenvalues and normals for each ``(n, k, 3)`` block."""
    k = neighbors.shape[1]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / k
    values, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(values[:, ::-1], 0.0, None)
    normals = vectors[:, :, 0]
    return eigenvalues, normals


def _features_from_eigen(
    eigenvalues: NDArray[np.float64], normals: NDArray[np.float64]
) -> dict[str, NDArray[Any]]:
    """Compute the geometric descriptors from eigenvalues and normals."""
    total = eigenvalues.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(total > 0, eigenvalues / total, 0.0)
        e1, e2, e3 = normalised[:, 0], normalised[:, 1], normalised[:, 2]
        safe_e1 = np.where(e1 > 0, e1, np.nan)
        linearity = (e1 - e2) / safe_e1
        planarity = (e2 - e3) / safe_e1
        sphericity = e3 / safe_e1
        anisotropy = (e1 - e3) / safe_e1
        omnivariance = np.cbrt(e1 * e2 * e3)
        eigenentropy = -(_xlogx(e1) + _xlogx(e2) + _xlogx(e3))
    return {
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "anisotropy": anisotropy,
        "omnivariance": omnivariance,
        "eigenentropy": eigenentropy,
        "surface_variation": e3,
        "verticality": 1.0 - np.abs(normals[:, 2]),
        "sum_eigenvalues": eigenvalues.sum(axis=1),
    }


def _knn_eigen(
    tree: cKDTree, xyz: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and normals from the ``k`` nearest neighbours of each point."""
    _, neighbor_indices = tree.query(xyz, k=k, workers=-1)
    return _eigen_from_neighbors(xyz[neighbor_indices])


def _radius_eigen(
    tree: cKDTree, xyz: NDArray[np.float64], radius: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and normals from a fixed-radius neighbourhood of each point.

    Points with fewer than three neighbours within ``radius`` get NaN features.
    """
    n_points = len(xyz)
    eigenvalues = np.full((n_points, 3), np.nan)
    normals = np.full((n_points, 3), np.nan)
    neighbor_lists = tree.query_ball_point(xyz, r=radius, workers=-1)
    for i, neighbor_ids in enumerate(neighbor_lists):
        if len(neighbor_ids) < 3:
            continue
        points = xyz[neighbor_ids]
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / len(neighbor_ids)
        values, vectors = np.linalg.eigh(covariance)
        eigenvalues[i] = np.clip(values[::-1], 0.0, None)
        normals[i] = vectors[:, 0]
    return eigenvalues, normals


def geometric_features(
    cloud: PointCloud,
    *,
    k: int | None = None,
    radius: float | None = None,
) -> PointCloud:
    """Add eigenvalue-based geometric feature columns to a cloud.

    Provide exactly one of ``k`` (a fixed number of neighbours) or ``radius``
    (a fixed spatial scale). If neither is given, ``k=20`` is used.

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
        eigenvalues, normals = _radius_eigen(tree, xyz, radius)
    else:
        neighbors = _DEFAULT_K if k is None else k
        if neighbors < 3:
            msg = "k must be at least 3 to fit a local neighbourhood."
            raise ValueError(msg)
        eigenvalues, normals = _knn_eigen(tree, xyz, min(neighbors, len(cloud)))

    features = _features_from_eigen(eigenvalues, normals)
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
        eigenvalues, normals = _knn_eigen(tree, xyz, min(k, n_points))
        scale_features = _features_from_eigen(eigenvalues, normals)
        per_scale.append(scale_features)
        entropies[:, column] = scale_features["eigenentropy"]

    filled = np.where(np.isnan(entropies), np.inf, entropies)
    best = np.argmin(filled, axis=1)

    chosen: dict[str, NDArray[Any]] = {
        name: np.empty(n_points) for name in _FEATURE_NAMES
    }
    for column in range(len(candidates)):
        mask = best == column
        for name in _FEATURE_NAMES:
            chosen[name][mask] = per_scale[column][name][mask]
    chosen["optimal_k"] = np.array([candidates[b] for b in best], dtype=np.float64)

    return attach_attributes(
        cloud, chosen, "multiscale_features", {"ks": list(candidates)}
    )
