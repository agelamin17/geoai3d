"""Shared neighbourhood math for geometric features.

Private module. Both the in-memory feature functions and the out-of-core
(tiled) engine call the exact same per-point computation here, so that a tiled
result is bit-for-bit identical to the whole-cloud result.

The subtlety that makes bit-identity possible: floating-point addition is not
associative, so summing a point's neighbours in a different order can change the
last bit. Every point's neighbours are therefore ordered by their *global* point
index before the covariance is summed. Because the halo guarantees the tiled and
whole-cloud paths see the same set of neighbours, ordering them the same way
makes the two paths produce identical bits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = (
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


def _xlogx(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``x * log(x)``, defined as 0 where ``x <= 0``."""
    return np.where(values > 0, values * np.log(values), 0.0)


def features_from_eigen(
    eigenvalues: NDArray[np.float64], normals: NDArray[np.float64]
) -> dict[str, NDArray[Any]]:
    """Compute the geometric descriptors from eigenvalues and normals.

    Args:
        eigenvalues: Per-point eigenvalues in descending order, shape
            ``(n_points, 3)``.
        normals: Per-point surface normals, shape ``(n_points, 3)``.

    Returns:
        A mapping of feature name to a per-point array.
    """
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


def knn_eigen(
    neighbors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Descending eigenvalues and normals for each ``(n, k, 3)`` neighbour block."""
    k = neighbors.shape[1]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / k
    values, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(values[:, ::-1], 0.0, None)
    normals = vectors[:, :, 0]
    return eigenvalues, normals


def radius_eigen(
    pool_xyz: NDArray[np.float64],
    pool_global: NDArray[np.intp],
    tree: Any,
    query_positions: NDArray[np.intp],
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and normals from a fixed-radius neighbourhood.

    Neighbours are ordered by their global index (``pool_global``) before the
    covariance is summed, which is what makes the whole-cloud and tiled paths
    bit-for-bit identical. Query points with fewer than three neighbours within
    ``radius`` get NaN.

    Args:
        pool_xyz: Coordinates of the neighbour pool, shape ``(m, 3)``.
        pool_global: Global point index of each pool point, shape ``(m,)``.
        tree: A ``scipy.spatial.cKDTree`` built over ``pool_xyz``.
        query_positions: Positions within the pool of the points to describe.
        radius: Neighbourhood radius.

    Returns:
        Descending eigenvalues and normals for the query points.
    """
    query_points = pool_xyz[query_positions]
    neighbor_lists = tree.query_ball_point(query_points, r=radius, workers=-1)
    count = len(query_positions)
    eigenvalues = np.full((count, 3), np.nan)
    normals = np.full((count, 3), np.nan)
    for output_index, local_neighbors in enumerate(neighbor_lists):
        if len(local_neighbors) < 3:
            continue
        local = np.asarray(local_neighbors, dtype=np.intp)
        order = np.argsort(pool_global[local], kind="stable")
        points = pool_xyz[local[order]]
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / len(points)
        values, vectors = np.linalg.eigh(covariance)
        eigenvalues[output_index] = np.clip(values[::-1], 0.0, None)
        normals[output_index] = vectors[:, 0]
    return eigenvalues, normals
