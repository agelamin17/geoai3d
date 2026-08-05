"""Spatially-blocked cross-validation.

A random train/test split on a point cloud leaks: neighbouring points are
strongly correlated, so a random test point almost always sits beside a training
point, and reported accuracy is inflated -- sometimes greatly. The fix, standard
in the 2D geospatial-ML literature and still rare in 3D, is to split by space:
group points into square blocks and assign whole blocks to folds, so a fold's
test points are spatially separated from its training points.

:func:`spatial_block_split` returns scikit-learn-style ``(train, test)`` index
pairs, so it drops into an ordinary cross-validation loop. Pure NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud


def spatial_block_split(
    cloud: PointCloud,
    *,
    block_size: float,
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[NDArray[Any], NDArray[Any]]]:
    """Split a cloud into spatially-blocked cross-validation folds.

    Points are grouped into square ``block_size`` blocks in the horizontal
    plane, the blocks are shuffled and dealt round-robin into ``n_folds`` folds,
    and each fold's test set is the points of its blocks with the rest as the
    training set. Because a whole block goes to one fold, test points never
    neighbour their own training points.

    Args:
        cloud: The cloud to split.
        block_size: Edge length of the square blocks, in the cloud's CRS units.
            Make it several times the feature neighbourhood so blocks are
            genuinely independent.
        n_folds: Number of folds.
        seed: Seed for shuffling blocks into folds.

    Returns:
        A list of ``n_folds`` ``(train_indices, test_indices)`` pairs of integer
        index arrays into the cloud.

    Raises:
        ValueError: If the cloud is empty, ``block_size`` is not positive,
            ``n_folds`` is below 2, or there are fewer blocks than folds.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, spatial_block_split
        >>> pts = np.random.default_rng(0).uniform(0.0, 100.0, (2000, 3))
        >>> folds = spatial_block_split(
        ...     PointCloud(pts, crs=28992), block_size=10.0, n_folds=5
        ... )
        >>> len(folds)
        5
    """
    n_points = len(cloud)
    if n_points == 0:
        msg = "Cannot split an empty cloud."
        raise ValueError(msg)
    if block_size <= 0.0:
        msg = "block_size must be a positive number."
        raise ValueError(msg)
    if n_folds < 2:
        msg = "n_folds must be at least 2."
        raise ValueError(msg)

    xyz = cloud.xyz
    block_col = np.floor((xyz[:, 0] - xyz[:, 0].min()) / block_size).astype(np.int64)
    block_row = np.floor((xyz[:, 1] - xyz[:, 1].min()) / block_size).astype(np.int64)
    # A single integer key per block; np.unique gives each point its block index.
    _, block_index = np.unique(
        np.column_stack([block_col, block_row]), axis=0, return_inverse=True
    )
    block_index = np.asarray(block_index).ravel()
    n_blocks = int(block_index.max()) + 1
    if n_blocks < n_folds:
        msg = (
            f"Only {n_blocks} spatial blocks for {n_folds} folds. Use a smaller "
            "block_size or fewer folds."
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    fold_of_block = rng.permutation(n_blocks) % n_folds
    point_fold = fold_of_block[block_index]

    folds: list[tuple[NDArray[Any], NDArray[Any]]] = []
    for fold in range(n_folds):
        test_indices = np.flatnonzero(point_fold == fold)
        train_indices = np.flatnonzero(point_fold != fold)
        folds.append((train_indices, test_indices))
    return folds
