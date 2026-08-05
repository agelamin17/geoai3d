"""Clustering points into groups by spatial connectivity.

:func:`connected_components` labels points into instances by connectivity: two
points are joined if they are within a distance of each other, and each
connected group becomes one label. It is the simplest instance-extraction
method (B8) -- no density or shape assumptions -- and a useful building block
for segmentation, for example splitting the non-ground points of a scene into
individual objects.

The distance graph is built with a SciPy KD-tree and the components are found
with SciPy's sparse-graph routines, so it scales to large clouds without
forming the full pairwise distance matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _sparse_components
from scipy.spatial import cKDTree

from geoai3d.core._derive import attach_attributes

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud


def connected_components(
    cloud: PointCloud,
    *,
    distance: float,
    min_size: int = 1,
    output_attribute: str = "component",
) -> PointCloud:
    """Label points into connected components by a distance threshold.

    Two points are connected if they lie within ``distance`` of each other, and
    each connected group gets a distinct integer label. Components with fewer
    than ``min_size`` points are labelled ``-1`` (noise); the rest are numbered
    ``0, 1, 2, ...`` in order of first appearance.

    Args:
        cloud: The cloud to label.
        distance: Maximum distance, in CRS units, between two points for them to
            be connected.
        min_size: Minimum number of points for a component to be kept; smaller
            groups are labelled ``-1``.
        output_attribute: Name of the integer label column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the integer component label
        column added, the CRS and provenance carried through.

    Raises:
        ValueError: If the cloud is empty, ``distance`` is not positive, or
            ``min_size`` is below 1.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, connected_components
        >>> a = np.random.default_rng(0).normal(0.0, 0.1, (50, 3))
        >>> b = np.random.default_rng(1).normal(10.0, 0.1, (50, 3))
        >>> cloud = connected_components(
        ...     PointCloud(np.vstack([a, b]), crs=28992), distance=0.5
        ... )
        >>> len(set(cloud.attribute("component").tolist()))
        2
    """
    n_points = len(cloud)
    if n_points == 0:
        msg = "Cannot cluster an empty cloud."
        raise ValueError(msg)
    if distance <= 0.0:
        msg = "distance must be a positive number."
        raise ValueError(msg)
    if min_size < 1:
        msg = "min_size must be at least 1."
        raise ValueError(msg)

    tree = cKDTree(cloud.xyz)
    pairs = tree.query_pairs(distance, output_type="ndarray")
    if pairs.size:
        edge_weights = np.ones(len(pairs))
        graph = coo_matrix(
            (edge_weights, (pairs[:, 0], pairs[:, 1])),
            shape=(n_points, n_points),
        )
        _, raw_labels = _sparse_components(graph, directed=False)
    else:
        raw_labels = np.arange(n_points, dtype=np.int64)

    counts = np.bincount(raw_labels)
    keep = counts >= min_size
    relabel = np.full(len(counts), -1, dtype=np.int64)
    relabel[np.flatnonzero(keep)] = np.arange(int(np.count_nonzero(keep)))
    component = relabel[raw_labels]

    return attach_attributes(
        cloud,
        {output_attribute: component},
        "connected_components",
        {"distance": distance, "min_size": min_size},
    )
