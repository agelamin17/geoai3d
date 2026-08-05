"""Clustering and unsupervised segmentation of point clouds.

Three complementary ways to label points into groups:

* :func:`connected_components` joins points within a distance of each other --
  the simplest instance extraction, no density or shape assumptions (B8).
* :func:`dbscan` adds a density requirement, so sparse points become noise
  rather than joining a cluster (B5).
* :func:`region_growing` grows smooth regions from seeds using surface normals,
  splitting a scene by orientation -- the classical unsupervised geometric
  segmentation (B2).

The distance graphs are built with a SciPy KD-tree so they scale to large
clouds without forming the full pairwise distance matrix.
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

_NORMAL_ATTRIBUTES = ("nx", "ny", "nz")


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


def dbscan(
    cloud: PointCloud,
    *,
    eps: float,
    min_samples: int = 5,
    output_attribute: str = "cluster",
) -> PointCloud:
    """Cluster points by density with DBSCAN.

    DBSCAN groups points that have at least ``min_samples`` neighbours within
    ``eps``, growing clusters through those dense points and leaving sparse
    points unclustered. Unlike :func:`connected_components`, a thin bridge of
    isolated points does not merge two clusters.

    Args:
        cloud: The cloud to cluster.
        eps: Neighbourhood radius, in CRS units.
        min_samples: Minimum neighbours (including the point itself) for a point
            to be a dense core of a cluster.
        output_attribute: Name of the integer label column to add. Cluster
            labels are ``0, 1, 2, ...`` and unclustered points are ``-1``.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the integer cluster label column
        added, the CRS and provenance carried through.

    Raises:
        ValueError: If the cloud is empty, ``eps`` is not positive, or
            ``min_samples`` is below 1.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, dbscan
        >>> a = np.random.default_rng(0).normal(0.0, 0.1, (50, 3))
        >>> b = np.random.default_rng(1).normal(10.0, 0.1, (50, 3))
        >>> out = dbscan(PointCloud(np.vstack([a, b]), crs=28992), eps=0.5)
        >>> int(out.attribute("cluster").max())
        1
    """
    if len(cloud) == 0:
        msg = "Cannot cluster an empty cloud."
        raise ValueError(msg)
    if eps <= 0.0:
        msg = "eps must be a positive number."
        raise ValueError(msg)
    if min_samples < 1:
        msg = "min_samples must be at least 1."
        raise ValueError(msg)

    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cloud.xyz)
    return attach_attributes(
        cloud,
        {output_attribute: labels.astype(np.int64)},
        "dbscan",
        {"eps": eps, "min_samples": min_samples},
    )


def region_growing(
    cloud: PointCloud,
    *,
    k: int = 30,
    smoothness_degrees: float = 15.0,
    curvature_attribute: str = "surface_variation",
    curvature_threshold: float | None = None,
    min_size: int = 10,
    output_attribute: str = "segment",
) -> PointCloud:
    """Segment a cloud into smooth regions by growing from seeds.

    Starting from the smoothest points, a region absorbs a neighbour when the
    angle between their surface normals is below ``smoothness_degrees``; the
    neighbour then continues the growth unless a curvature limit stops it. This
    is the classical normal-based region growing (after Rabbani et al.), which
    separates a scene by surface orientation -- ground from walls from roofs --
    with no training data.

    The cloud must carry surface normals (``nx``, ``ny``, ``nz`` from
    :func:`~geoai3d.estimate_normals`). If it also carries a curvature attribute
    (for example ``surface_variation`` from :func:`~geoai3d.geometric_features`)
    the smoothest points seed first, which gives cleaner regions.

    Args:
        cloud: The cloud to segment. Must carry surface normals.
        k: Number of nearest neighbours considered for growth.
        smoothness_degrees: Maximum normal-to-normal angle, in degrees, for a
            neighbour to join a region.
        curvature_attribute: Attribute used to order seeds (smoothest first) and,
            with ``curvature_threshold``, to limit propagation. Ignored if the
            cloud does not carry it.
        curvature_threshold: If set, only points with curvature below it continue
            growing a region. ``None`` lets every joined point continue.
        min_size: Minimum points for a region to be kept; smaller regions are
            labelled ``-1``.
        output_attribute: Name of the integer segment label column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the integer segment label column
        added (contiguous ids, ``-1`` for unassigned points), the CRS and
        provenance carried through.

    Raises:
        ValueError: If the cloud is empty, lacks the normal attributes, ``k`` or
            ``min_size`` is below 1, or ``smoothness_degrees`` is not in
            ``(0, 180)``.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, estimate_normals, region_growing
        >>> rng = np.random.default_rng(0)
        >>> flat = rng.uniform(0.0, 5.0, (400, 3))
        >>> flat[:, 2] = 0.0
        >>> cloud = estimate_normals(PointCloud(flat, crs=28992), k=12)
        >>> segmented = region_growing(cloud, k=12, min_size=20)
        >>> int(segmented.attribute("segment").max())
        0
    """
    n_points = len(cloud)
    if n_points == 0:
        msg = "Cannot segment an empty cloud."
        raise ValueError(msg)
    if k < 1:
        msg = "k must be at least 1."
        raise ValueError(msg)
    if min_size < 1:
        msg = "min_size must be at least 1."
        raise ValueError(msg)
    if not 0.0 < smoothness_degrees < 180.0:
        msg = "smoothness_degrees must be between 0 and 180."
        raise ValueError(msg)
    missing = [n for n in _NORMAL_ATTRIBUTES if n not in cloud.attribute_names]
    if missing:
        msg = (
            "region_growing needs surface normals "
            f"{list(_NORMAL_ATTRIBUTES)}; missing {missing}. Run estimate_normals "
            "on the cloud first."
        )
        raise ValueError(msg)

    normals = np.column_stack(
        [cloud.attribute(name).astype(np.float64) for name in _NORMAL_ATTRIBUTES]
    )
    lengths = np.linalg.norm(normals, axis=1)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    normals = normals / lengths[:, None]

    tree = cKDTree(cloud.xyz)
    neighbor_count = min(k + 1, n_points)
    _, neighbors = tree.query(cloud.xyz, k=neighbor_count, workers=-1)
    neighbors = np.atleast_2d(neighbors)[:, 1:]  # drop each point's self match

    has_curvature = curvature_attribute in cloud.attribute_names
    if has_curvature:
        curvature = cloud.attribute(curvature_attribute).astype(np.float64)
        seed_order = np.argsort(curvature)
    else:
        curvature = None
        seed_order = np.arange(n_points)

    cos_threshold = float(np.cos(np.radians(smoothness_degrees)))
    labels = np.full(n_points, -1, dtype=np.int64)
    visited = np.zeros(n_points, dtype=bool)
    next_label = 0
    for seed in seed_order:
        if visited[seed]:
            continue
        queue = [int(seed)]
        visited[seed] = True
        region = [int(seed)]
        while queue:
            point = queue.pop()
            for neighbor in neighbors[point]:
                neighbor = int(neighbor)
                if visited[neighbor]:
                    continue
                if abs(float(normals[point] @ normals[neighbor])) < cos_threshold:
                    continue
                visited[neighbor] = True
                region.append(neighbor)
                propagate = curvature_threshold is None or (
                    curvature is not None and curvature[neighbor] < curvature_threshold
                )
                if propagate:
                    queue.append(neighbor)
        if len(region) >= min_size:
            labels[region] = next_label
            next_label += 1

    return attach_attributes(
        cloud,
        {output_attribute: labels},
        "region_growing",
        {
            "k": k,
            "smoothness_degrees": smoothness_degrees,
            "curvature_threshold": curvature_threshold,
            "min_size": min_size,
        },
    )
