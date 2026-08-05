"""Unsupervised full-scene decomposition.

:func:`segment` chains the frugal building blocks into a single call that turns
a raw cloud into a labelled scene with no training data, no labels, and no GPU:
filter the ground with the cloth simulation, describe the remaining points with
surface normals and eigenvalue features, then grow smooth regions on them. The
result is one integer label per point -- ground as a single segment and each
above-ground object as its own -- which a survey office can run on a laptop.

This is deliberately opinionated: it exposes a few high-level knobs and sensible
defaults. For finer control, call :func:`~geoai3d.ground`,
:func:`~geoai3d.estimate_normals`, :func:`~geoai3d.geometric_features`, and
:func:`~geoai3d.region_growing` yourself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoai3d.cluster import region_growing
from geoai3d.core._derive import attach_attributes
from geoai3d.features import geometric_features
from geoai3d.geometry import estimate_normals
from geoai3d.ground import ground

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud

_GROUND_LABEL = 0
_MIN_POINTS_FOR_FEATURES = 3


def segment(
    cloud: PointCloud,
    *,
    detect_ground: bool = True,
    ground_resolution: float = 1.0,
    ground_class_threshold: float = 0.5,
    normal_k: int = 16,
    smoothness_degrees: float = 15.0,
    min_region_size: int = 10,
    output_attribute: str = "segment",
) -> PointCloud:
    """Decompose a scene into ground and object segments, unsupervised.

    Runs, in order: ground filtering (unless ``detect_ground`` is false and the
    cloud already carries an ``is_ground`` column), then surface normals and
    geometric features on the non-ground points, then normal-based region
    growing. Ground points get label ``0``; each grown region gets a distinct
    label from ``1`` upward; points left unassigned by region growing get ``-1``.

    Args:
        cloud: The cloud to segment. Its CRS must be set.
        detect_ground: If true, run the cloth-simulation ground filter. If
            false, use an existing ``is_ground`` column, or treat all points as
            non-ground if there is none.
        ground_resolution: Cloth resolution for the ground filter, in CRS units.
        ground_class_threshold: Ground distance threshold for the filter.
        normal_k: Neighbours used for normals, features, and region growth.
        smoothness_degrees: Maximum normal-to-normal angle for region growth.
        min_region_size: Minimum points for an object region to be kept.
        output_attribute: Name of the integer segment label column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the integer segment label column
        added, the CRS and provenance carried through.

    Raises:
        ValueError: If the cloud is empty or has no CRS.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, segment
        >>> rng = np.random.default_rng(0)
        >>> ground_pts = rng.uniform(0.0, 20.0, (4000, 3))
        >>> ground_pts[:, 2] = 0.0
        >>> segmented = segment(PointCloud(ground_pts, crs=28992))
        >>> int(segmented.attribute("segment").min())
        0
    """
    n_points = len(cloud)
    if n_points == 0:
        msg = "Cannot segment an empty cloud."
        raise ValueError(msg)
    if cloud.crs is None:
        msg = (
            "Cannot segment a cloud without a coordinate reference system. "
            "Set the cloud's CRS first."
        )
        raise ValueError(msg)

    if detect_ground:
        classified = ground(
            cloud,
            cloth_resolution=ground_resolution,
            class_threshold=ground_class_threshold,
        )
        is_ground = np.asarray(classified.attribute("is_ground")).astype(bool)
    elif "is_ground" in cloud.attribute_names:
        is_ground = np.asarray(cloud.attribute("is_ground")).astype(bool)
    else:
        is_ground = np.zeros(n_points, dtype=bool)

    labels = np.full(n_points, -1, dtype=np.int64)
    labels[is_ground] = _GROUND_LABEL

    object_positions = np.flatnonzero(~is_ground)
    if object_positions.size >= _MIN_POINTS_FOR_FEATURES:
        objects = cloud[~is_ground]
        objects = estimate_normals(objects, k=normal_k)
        objects = geometric_features(objects, k=normal_k)
        objects = region_growing(
            objects,
            k=normal_k,
            smoothness_degrees=smoothness_degrees,
            min_size=min_region_size,
        )
        region_labels = np.asarray(objects.attribute("segment"))
        # Offset region ids by one so they do not collide with the ground label,
        # and keep region-growing's -1 (unassigned) as -1.
        mapped = np.where(region_labels >= 0, region_labels + 1, -1)
        labels[object_positions] = mapped

    return attach_attributes(
        cloud,
        {output_attribute: labels},
        "segment",
        {
            "detect_ground": detect_ground,
            "ground_resolution": ground_resolution,
            "ground_class_threshold": ground_class_threshold,
            "normal_k": normal_k,
            "smoothness_degrees": smoothness_degrees,
            "min_region_size": min_region_size,
        },
    )
