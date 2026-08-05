"""Fitting geometric primitives to point clouds with RANSAC.

RANSAC (RANdom SAmple Consensus) fits a shape to data that contains outliers by
repeatedly proposing a model from a minimal random sample and keeping the one
that most points agree with (its inliers). This module fits the workhorse
primitive -- an infinite plane -- which recovers ground, walls, and roof
facets, and gives a robust dominant-plane estimate for terrestrial scans where
the cloth filter's 2.5-D assumption is weak. Cylinder and sphere fitting are
planned additions.

The plane is ``a*x + b*y + c*z + d = 0`` with unit normal ``(a, b, c)``. After
the consensus step the plane is refit to all inliers by a least-squares
(principal-component) fit, so the returned model uses every agreeing point, not
just the three that seeded it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud


@dataclass(frozen=True)
class Plane:
    """A fitted plane ``a*x + b*y + c*z + d = 0`` and its inliers.

    Attributes:
        normal: The unit normal ``(a, b, c)``.
        offset: The plane offset ``d``.
        inliers: Boolean mask over the fitted cloud, true for inlier points.
    """

    normal: tuple[float, float, float]
    offset: float
    inliers: NDArray[Any]

    @property
    def num_inliers(self) -> int:
        """Return the number of inlier points."""
        return int(np.count_nonzero(self.inliers))

    def signed_distance(self, points: NDArray[Any]) -> NDArray[Any]:
        """Return the signed perpendicular distance of points to the plane.

        Args:
            points: An ``(n, 3)`` array of coordinates.

        Returns:
            An ``(n,)`` array of signed distances (the normal side is positive).
        """
        normal = np.asarray(self.normal, dtype=np.float64)
        return np.asarray(points, dtype=np.float64) @ normal + self.offset


def fit_plane(
    cloud: PointCloud,
    *,
    distance_threshold: float = 0.1,
    max_iterations: int = 1000,
    seed: int = 0,
) -> Plane:
    """Fit the dominant plane to a cloud with RANSAC.

    Args:
        cloud: The cloud to fit. Needs at least three points.
        distance_threshold: Maximum perpendicular distance, in CRS units, for a
            point to count as an inlier.
        max_iterations: Number of random three-point samples to try.
        seed: Seed for the random sampling, so the fit is reproducible.

    Returns:
        The fitted :class:`Plane`, refit to its inliers.

    Raises:
        ValueError: If the cloud has fewer than three points, or
            ``distance_threshold`` or ``max_iterations`` is not positive, or no
            plane could be fitted (all samples were collinear).

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, fit_plane
        >>> rng = np.random.default_rng(0)
        >>> xy = rng.uniform(0.0, 10.0, (500, 2))
        >>> z = 0.0 * xy[:, 0]  # a horizontal plane at z = 0
        >>> pts = np.column_stack([xy, z])
        >>> plane = fit_plane(PointCloud(pts, crs=28992), distance_threshold=0.01)
        >>> bool(abs(abs(plane.normal[2]) - 1.0) < 1e-6)
        True
    """
    n_points = len(cloud)
    if n_points < 3:
        msg = "fit_plane needs at least three points."
        raise ValueError(msg)
    if distance_threshold <= 0.0:
        msg = "distance_threshold must be a positive number."
        raise ValueError(msg)
    if max_iterations < 1:
        msg = "max_iterations must be at least 1."
        raise ValueError(msg)

    xyz = cloud.xyz
    rng = np.random.default_rng(seed)
    best_inliers: NDArray[Any] | None = None
    best_count = -1
    for _ in range(max_iterations):
        sample = xyz[rng.choice(n_points, size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:  # collinear sample, no plane
            continue
        normal = normal / norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(xyz @ normal + offset) < distance_threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        msg = "Could not fit a plane: every sample was collinear."
        raise ValueError(msg)

    # Refit to all inliers by least squares: the normal is the direction of
    # least spread (the smallest right-singular vector of the centred inliers).
    inlier_points = xyz[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(inlier_points - centroid)
    normal = right_vectors[-1]
    offset = -float(normal @ centroid)
    inliers = np.abs(xyz @ normal + offset) < distance_threshold
    return Plane(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset=offset,
        inliers=inliers,
    )
