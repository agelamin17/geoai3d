"""Ground filtering and DTM extraction by cloth simulation (CSF).

The Cloth Simulation Filter (Zhang et al., 2016) separates ground from
non-ground points without training data. The idea is physical: turn the cloud
upside down, then drape a stiff cloth over it from above under gravity. The
cloth settles onto the (now uppermost) bare-earth surface and drapes over the
inverted buildings and vegetation without falling through them. Points lying
within a threshold of the settled cloth are ground; the rest are objects.

The cloth is a regular grid of particles that move only vertically. Each
iteration applies gravity, stops any particle that reaches the terrain height
below it (marking it fixed), and then runs a few rigidity passes that keep
neighbouring particles at similar heights -- the ``rigidness`` control. A
stiffer cloth (higher ``rigidness``) follows flat, built-up terrain well; a
softer cloth (``rigidness=1``) follows steep natural terrain better.

This is a pure-NumPy implementation working on the cloth grid and a rasterised
terrain, so its memory scales with the grid, not the point count, and it needs
no compiler. It assumes an airborne, roughly 2.5-D scene (one ground height per
location); for terrestrial scans with walls and overhangs, treat the result as
a first pass. An optional ``backend="pdal"`` delegates to PDAL's ``filters.csf``
for users who have PDAL installed.

Reference:
    Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016).
    An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth
    Simulation. Remote Sensing, 8(6), 501.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from geoai3d.core._derive import attach_attributes
from geoai3d.core.pointcloud import PointCloud

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _fill_holes(terrain: NDArray[Any]) -> NDArray[Any]:
    """Fill empty (``-inf``) grid cells from their filled neighbours.

    Repeated 4-neighbour dilation, taking the highest neighbouring terrain
    height, so gaps in the collision surface are bridged by the surrounding
    ground rather than letting the cloth fall through them.
    """
    filled_terrain = terrain.copy()
    for _ in range(terrain.shape[0] + terrain.shape[1] + 1):
        known = np.isfinite(filled_terrain)
        if known.all():
            break
        best = np.full_like(filled_terrain, -np.inf)
        best[1:, :] = np.maximum(
            best[1:, :], np.where(known[:-1, :], filled_terrain[:-1, :], -np.inf)
        )
        best[:-1, :] = np.maximum(
            best[:-1, :], np.where(known[1:, :], filled_terrain[1:, :], -np.inf)
        )
        best[:, 1:] = np.maximum(
            best[:, 1:], np.where(known[:, :-1], filled_terrain[:, :-1], -np.inf)
        )
        best[:, :-1] = np.maximum(
            best[:, :-1], np.where(known[:, 1:], filled_terrain[:, 1:], -np.inf)
        )
        take = (~known) & np.isfinite(best)
        if not take.any():
            break
        filled_terrain[take] = best[take]
    return filled_terrain


def _csf(
    xyz: NDArray[Any],
    *,
    cloth_resolution: float,
    class_threshold: float,
    rigidness: int,
    iterations: int,
    time_step: float,
) -> NDArray[Any]:
    """Run the cloth simulation and return a boolean ground mask."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    inverted = -z  # ground (low z) becomes the uppermost surface
    origin_x, origin_y = float(x.min()), float(y.min())
    n_x = int(np.floor((float(x.max()) - origin_x) / cloth_resolution)) + 2
    n_y = int(np.floor((float(y.max()) - origin_y) / cloth_resolution)) + 2

    # Collision surface: the highest inverted height (lowest ground) per cell.
    cell_i = np.clip(
        np.round((x - origin_x) / cloth_resolution).astype(np.int64), 0, n_x - 1
    )
    cell_j = np.clip(
        np.round((y - origin_y) / cloth_resolution).astype(np.int64), 0, n_y - 1
    )
    flat_cell = cell_i * n_y + cell_j
    terrain = np.full(n_x * n_y, -np.inf)
    np.maximum.at(terrain, flat_cell, inverted)
    terrain = _fill_holes(terrain.reshape(n_x, n_y))

    cloth = np.full((n_x, n_y), float(inverted.max()) + 1.0)
    previous = cloth.copy()
    movable = np.ones((n_x, n_y), dtype=bool)
    gravity = time_step**2

    for iteration in range(iterations):
        snapshot = cloth.copy()
        stepped = 2.0 * cloth - previous  # Verlet integration, vertical only
        stepped[movable] -= gravity
        previous[movable] = snapshot[movable]
        cloth[movable] = stepped[movable]
        landed = cloth < terrain
        cloth[landed] = terrain[landed]
        movable[landed] = False
        previous[landed] = terrain[landed]

        for _ in range(rigidness):
            for axis in (0, 1):
                low = [slice(None), slice(None)]
                high = [slice(None), slice(None)]
                low[axis] = slice(0, -1)
                high[axis] = slice(1, None)
                z_low, z_high = cloth[tuple(low)], cloth[tuple(high)]
                movable_low, movable_high = movable[tuple(low)], movable[tuple(high)]
                difference = z_high - z_low
                both = movable_low & movable_high
                only_low = movable_low & ~movable_high
                only_high = ~movable_low & movable_high
                new_low, new_high = z_low.copy(), z_high.copy()
                new_low[both] += 0.5 * difference[both]
                new_high[both] -= 0.5 * difference[both]
                new_low[only_low] += difference[only_low]
                new_high[only_high] -= difference[only_high]
                cloth[tuple(low)] = new_low
                cloth[tuple(high)] = new_high
            landed = cloth < terrain
            cloth[landed] = terrain[landed]
            movable[landed] = False

        if iteration > 0 and float(np.max(np.abs(cloth - snapshot))) < 1.0e-3:
            break

    frac_x = (x - origin_x) / cloth_resolution
    frac_y = (y - origin_y) / cloth_resolution
    index_x = np.clip(np.floor(frac_x).astype(np.int64), 0, n_x - 2)
    index_y = np.clip(np.floor(frac_y).astype(np.int64), 0, n_y - 2)
    weight_x = frac_x - index_x
    weight_y = frac_y - index_y
    cloth_at_point = (
        cloth[index_x, index_y] * (1.0 - weight_x) * (1.0 - weight_y)
        + cloth[index_x + 1, index_y] * weight_x * (1.0 - weight_y)
        + cloth[index_x, index_y + 1] * (1.0 - weight_x) * weight_y
        + cloth[index_x + 1, index_y + 1] * weight_x * weight_y
    )
    ground_mask = (cloth_at_point - inverted) < class_threshold
    return np.asarray(ground_mask, dtype=np.bool_)


def _csf_pdal(
    xyz: NDArray[Any],
    *,
    cloth_resolution: float,
    class_threshold: float,
    rigidness: int,
    iterations: int,
    time_step: float,
) -> NDArray[Any]:
    """Run PDAL's ``filters.csf`` and return a boolean ground mask."""
    try:
        import pdal
    except ImportError as exc:
        msg = (
            "backend='pdal' needs PDAL's Python bindings, which are not "
            "installed. Install them (for example 'conda install -c conda-forge "
            "python-pdal') or use backend='csf', the pure-NumPy default."
        )
        raise ValueError(msg) from exc

    array = np.zeros(len(xyz), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    array["X"] = xyz[:, 0]
    array["Y"] = xyz[:, 1]
    array["Z"] = xyz[:, 2]
    pipeline_json = json.dumps(
        [
            {
                "type": "filters.csf",
                "resolution": cloth_resolution,
                "threshold": class_threshold,
                "rigidness": rigidness,
                "iterations": iterations,
                "step": time_step,
            }
        ]
    )
    pipeline = pdal.Pipeline(pipeline_json, arrays=[array])
    pipeline.execute()
    classification = np.asarray(pipeline.arrays[0]["Classification"])
    return np.asarray(classification == 2, dtype=np.bool_)  # ASPRS class 2 == ground


def ground(
    cloud: PointCloud,
    *,
    backend: str = "csf",
    cloth_resolution: float = 1.0,
    class_threshold: float = 0.5,
    rigidness: int = 3,
    iterations: int = 500,
    time_step: float = 0.65,
    output_attribute: str = "is_ground",
) -> PointCloud:
    """Classify ground points with the Cloth Simulation Filter.

    Adds a boolean column that is true for ground points, using the
    training-free cloth simulation described in the module docstring. Runs on a
    CPU with no compiler; memory scales with the cloth grid, not the point
    count.

    Args:
        cloud: The cloud to filter.
        backend: ``"csf"`` for the built-in pure-NumPy filter (default), or
            ``"pdal"`` to delegate to PDAL's ``filters.csf`` (requires PDAL).
        cloth_resolution: Cloth grid spacing in the cloud's CRS units. Smaller
            resolves finer terrain but costs more memory and time.
        class_threshold: Maximum distance from the settled cloth, in CRS units,
            for a point to count as ground.
        rigidness: Cloth stiffness as a number of rigidity passes per iteration.
            Use 3 for flat or built-up terrain, 1 for steep natural terrain.
        iterations: Maximum simulation iterations.
        time_step: Simulation time step; larger settles faster but less stably.
        output_attribute: Name of the boolean ground column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the boolean ground column added,
        the CRS and provenance carried through.

    Raises:
        ValueError: If the cloud is empty; if ``backend`` is not ``"csf"`` or
            ``"pdal"``; if ``cloth_resolution``, ``class_threshold`` or
            ``time_step`` is not positive; or if ``rigidness`` or ``iterations``
            is below 1. Also if ``backend="pdal"`` and PDAL is not installed.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, ground
        >>> rng = np.random.default_rng(0)
        >>> flat = rng.uniform(0.0, 10.0, (500, 3))
        >>> flat[:, 2] = 0.0  # a flat ground plane
        >>> out = ground(PointCloud(flat, crs=28992), cloth_resolution=1.0)
        >>> bool(out.attribute("is_ground").all())
        True
    """
    if len(cloud) == 0:
        msg = "Cannot filter ground on an empty cloud."
        raise ValueError(msg)
    if cloth_resolution <= 0.0:
        msg = "cloth_resolution must be a positive number."
        raise ValueError(msg)
    if class_threshold <= 0.0:
        msg = "class_threshold must be a positive number."
        raise ValueError(msg)
    if rigidness < 1:
        msg = "rigidness must be at least 1."
        raise ValueError(msg)
    if iterations < 1:
        msg = "iterations must be at least 1."
        raise ValueError(msg)
    if time_step <= 0.0:
        msg = "time_step must be a positive number."
        raise ValueError(msg)

    xyz = cloud.xyz
    if backend == "csf":
        is_ground = _csf(
            xyz,
            cloth_resolution=cloth_resolution,
            class_threshold=class_threshold,
            rigidness=rigidness,
            iterations=iterations,
            time_step=time_step,
        )
    elif backend == "pdal":
        is_ground = _csf_pdal(
            xyz,
            cloth_resolution=cloth_resolution,
            class_threshold=class_threshold,
            rigidness=rigidness,
            iterations=iterations,
            time_step=time_step,
        )
    else:
        msg = f"Unknown backend {backend!r}; use 'csf' or 'pdal'."
        raise ValueError(msg)

    return attach_attributes(
        cloud,
        {output_attribute: is_ground},
        "ground",
        {
            "backend": backend,
            "cloth_resolution": cloth_resolution,
            "class_threshold": class_threshold,
            "rigidness": rigidness,
            "iterations": iterations,
            "time_step": time_step,
        },
    )
