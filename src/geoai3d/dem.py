"""Digital terrain and surface models, differencing, and volumes.

Rasterises a point cloud onto a regular grid to produce a Digital Terrain Model
(DTM, the bare earth) or a Digital Surface Model (DSM, the top of everything),
differences two aligned rasters -- DSM minus DTM for object heights, or two
epochs for change -- and computes cut and fill volumes.

Both models are gridded on the cloud's full horizontal extent, so a DTM and DSM
made from the same cloud share a grid and can be differenced directly. A cell's
value is the lowest z within it for a DTM and the highest z for a DSM; cells
with no points are nodata. Everything here is pure NumPy; only writing a result
to a GeoTIFF needs the optional ``[gis]`` extra (see :mod:`geoai3d.io.geotiff`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.core.raster import Raster

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud


def _check_inputs(cloud: PointCloud, resolution: float) -> None:
    """Validate the common rasterisation inputs."""
    if len(cloud) == 0:
        msg = "Cannot rasterise an empty cloud."
        raise ValueError(msg)
    if resolution <= 0.0:
        msg = "resolution must be a positive number."
        raise ValueError(msg)
    if cloud.crs is None:
        msg = (
            "Cannot rasterise a cloud without a coordinate reference system. "
            "Set the cloud's CRS first."
        )
        raise ValueError(msg)


def _cloud_bounds(cloud: PointCloud) -> tuple[float, float, float, float]:
    """Return the cloud's horizontal bounds ``(min_x, min_y, max_x, max_y)``."""
    min_x, min_y, _, max_x, max_y, _ = cloud.bounds
    return (min_x, min_y, max_x, max_y)


def _rasterize(
    xyz: NDArray[Any],
    grid_bounds: tuple[float, float, float, float],
    crs: object,
    resolution: float,
    statistic: str,
    nodata: float,
) -> Raster:
    """Bin points onto a grid, reducing each cell by min or max z."""
    min_x, min_y, max_x, max_y = grid_bounds
    n_col = max(1, int(np.ceil((max_x - min_x) / resolution)))
    n_row = max(1, int(np.ceil((max_y - min_y) / resolution)))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    col = np.clip(((x - min_x) / resolution).astype(np.int64), 0, n_col - 1)
    row = np.clip(((max_y - y) / resolution).astype(np.int64), 0, n_row - 1)
    flat = row * n_col + col
    if statistic == "min":
        accumulator = np.full(n_row * n_col, np.inf)
        np.minimum.at(accumulator, flat, z)
    else:
        accumulator = np.full(n_row * n_col, -np.inf)
        np.maximum.at(accumulator, flat, z)
    grid = np.where(np.isfinite(accumulator), accumulator, np.nan)
    grid = grid.reshape(n_row, n_col)
    if not np.isnan(nodata):
        grid = np.where(np.isnan(grid), nodata, grid)
    transform = (resolution, 0.0, min_x, 0.0, -resolution, max_y)
    return Raster(grid, transform, crs, nodata=nodata)


def _finite(raster: Raster) -> NDArray[Any]:
    """Return the raster's data as float64 with nodata replaced by NaN."""
    data = raster.data.astype(np.float64)
    if not np.isnan(raster.nodata):
        data = np.where(data == raster.nodata, np.nan, data)
    return data


def to_dtm(
    cloud: PointCloud,
    *,
    resolution: float,
    ground_attribute: str = "is_ground",
    bounds: tuple[float, float, float, float] | None = None,
    nodata: float = float("nan"),
) -> Raster:
    """Rasterise a cloud to a Digital Terrain Model (bare earth).

    If the cloud carries a boolean ground attribute (from
    :func:`~geoai3d.ground`), only those points are used; otherwise every point
    is used and the lowest z per cell approximates the bare earth.

    Args:
        cloud: The cloud to rasterise. Its CRS must be set.
        resolution: Cell size in the cloud's CRS units.
        ground_attribute: Boolean attribute selecting ground points, if present.
        bounds: Grid extent ``(min_x, min_y, max_x, max_y)``. Defaults to the
            cloud's full horizontal extent, so a DTM and DSM from the same cloud
            align for differencing.
        nodata: Value for empty cells. Defaults to ``NaN``.

    Returns:
        A :class:`~geoai3d.Raster` of ground heights. Write it to disk with
        :func:`~geoai3d.to_geotiff`.

    Raises:
        ValueError: If the cloud is empty or has no CRS, ``resolution`` is not
            positive, or the ground attribute is present but all false.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_dtm
        >>> pts = np.random.default_rng(0).uniform(0.0, 10.0, (2000, 3))
        >>> dtm = to_dtm(PointCloud(pts, crs=28992), resolution=1.0)
        >>> dtm.shape
        (10, 10)
    """
    _check_inputs(cloud, resolution)
    if ground_attribute in cloud.attribute_names:
        mask = np.asarray(cloud.attribute(ground_attribute)).astype(bool)
        xyz = cloud.xyz[mask]
        if xyz.shape[0] == 0:
            msg = (
                f"No ground points to rasterise: attribute {ground_attribute!r} "
                "is all false."
            )
            raise ValueError(msg)
    else:
        xyz = cloud.xyz
    grid_bounds = bounds if bounds is not None else _cloud_bounds(cloud)
    return _rasterize(xyz, grid_bounds, cloud.crs, resolution, "min", nodata)


def to_dsm(
    cloud: PointCloud,
    *,
    resolution: float,
    bounds: tuple[float, float, float, float] | None = None,
    nodata: float = float("nan"),
) -> Raster:
    """Rasterise a cloud to a Digital Surface Model (the top surface).

    Uses the highest z per cell over every point, capturing the top of
    vegetation and buildings.

    Args:
        cloud: The cloud to rasterise. Its CRS must be set.
        resolution: Cell size in the cloud's CRS units.
        bounds: Grid extent ``(min_x, min_y, max_x, max_y)``. Defaults to the
            cloud's full horizontal extent.
        nodata: Value for empty cells. Defaults to ``NaN``.

    Returns:
        A :class:`~geoai3d.Raster` of surface heights. Write it to disk with
        :func:`~geoai3d.to_geotiff`.

    Raises:
        ValueError: If the cloud is empty or has no CRS, or ``resolution`` is
            not positive.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_dsm
        >>> pts = np.random.default_rng(0).uniform(0.0, 10.0, (2000, 3))
        >>> dsm = to_dsm(PointCloud(pts, crs=28992), resolution=1.0)
        >>> dsm.shape
        (10, 10)
    """
    _check_inputs(cloud, resolution)
    grid_bounds = bounds if bounds is not None else _cloud_bounds(cloud)
    return _rasterize(cloud.xyz, grid_bounds, cloud.crs, resolution, "max", nodata)


def difference(minuend: Raster, subtrahend: Raster) -> Raster:
    """Subtract one aligned raster from another, cell by cell.

    ``difference(dsm, dtm)`` gives a normalised surface (object heights);
    ``difference(later, earlier)`` gives elevation change between epochs. A cell
    is nodata (``NaN``) where either input is nodata.

    Args:
        minuend: The raster to subtract from.
        subtrahend: The raster to subtract.

    Returns:
        A :class:`~geoai3d.Raster` of ``minuend - subtrahend``.

    Raises:
        ValueError: If the rasters differ in shape, affine transform, or CRS
            (rasterise both on the same ``bounds`` and ``resolution`` first).

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_dtm, to_dsm, difference
        >>> pts = np.random.default_rng(0).uniform(0.0, 10.0, (2000, 3))
        >>> cloud = PointCloud(pts, crs=28992)
        >>> ndsm = difference(to_dsm(cloud, resolution=1.0),
        ...                   to_dtm(cloud, resolution=1.0))
        >>> ndsm.shape
        (10, 10)
    """
    if minuend.shape != subtrahend.shape:
        msg = f"Rasters are not aligned: shapes {minuend.shape} vs {subtrahend.shape}."
        raise ValueError(msg)
    if not np.allclose(minuend.transform, subtrahend.transform):
        msg = (
            "Rasters are not aligned: their affine transforms differ. "
            "Rasterise both on the same bounds and resolution."
        )
        raise ValueError(msg)
    if minuend.crs != subtrahend.crs:
        msg = "Rasters have different CRS; reproject to a common CRS first."
        raise ValueError(msg)
    result = _finite(minuend) - _finite(subtrahend)
    return Raster(result, minuend.transform, minuend.crs, nodata=float("nan"))


def volume(raster: Raster, *, base: float = 0.0) -> dict[str, float]:
    """Compute cut and fill volumes of a raster relative to a base level.

    Interprets the raster as a height field (typically a difference raster) and
    integrates it against ``base``: cells above ``base`` contribute fill, cells
    below contribute cut, each weighted by the cell area. Nodata cells are
    ignored.

    Args:
        raster: The height raster to integrate.
        base: The reference level to measure against.

    Returns:
        A dict with ``"cut"``, ``"fill"``, and ``"net"`` (fill minus cut)
        volumes, in cubic CRS units.

    Example:
        >>> import numpy as np
        >>> from geoai3d import Raster, volume
        >>> data = np.array([[1.0, 1.0], [1.0, 1.0]])  # 1 m over four 2 m cells
        >>> raster = Raster(data, (2.0, 0.0, 0.0, 0.0, -2.0, 4.0), 28992)
        >>> volume(raster)["fill"]
        16.0
    """
    data = _finite(raster)
    valid = ~np.isnan(data)
    difference_from_base = data - base
    res_x, res_y = raster.resolution
    cell_area = res_x * res_y
    fill = float(
        np.sum(difference_from_base[valid & (difference_from_base > 0.0)]) * cell_area
    )
    cut = float(
        -np.sum(difference_from_base[valid & (difference_from_base < 0.0)]) * cell_area
    )
    return {"cut": cut, "fill": fill, "net": fill - cut}
