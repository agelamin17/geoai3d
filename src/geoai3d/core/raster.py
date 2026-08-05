"""The in-memory raster grid used for DTM/DSM and derived surfaces.

:class:`Raster` is to gridded data what :class:`~geoai3d.PointCloud` is to
points: a plain container that keeps the values together with their
georeferencing so a coordinate reference system can never be dropped by
accident. It holds a 2D array, an affine transform, a CRS, and a nodata value.

The transform follows the GDAL/affine convention mapping pixel ``(column, row)``
to world ``(x, y)``::

    x = a * column + b * row + c
    y = d * column + e * row + f

stored as the 6-tuple ``(a, b, c, d, e, f)``. A north-up raster has
``a = resolution``, ``e = -resolution``, ``b = d = 0``, and ``(c, f)`` at the
top-left corner. Empty cells are marked with the nodata value (``NaN`` by
default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.core._crs import coerce_crs

if TYPE_CHECKING:
    import pyproj

_Transform = tuple[float, float, float, float, float, float]


class Raster:
    """A 2D georeferenced raster: values with an affine transform and CRS."""

    def __init__(
        self,
        data: NDArray[Any],
        transform: tuple[float, ...],
        crs: object,
        nodata: float = float("nan"),
    ) -> None:
        """Validate and store the grid, transform, CRS, and nodata value.

        Args:
            data: A 2D array of cell values, shape ``(rows, cols)``.
            transform: The affine transform as a 6-tuple ``(a, b, c, d, e, f)``.
            crs: The coordinate reference system (EPSG code, WKT, or
                ``pyproj.CRS``); coerced to a :class:`pyproj.CRS`.
            nodata: Value marking empty cells. Defaults to ``NaN``.

        Raises:
            ValueError: If ``data`` is not 2D, ``transform`` is not length 6,
                or ``crs`` is ``None`` or uninterpretable.
        """
        array: NDArray[Any] = np.ascontiguousarray(data)
        if array.ndim != 2:
            msg = f"data must be 2D (rows, cols); got shape {array.shape}."
            raise ValueError(msg)
        if len(tuple(transform)) != 6:
            msg = "transform must be a 6-tuple (a, b, c, d, e, f)."
            raise ValueError(msg)
        if crs is None:
            msg = (
                "A raster requires a coordinate reference system; pass crs= "
                "(an EPSG code, WKT string, or pyproj.CRS)."
            )
            raise ValueError(msg)
        self._data = array
        self._transform: _Transform = (
            float(transform[0]),
            float(transform[1]),
            float(transform[2]),
            float(transform[3]),
            float(transform[4]),
            float(transform[5]),
        )
        self._crs = coerce_crs(crs)
        self._nodata = float(nodata)

    @property
    def data(self) -> NDArray[Any]:
        """Return the cell values as a 2D ``(rows, cols)`` array."""
        return self._data

    @property
    def transform(self) -> _Transform:
        """Return the affine transform as ``(a, b, c, d, e, f)``."""
        return self._transform

    @property
    def crs(self) -> pyproj.CRS:
        """Return the coordinate reference system."""
        return self._crs

    @property
    def nodata(self) -> float:
        """Return the value marking empty cells."""
        return self._nodata

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid shape as ``(rows, cols)``."""
        return (int(self._data.shape[0]), int(self._data.shape[1]))

    @property
    def height(self) -> int:
        """Return the number of rows."""
        return int(self._data.shape[0])

    @property
    def width(self) -> int:
        """Return the number of columns."""
        return int(self._data.shape[1])

    @property
    def resolution(self) -> tuple[float, float]:
        """Return the ``(x, y)`` cell size in CRS units (both positive)."""
        return (abs(self._transform[0]), abs(self._transform[4]))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return axis-aligned bounds ``(min_x, min_y, max_x, max_y)``."""
        a, _, c, _, e, f = self._transform
        min_x = c
        max_y = f
        max_x = c + self.width * a
        min_y = f + self.height * e
        return (
            float(min(min_x, max_x)),
            float(min(min_y, max_y)),
            float(max(min_x, max_x)),
            float(max(min_y, max_y)),
        )

    def __repr__(self) -> str:
        """Return a concise, informative representation."""
        epsg = self._crs.to_epsg()
        crs_label = f"EPSG:{epsg}" if epsg is not None else "CRS"
        res_x, res_y = self.resolution
        return (
            f"Raster(shape=({self.height}, {self.width}), "
            f"resolution=({res_x:g}, {res_y:g}), {crs_label})"
        )
