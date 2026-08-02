"""The in-memory point-cloud representation used throughout GEOAI_3D.

:class:`PointCloud` stores coordinates and per-point attributes in a columnar
(struct-of-arrays) layout: coordinates in one contiguous ``float64`` array and
each attribute in its own array. Two properties matter for later stages. Adding
a computed feature column (planarity, linearity, ...) is cheap because it is
just another array set alongside the others, and the contiguous coordinate
array can be handed straight to spatial-index and geometry routines. Coordinates
are always ``float64``: single precision cannot represent projected national
coordinates without silently losing centimetres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.core._crs import coerce_crs

if TYPE_CHECKING:
    import pyproj

    from geoai3d.core.provenance import Provenance


class PointCloud:
    """A georeferenced 3D point cloud stored column by column.

    Args:
        xyz: Point coordinates as an array of shape ``(n_points, 3)``. Values
            are stored as ``float64``; other numeric inputs are converted.
        attributes: Optional per-point attribute arrays keyed by name. Each
            array must be one-dimensional with length ``n_points`` (for
            example ``intensity`` or ``classification``).
        crs: Coordinate reference system the coordinates are expressed in.
            Optional at construction; the input/output and transform routines
            added in later stages require it and raise if it is missing. This
            slot will hold a ``pyproj`` CRS once the georeferencing layer lands.
        provenance: Lineage record describing how this cloud was produced.

    Raises:
        ValueError: If ``xyz`` is not a two-dimensional array with three
            columns, or if an attribute is not one-dimensional or does not
            match the number of points.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud
        >>> xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> cloud = PointCloud(xyz, attributes={"intensity": np.array([10, 20])})
        >>> len(cloud)
        2
        >>> cloud.attribute_names
        ['intensity']
    """

    def __init__(
        self,
        xyz: NDArray[Any],
        attributes: dict[str, NDArray[Any]] | None = None,
        crs: object | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        """Validate the inputs and store them in columnar form.

        The ``crs`` argument is coerced to a :class:`pyproj.CRS` so that
        :attr:`crs` always returns a resolved reference system (or ``None``),
        however the cloud was built.

        Raises:
            ValueError: If ``xyz`` is not ``(n_points, 3)``; if an attribute is
                not one-dimensional or its length does not match the point
                count; or if ``crs`` cannot be interpreted as a coordinate
                reference system.
        """
        coordinates: NDArray[np.float64] = np.ascontiguousarray(xyz, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            msg = (
                "xyz must have shape (n_points, 3); "
                f"got an array with shape {coordinates.shape}."
            )
            raise ValueError(msg)
        self._xyz: NDArray[np.float64] = coordinates
        self._attributes: dict[str, NDArray[Any]] = {}
        if attributes is not None:
            for name, values in attributes.items():
                self._set_attribute_checked(name, values)
        self._crs = coerce_crs(crs) if crs is not None else None
        self._provenance = provenance

    def _set_attribute_checked(self, name: str, values: NDArray[Any]) -> None:
        """Validate an attribute array and store it under ``name``."""
        array: NDArray[Any] = np.asarray(values)
        if array.ndim != 1:
            msg = (
                f"Attribute {name!r} must be one-dimensional; "
                f"got an array with {array.ndim} dimensions."
            )
            raise ValueError(msg)
        if array.shape[0] != self._xyz.shape[0]:
            msg = (
                f"Attribute {name!r} has length {array.shape[0]}, but the cloud "
                f"has {self._xyz.shape[0]} points."
            )
            raise ValueError(msg)
        self._attributes[name] = array

    @property
    def xyz(self) -> NDArray[np.float64]:
        """Return the point coordinates as a ``(n_points, 3)`` ``float64`` array."""
        return self._xyz

    @property
    def crs(self) -> pyproj.CRS | None:
        """Return the coordinate reference system, or ``None`` if unset."""
        return self._crs

    @property
    def provenance(self) -> Provenance | None:
        """Return the lineage record, or ``None`` if unset."""
        return self._provenance

    @property
    def attribute_names(self) -> list[str]:
        """Return the attribute names, in insertion order."""
        return list(self._attributes)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return axis-aligned bounds ``(minx, miny, minz, maxx, maxy, maxz)``.

        Raises:
            ValueError: If the cloud is empty.
        """
        if len(self) == 0:
            msg = "Cannot compute the bounds of an empty point cloud."
            raise ValueError(msg)
        mins = self._xyz.min(axis=0)
        maxs = self._xyz.max(axis=0)
        return (
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        )

    def attribute(self, name: str) -> NDArray[Any]:
        """Return the attribute array stored under ``name``.

        Args:
            name: Attribute name.

        Returns:
            The one-dimensional attribute array.

        Raises:
            KeyError: If no attribute with that name exists.
        """
        if name not in self._attributes:
            msg = (
                f"No attribute named {name!r}; "
                f"available attributes: {self.attribute_names}."
            )
            raise KeyError(msg)
        return self._attributes[name]

    def with_attribute(self, name: str, values: NDArray[Any]) -> PointCloud:
        """Return a new cloud with an attribute column added or replaced.

        The coordinate array is shared with this cloud rather than copied, so
        attaching a computed feature to a large cloud is inexpensive.

        Args:
            name: Attribute name to add or replace.
            values: One-dimensional array of length ``len(self)``.

        Returns:
            A new :class:`PointCloud` carrying the extra attribute. The original
            cloud is left unchanged.

        Raises:
            ValueError: If ``values`` is not one-dimensional or its length does
                not match the number of points.

        Example:
            >>> import numpy as np
            >>> from geoai3d import PointCloud
            >>> cloud = PointCloud(np.zeros((3, 3)))
            >>> planar = cloud.with_attribute("planarity", np.ones(3))
            >>> planar.attribute("planarity").tolist()
            [1.0, 1.0, 1.0]
            >>> cloud.attribute_names
            []
        """
        new = PointCloud(
            self._xyz,
            attributes=dict(self._attributes),
            crs=self._crs,
            provenance=self._copy_provenance(),
        )
        new._set_attribute_checked(name, values)
        return new

    def __len__(self) -> int:
        """Return the number of points in the cloud."""
        return int(self._xyz.shape[0])

    def __getitem__(self, selector: slice | NDArray[Any]) -> PointCloud:
        """Return a new cloud containing only the selected points.

        Args:
            selector: A slice, a boolean mask of length ``len(self)``, or an
                array of integer indices.

        Returns:
            A new :class:`PointCloud` with coordinates and every attribute
            filtered the same way, and the CRS and provenance carried through.

        Example:
            >>> import numpy as np
            >>> from geoai3d import PointCloud
            >>> cloud = PointCloud(np.arange(9.0).reshape(3, 3))
            >>> cloud[np.array([True, False, True])].xyz.shape
            (2, 3)
        """
        selected_attributes = {
            name: values[selector] for name, values in self._attributes.items()
        }
        return PointCloud(
            self._xyz[selector],
            attributes=selected_attributes,
            crs=self._crs,
            provenance=self._copy_provenance(),
        )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return (
            f"PointCloud(n_points={len(self)}, "
            f"attributes={self.attribute_names}, crs={self._crs!r})"
        )

    def _copy_provenance(self) -> Provenance | None:
        """Return an independent copy of the provenance, or ``None``."""
        return self._provenance.copy() if self._provenance is not None else None
