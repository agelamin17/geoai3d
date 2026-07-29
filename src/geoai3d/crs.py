"""Reprojecting point clouds between coordinate reference systems.

Two operations live here. :func:`reproject` changes the horizontal CRS only,
leaving heights untouched. :func:`reproject_3d` performs a full three-dimensional
transform, including the vertical datum, so it can convert between ellipsoidal
heights (as GNSS delivers) and orthometric heights (relative to a geoid model
such as NAP or NAVD88). That vertical conversion is the one most 3D tools get
wrong: mixing ellipsoidal and orthometric heights silently introduces errors of
tens of metres, and it is exactly what this module exists to handle correctly.

The geoid separation needed by :func:`reproject_3d` comes from PROJ transform
grids. If a grid is not available locally, pass ``allow_network=True`` to let
pyproj fetch it from the PROJ CDN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj
from numpy.typing import NDArray

from geoai3d.core._crs import coerce_crs
from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import Provenance


def _attributes_of(cloud: PointCloud) -> dict[str, NDArray[Any]]:
    """Return a shallow copy of a cloud's attribute columns by name."""
    return {name: cloud.attribute(name) for name in cloud.attribute_names}


def _provenance_for(cloud: PointCloud) -> Provenance:
    """Return a copy of the cloud's provenance, or a fresh record."""
    return cloud.provenance.copy() if cloud.provenance is not None else Provenance()


def _has_vertical_axis(crs: pyproj.CRS) -> bool:
    """Return whether a CRS carries a vertical (height) axis."""
    return len(crs.axis_info) >= 3


def reproject(cloud: PointCloud, target_crs: object) -> PointCloud:
    """Reproject a cloud's horizontal coordinates to another CRS.

    Only the ``x`` and ``y`` coordinates are transformed; heights are carried
    through unchanged. For a transform that also converts heights between
    vertical datums, use :func:`reproject_3d`.

    Args:
        cloud: The cloud to reproject. Its CRS must be set.
        target_crs: Target CRS as an EPSG code, WKT string, or ``pyproj.CRS``.

    Returns:
        A new :class:`PointCloud` in ``target_crs``, with attributes carried
        through and a provenance step appended.

    Raises:
        ValueError: If the cloud has no CRS, or a CRS cannot be interpreted.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, reproject
        >>> cloud = PointCloud(np.array([[155000.0, 463000.0, 0.0]]), crs=28992)
        >>> reproject(cloud, 4326).crs.to_epsg()
        4326
    """
    if cloud.crs is None:
        msg = "Cannot reproject a cloud with no CRS. Set the cloud's CRS first."
        raise ValueError(msg)
    source = coerce_crs(cloud.crs)
    target = coerce_crs(target_crs)
    transformer = pyproj.Transformer.from_crs(source, target, always_xy=True)

    x, y = transformer.transform(cloud.xyz[:, 0], cloud.xyz[:, 1])
    coordinates = np.column_stack([np.asarray(x), np.asarray(y), cloud.xyz[:, 2]])

    provenance = _provenance_for(cloud)
    provenance.add_step(
        "reproject (horizontal)",
        {"source_crs": source.to_string(), "target_crs": target.to_string()},
    )
    return PointCloud(
        coordinates,
        attributes=_attributes_of(cloud),
        crs=target,
        provenance=provenance,
    )


def reproject_3d(
    cloud: PointCloud,
    target_crs: object,
    *,
    allow_network: bool = False,
) -> PointCloud:
    """Reproject a cloud in three dimensions, including the vertical datum.

    Transforms ``x``, ``y``, and ``z`` together, so heights are converted
    between the source and target vertical datums (for example ellipsoidal to
    NAP orthometric) using the appropriate geoid grid. Both the source and
    target CRS must carry a vertical axis.

    Args:
        cloud: The cloud to reproject. Its CRS must be set and three-dimensional.
        target_crs: A 3D or compound target CRS (for example ``4979`` for
            WGS84 ellipsoidal, or ``7415`` for RD New + NAP orthometric).
        allow_network: If true, let pyproj download any missing transform grid
            from the PROJ CDN. If false and a grid is missing, an error is
            raised rather than returning silently wrong heights.

    Returns:
        A new :class:`PointCloud` in ``target_crs``, with attributes carried
        through and a provenance step appended.

    Raises:
        ValueError: If the cloud has no CRS, either CRS lacks a vertical axis,
            or the transform produced non-finite coordinates (usually a missing
            geoid grid).

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, reproject_3d
        >>> cloud = PointCloud(np.array([[5.0, 52.0, 43.0]]), crs=4979)
        >>> reproject_3d(cloud, 4937).crs.to_epsg()
        4937
    """
    if cloud.crs is None:
        msg = "Cannot reproject a cloud with no CRS. Set the cloud's CRS first."
        raise ValueError(msg)
    source = coerce_crs(cloud.crs)
    target = coerce_crs(target_crs)
    for role, crs_obj in (("source", source), ("target", target)):
        if not _has_vertical_axis(crs_obj):
            msg = (
                f"The {role} CRS {crs_obj.to_string()!r} has no vertical axis, "
                "so a 3D transform cannot convert heights. Use reproject() for "
                "a horizontal transform, or supply a 3D/compound CRS such as "
                "EPSG:4979 (ellipsoidal) or EPSG:7415 (RD New + NAP)."
            )
            raise ValueError(msg)

    if allow_network:
        # pyproj's type stubs do not export this runtime function.
        pyproj.network.set_network_enabled(active=True)  # type: ignore[attr-defined]
    transformer = pyproj.Transformer.from_crs(source, target, always_xy=True)

    x, y, z = transformer.transform(cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2])
    coordinates = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    if not bool(np.all(np.isfinite(coordinates))):
        msg = (
            "The 3D transform produced non-finite coordinates, usually because "
            "the geoid/datum grid it needs is not available offline. Retry with "
            "allow_network=True, or install the PROJ grid locally."
        )
        raise ValueError(msg)

    provenance = _provenance_for(cloud)
    provenance.add_step(
        "reproject (3D)",
        {"source_crs": source.to_string(), "target_crs": target.to_string()},
    )
    return PointCloud(
        coordinates,
        attributes=_attributes_of(cloud),
        crs=target,
        provenance=provenance,
    )
