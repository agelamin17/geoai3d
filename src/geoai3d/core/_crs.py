"""Internal helpers for turning user CRS inputs into pyproj objects.

Private module: not part of the public API. Shared by the IO and reprojection
modules so the coercion rule and its error message live in one place.
"""

from __future__ import annotations

import pyproj


def coerce_crs(crs_input: object) -> pyproj.CRS:
    """Convert a user CRS input into a :class:`pyproj.CRS`.

    Args:
        crs_input: An EPSG code, a WKT/PROJ string, or an existing
            ``pyproj.CRS``.

    Returns:
        The resolved coordinate reference system.

    Raises:
        ValueError: If the input cannot be interpreted as a CRS.
    """
    try:
        return pyproj.CRS.from_user_input(crs_input)
    except pyproj.exceptions.CRSError as exc:
        msg = (
            f"Could not interpret {crs_input!r} as a coordinate reference "
            f"system. Pass an EPSG code, a WKT string, or a pyproj.CRS. ({exc})"
        )
        raise ValueError(msg) from exc
