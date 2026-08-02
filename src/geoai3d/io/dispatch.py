"""Choosing a point-cloud reader from a file's extension.

:func:`read_lidar` is the high-level entry point for loading LiDAR and
point-cloud data: it looks at the file extension and calls the matching
format-specific reader. The format-specific readers
(:func:`~geoai3d.read_las`, :func:`~geoai3d.read_xyz`) stay public for when you
already know the format or need format-specific options such as a custom
delimiter or column names.

Parquet is GEOAI_3D's own at-rest serialization format rather than an
acquisition format, so it is read with :func:`~geoai3d.read_parquet` and is
deliberately not routed through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from geoai3d.io.las import read_las
from geoai3d.io.xyz import read_xyz

if TYPE_CHECKING:
    import os

    from geoai3d.core.pointcloud import PointCloud

# Suffixes routed to each format-specific reader. Text suffixes are the
# whitespace-delimited layouts read_xyz handles with its defaults; delimited or
# custom-column text should call read_xyz directly.
_LAS_SUFFIXES = (".las", ".laz")
_TEXT_SUFFIXES = (".xyz", ".txt", ".asc", ".pts")


def read_lidar(
    source: str | os.PathLike[str],
    *,
    crs: object | None = None,
) -> PointCloud:
    """Read a LiDAR or point-cloud file, choosing the reader by extension.

    Dispatches on the file suffix: ``.las`` and ``.laz`` are read with
    :func:`~geoai3d.read_las`, and whitespace-delimited text
    (``.xyz``, ``.txt``, ``.asc``, ``.pts``) with :func:`~geoai3d.read_xyz`
    using its defaults. For delimited text, non-default column layouts, or to
    force a particular format, call the format-specific reader directly.

    Args:
        source: Path to the point-cloud file.
        crs: Coordinate reference system to assign, overriding any the file
            carries. An EPSG code, WKT/PROJ string, or ``pyproj.CRS``. Required
            when the file (and, for text, its sibling ``.prj``) has none.

    Returns:
        A :class:`~geoai3d.PointCloud` with the resolved CRS and a provenance
        record noting the read.

    Raises:
        ValueError: If the extension is ``.parquet`` (read it with
            :func:`~geoai3d.read_parquet`), or is not a recognised
            point-cloud format.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, read_lidar, to_las
        >>> cloud = PointCloud(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), crs=28992)
        >>> path = os.path.join(tempfile.mkdtemp(), "scan.las")
        >>> to_las(cloud, path)
        >>> read_lidar(path).crs.to_epsg()
        28992
    """
    suffix = Path(source).suffix.lower()
    if suffix in _LAS_SUFFIXES:
        return read_las(source, crs=crs)
    if suffix in _TEXT_SUFFIXES:
        return read_xyz(source, crs=crs)
    if suffix == ".parquet":
        msg = (
            "Parquet is GEOAI_3D's serialization format, not an acquisition "
            "format; read it with read_parquet() rather than read_lidar()."
        )
        raise ValueError(msg)
    supported = ", ".join(_LAS_SUFFIXES + _TEXT_SUFFIXES)
    msg = (
        f"Unsupported point-cloud format {suffix!r} for {str(source)!r}. "
        f"read_lidar handles: {supported}. For Parquet use read_parquet()."
    )
    raise ValueError(msg)
