"""Reading and writing LAS/LAZ point clouds with laspy.

``read_las`` loads a LAS or LAZ file into a ``PointCloud``, keeping every
per-point dimension (standard fields and user-defined extra dimensions) and the
coordinate reference system from the file header. ``to_las`` writes a cloud back
out, reconstructing the header, the extra dimensions, and the CRS record.

Coordinates are carried as scaled ``float64`` values. On write they are
re-quantised to integers using the file's scales and offsets, so coordinates
round-trip to the file's precision (a millimetre by default) while all other
attributes round-trip exactly.

Reading and writing compressed ``.laz`` requires the optional ``lazrs`` backend
(``pip install geoai3d[laz]``); uncompressed ``.las`` needs nothing extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import laspy
import numpy as np
import pyproj
from numpy.typing import NDArray

from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import Provenance

if TYPE_CHECKING:
    import os

# The raw integer coordinate fields. These are represented by the cloud's
# scaled float64 ``xyz`` array, so they are not stored again as attributes.
_COORDINATE_DIMENSIONS = ("X", "Y", "Z")

_DEFAULT_POINT_FORMAT = 6
_DEFAULT_FILE_VERSION = "1.4"
_DEFAULT_SCALE = 0.001


def _coerce_crs(crs_input: object) -> pyproj.CRS:
    """Convert a user CRS input into a :class:`pyproj.CRS`.

    Args:
        crs_input: An EPSG code, WKT/PROJ string, or existing ``pyproj.CRS``.

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


def read_las(
    source: str | os.PathLike[str],
    *,
    crs: object | None = None,
) -> PointCloud:
    """Read a LAS or LAZ file into a :class:`PointCloud`.

    Args:
        source: Path to a ``.las`` or ``.laz`` file.
        crs: Coordinate reference system to assign, overriding whatever the
            file declares. If the file has no CRS and none is given here, an
            error is raised rather than returning an unreferenced cloud.

    Returns:
        A :class:`PointCloud` with scaled ``float64`` coordinates, every
        non-coordinate dimension as an attribute, the resolved CRS, and a
        provenance record noting the read.

    Raises:
        ValueError: If the file has no CRS and ``crs`` is not supplied, or if a
            supplied ``crs`` cannot be interpreted.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, read_las, to_las
        >>> cloud = PointCloud(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), crs=28992)
        >>> path = os.path.join(tempfile.mkdtemp(), "example.las")
        >>> to_las(cloud, path)
        >>> read_las(path).crs.to_epsg()
        28992
    """
    las = laspy.read(str(source))
    header = las.header
    coordinates: NDArray[np.float64] = np.column_stack(
        (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
    ).astype(np.float64, copy=False)

    file_crs = header.parse_crs()
    if crs is not None:
        resolved_crs: object = _coerce_crs(crs)
    elif file_crs is not None:
        resolved_crs = file_crs
    else:
        msg = (
            f"{str(source)!r} has no coordinate reference system and none was "
            "supplied. Pass crs= (an EPSG code, WKT string, or pyproj.CRS) to "
            "set it explicitly."
        )
        raise ValueError(msg)

    attributes: dict[str, NDArray[Any]] = {}
    for name in list(las.point_format.dimension_names):
        if name in _COORDINATE_DIMENSIONS:
            continue
        attributes[name] = np.asarray(las[name]).copy()

    provenance = Provenance(source=str(source))
    provenance.add_step(
        "read LAS/LAZ",
        {
            "path": str(source),
            "point_format": int(las.point_format.id),
            "version": str(header.version),
            "point_count": int(header.point_count),
        },
    )
    return PointCloud(
        coordinates,
        attributes=attributes,
        crs=resolved_crs,
        provenance=provenance,
    )


def to_las(
    cloud: PointCloud,
    destination: str | os.PathLike[str],
    *,
    point_format: int = _DEFAULT_POINT_FORMAT,
    file_version: str = _DEFAULT_FILE_VERSION,
    scales: NDArray[np.float64] | Sequence[float] | None = None,
    offsets: NDArray[np.float64] | Sequence[float] | None = None,
) -> None:
    """Write a :class:`PointCloud` to a LAS or LAZ file.

    Attributes whose names are standard dimensions of ``point_format`` are
    written as those fields; any other attribute is written as a user-defined
    extra dimension, preserving its values and dtype.

    Args:
        cloud: The cloud to write. Its CRS must be set.
        destination: Output path. A ``.laz`` suffix triggers compression and
            requires the ``lazrs`` backend (``pip install geoai3d[laz]``).
        point_format: LAS point format id to write. The default, 6, carries
            coordinates, intensity, returns, classification, scan angle,
            GPS time, and point source id.
        file_version: LAS version string. The default, ``"1.4"``, stores the
            CRS as WKT, which supports arbitrary coordinate systems.
        scales: Per-axis coordinate scales. Defaults to one millimetre on each
            axis.
        offsets: Per-axis coordinate offsets. Defaults to the minimum
            coordinate on each axis.

    Raises:
        ValueError: If the cloud has no CRS, or the CRS cannot be interpreted.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_las
        >>> cloud = PointCloud(np.zeros((5, 3)), crs=28992)
        >>> to_las(cloud, os.path.join(tempfile.mkdtemp(), "out.laz"))
    """
    if cloud.crs is None:
        msg = (
            "Cannot write a LAS/LAZ file without a coordinate reference "
            "system. Set the cloud's CRS first, for example by reading it "
            "with read_las(..., crs=...)."
        )
        raise ValueError(msg)
    resolved_crs = _coerce_crs(cloud.crs)

    coordinates = cloud.xyz
    if offsets is None:
        offsets_array = (
            np.min(coordinates, axis=0) if len(cloud) else np.zeros(3, dtype=np.float64)
        )
    else:
        offsets_array = np.asarray(offsets, dtype=np.float64)
    scales_array = (
        np.full(3, _DEFAULT_SCALE)
        if scales is None
        else np.asarray(scales, dtype=np.float64)
    )

    header = laspy.LasHeader(version=file_version, point_format=point_format)
    header.offsets = offsets_array
    header.scales = scales_array

    standard_names = set(header.point_format.standard_dimension_names)
    for name in cloud.attribute_names:
        if name in standard_names or name in _COORDINATE_DIMENSIONS:
            continue
        values = cloud.attribute(name)
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=values.dtype))

    header.add_crs(resolved_crs)

    las = laspy.LasData(header)
    # Assigning the coordinates first sizes the point record; other
    # dimensions are then assigned into the allocated record.
    las.x = coordinates[:, 0]
    las.y = coordinates[:, 1]
    las.z = coordinates[:, 2]
    for name in cloud.attribute_names:
        if name in _COORDINATE_DIMENSIONS:
            continue
        las[name] = cloud.attribute(name)

    las.write(str(destination))
