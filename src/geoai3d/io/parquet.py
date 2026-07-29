"""Reading and writing point clouds as Parquet, the package's at-rest format.

Where LAS/LAZ is the interchange format the wider world uses, Parquet is the
columnar format GEOAI_3D spills to and reads back for its own out-of-core work.
Each column (``x``, ``y``, ``z``, and every attribute) is stored losslessly, and
the coordinate reference system and provenance record are written into the
file's schema metadata so they survive the round trip.

This is plain Parquet with a CRS in the metadata, not the full GeoParquet
geometry encoding; per-point clouds do not benefit from per-row WKB geometry.
The metadata keys are namespaced under ``geoai3d:`` to avoid clashes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
from numpy.typing import NDArray

from geoai3d.core._crs import coerce_crs
from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import Provenance

if TYPE_CHECKING:
    import os

_COORDINATE_COLUMNS = ("x", "y", "z")
_CRS_KEY = b"geoai3d:crs"
_PROVENANCE_KEY = b"geoai3d:provenance"
_SCHEMA_VERSION_KEY = b"geoai3d:schema_version"
_SCHEMA_VERSION = b"1"


def to_parquet(cloud: PointCloud, destination: str | os.PathLike[str]) -> None:
    """Write a :class:`PointCloud` to a Parquet file.

    Coordinates are stored as ``x``, ``y``, ``z`` columns and each attribute as
    a column of its own dtype. The CRS and provenance record are written into
    the schema metadata.

    Args:
        cloud: The cloud to write. Its CRS must be set.
        destination: Output ``.parquet`` path.

    Raises:
        ValueError: If the cloud has no CRS, or an attribute is named ``x``,
            ``y``, or ``z`` (reserved for coordinates).

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_parquet, read_parquet
        >>> cloud = PointCloud(np.zeros((4, 3)), crs=28992)
        >>> path = os.path.join(tempfile.mkdtemp(), "cloud.parquet")
        >>> to_parquet(cloud, path)
        >>> len(read_parquet(path))
        4
    """
    if cloud.crs is None:
        msg = (
            "Cannot write a Parquet file without a coordinate reference "
            "system. Set the cloud's CRS first."
        )
        raise ValueError(msg)
    reserved = set(_COORDINATE_COLUMNS) & set(cloud.attribute_names)
    if reserved:
        msg = (
            f"Attributes {sorted(reserved)} clash with the reserved coordinate "
            "column names 'x', 'y', 'z'. Rename them before writing."
        )
        raise ValueError(msg)

    coordinates = cloud.xyz
    columns: dict[str, NDArray[Any]] = {
        "x": np.ascontiguousarray(coordinates[:, 0]),
        "y": np.ascontiguousarray(coordinates[:, 1]),
        "z": np.ascontiguousarray(coordinates[:, 2]),
    }
    for name in cloud.attribute_names:
        columns[name] = cloud.attribute(name)

    metadata: dict[bytes, bytes] = {
        _SCHEMA_VERSION_KEY: _SCHEMA_VERSION,
        _CRS_KEY: coerce_crs(cloud.crs).to_wkt().encode("utf-8"),
    }
    if cloud.provenance is not None:
        metadata[_PROVENANCE_KEY] = json.dumps(cloud.provenance.to_dict()).encode(
            "utf-8"
        )

    table = pa.table(columns).replace_schema_metadata(metadata)
    pq.write_table(table, str(destination))


def read_parquet(
    source: str | os.PathLike[str],
    *,
    crs: object | None = None,
) -> PointCloud:
    """Read a Parquet file written by :func:`to_parquet` into a cloud.

    Args:
        source: Path to a ``.parquet`` file with ``x``, ``y``, ``z`` columns.
        crs: Coordinate reference system to assign, overriding the file's
            metadata. If the file carries no CRS and none is given here, an
            error is raised rather than returning an unreferenced cloud.

    Returns:
        A :class:`PointCloud` with the coordinates, every other column as an
        attribute, the resolved CRS, and the stored provenance record.

    Raises:
        ValueError: If the file lacks ``x``/``y``/``z`` columns, or has no CRS
            and ``crs`` is not supplied.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_parquet, read_parquet
        >>> path = os.path.join(tempfile.mkdtemp(), "cloud.parquet")
        >>> to_parquet(PointCloud(np.ones((2, 3)), crs=28992), path)
        >>> read_parquet(path).crs.to_epsg()
        28992
    """
    table = pq.read_table(str(source))
    missing = [name for name in _COORDINATE_COLUMNS if name not in table.column_names]
    if missing:
        msg = (
            f"{str(source)!r} is missing coordinate columns {missing}; it does "
            "not look like a geoai3d Parquet file."
        )
        raise ValueError(msg)

    coordinates = np.column_stack(
        [np.asarray(table.column(name)) for name in _COORDINATE_COLUMNS]
    ).astype(np.float64, copy=False)
    attributes: dict[str, NDArray[Any]] = {
        name: np.asarray(table.column(name))
        for name in table.column_names
        if name not in _COORDINATE_COLUMNS
    }

    metadata = table.schema.metadata or {}
    if crs is not None:
        resolved_crs: object = coerce_crs(crs)
    elif _CRS_KEY in metadata:
        resolved_crs = pyproj.CRS.from_wkt(metadata[_CRS_KEY].decode("utf-8"))
    else:
        msg = (
            f"{str(source)!r} has no coordinate reference system in its "
            "metadata and none was supplied. Pass crs= to set it explicitly."
        )
        raise ValueError(msg)

    provenance = None
    if _PROVENANCE_KEY in metadata:
        provenance = Provenance.from_dict(
            json.loads(metadata[_PROVENANCE_KEY].decode("utf-8"))
        )

    return PointCloud(
        coordinates,
        attributes=attributes,
        crs=resolved_crs,
        provenance=provenance,
    )
