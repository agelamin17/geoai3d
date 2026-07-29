"""Reading and writing plain XYZ / CSV text point clouds.

XYZ is a simple whitespace- or comma-delimited text format: one point per line,
coordinates first and optional extra numeric columns after. The format cannot
store a coordinate reference system, so GEOAI_3D writes the CRS to a sibling
``.prj`` file (the GIS convention) and reads it back from there when present.

Values are written as text and read back as ``float64``, so this is a lossy
exchange format: use Parquet or LAS to preserve exact dtypes and full metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.core._crs import coerce_crs
from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import Provenance

if TYPE_CHECKING:
    import os

_COORDINATE_NAMES = ("x", "y", "z")


def _prj_path(path: str | os.PathLike[str]) -> Path:
    """Return the sibling ``.prj`` path for a data file path."""
    return Path(path).with_suffix(".prj")


def _resolve_crs(source: str | os.PathLike[str], crs: object | None) -> object:
    """Resolve the CRS from an explicit argument or a sibling ``.prj`` file."""
    if crs is not None:
        return coerce_crs(crs)
    prj = _prj_path(source)
    if prj.exists():
        return coerce_crs(prj.read_text(encoding="utf-8"))
    msg = (
        f"{str(source)!r} has no CRS: XYZ files do not store one and no "
        f"sibling {prj.name!r} was found. Pass crs= to set it explicitly."
    )
    raise ValueError(msg)


def read_xyz(
    source: str | os.PathLike[str],
    *,
    crs: object | None = None,
    columns: tuple[str, ...] = _COORDINATE_NAMES,
    delimiter: str | None = None,
    comments: str = "#",
) -> PointCloud:
    """Read a whitespace- or delimiter-separated XYZ text file.

    Args:
        source: Path to the text file.
        crs: Coordinate reference system to assign. If omitted, a sibling
            ``.prj`` file is used; if neither is available, an error is raised.
        columns: Name of each column in order. Must include ``x``, ``y``, and
            ``z``; any other names become attributes.
        delimiter: Column delimiter. ``None`` (default) splits on any run of
            whitespace.
        comments: Character marking comment lines to skip.

    Returns:
        A :class:`PointCloud` with the resolved CRS and a provenance record.

    Raises:
        ValueError: If ``columns`` omits a coordinate axis, the column count
            does not match, or no CRS can be resolved.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, read_xyz, to_xyz
        >>> path = os.path.join(tempfile.mkdtemp(), "points.xyz")
        >>> to_xyz(PointCloud(np.zeros((3, 3)), crs=28992), path)
        >>> len(read_xyz(path))
        3
    """
    names = tuple(columns)
    for axis in _COORDINATE_NAMES:
        if axis not in names:
            msg = f"columns must include {axis!r}; got {names}."
            raise ValueError(msg)
    data: NDArray[np.float64] = np.loadtxt(
        str(source), delimiter=delimiter, comments=comments, dtype=np.float64, ndmin=2
    )
    if data.shape[1] != len(names):
        msg = (
            f"{str(source)!r} has {data.shape[1]} columns but {len(names)} "
            f"column names were given: {names}."
        )
        raise ValueError(msg)

    resolved_crs = _resolve_crs(source, crs)
    by_name = {name: data[:, index] for index, name in enumerate(names)}
    coordinates = np.column_stack([by_name["x"], by_name["y"], by_name["z"]])
    attributes: dict[str, NDArray[Any]] = {
        name: by_name[name] for name in names if name not in _COORDINATE_NAMES
    }

    provenance = Provenance(source=str(source))
    provenance.add_step("read XYZ", {"path": str(source), "columns": list(names)})
    return PointCloud(
        coordinates,
        attributes=attributes,
        crs=resolved_crs,
        provenance=provenance,
    )


def to_xyz(
    cloud: PointCloud,
    destination: str | os.PathLike[str],
    *,
    include_attributes: bool = True,
    delimiter: str = " ",
    fmt: str = "%.6f",
    write_header: bool = False,
    write_prj: bool = True,
) -> None:
    """Write a :class:`PointCloud` to an XYZ text file.

    Args:
        cloud: The cloud to write.
        destination: Output path.
        include_attributes: If true, write attribute columns after ``x y z``.
        delimiter: Column delimiter.
        fmt: ``numpy.savetxt`` format applied to every column.
        write_header: If true, write a commented header line of column names.
        write_prj: If true and the cloud has a CRS, write it as WKT to a
            sibling ``.prj`` file so ``read_xyz`` can recover it.

    Example:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from geoai3d import PointCloud, to_xyz
        >>> cloud = PointCloud(np.zeros((2, 3)), crs=28992)
        >>> to_xyz(cloud, os.path.join(tempfile.mkdtemp(), "out.xyz"))
    """
    coordinates = cloud.xyz
    arrays: list[NDArray[Any]] = [
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
    ]
    names = ["x", "y", "z"]
    if include_attributes:
        for name in cloud.attribute_names:
            arrays.append(np.asarray(cloud.attribute(name), dtype=np.float64))
            names.append(name)

    table = np.column_stack(arrays)
    header = delimiter.join(names) if write_header else ""
    np.savetxt(
        str(destination),
        table,
        delimiter=delimiter,
        fmt=fmt,
        header=header,
        comments="# ",
    )
    if write_prj and cloud.crs is not None:
        _prj_path(destination).write_text(
            coerce_crs(cloud.crs).to_wkt(), encoding="utf-8"
        )
