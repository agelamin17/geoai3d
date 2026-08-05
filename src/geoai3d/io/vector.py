"""Exporting point clouds as GIS vector layers.

Turning a classified or segmented cloud into a GeoPackage is the deliverable a
survey office or GIS analyst actually wants: points as geometry, the labels and
features as attribute columns, and the coordinate reference system attached so
the layer drops straight into QGIS or ArcGIS. :func:`to_geopackage` writes 3D
points (``PointZ``) so elevation survives, and, when the cloud carries a
provenance record, writes it to a sidecar JSON alongside the file.

Vector IO uses geopandas with the pyogrio engine and shapely, which ship as the
optional ``[gis]`` extra (``pip install geoai3d[gis]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from geoai3d.core.pointcloud import PointCloud

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

_GIS_HINT = (
    "Writing GIS vector layers needs geopandas, pyogrio, and shapely, which are "
    "not installed. Install the GIS extra with 'pip install geoai3d[gis]'."
)


def to_geopackage(
    cloud: PointCloud,
    destination: str | os.PathLike[str],
    *,
    attributes: Sequence[str] | None = None,
    layer: str = "points",
    include_z: bool = True,
) -> None:
    """Write a point cloud to a GeoPackage vector layer.

    Each point becomes a (3D) point feature; the chosen attributes become
    columns. The coordinate reference system is written into the file. If the
    cloud carries a provenance record, it is written to a sibling
    ``<destination>.provenance.json``.

    Args:
        cloud: The cloud to write. Its CRS must be set.
        destination: Output ``.gpkg`` path.
        attributes: Attribute columns to include. If omitted, every attribute is
            written.
        layer: Layer name inside the GeoPackage.
        include_z: If true, write 3D ``PointZ`` geometry carrying elevation; if
            false, write 2D points.

    Raises:
        ValueError: If geopandas (the ``[gis]`` extra) is not installed, or the
            cloud has no CRS.

    Example:
        >>> import numpy as np
        >>> import os, tempfile
        >>> from geoai3d import PointCloud, to_geopackage
        >>> cloud = PointCloud(
        ...     np.zeros((3, 3)),
        ...     attributes={"classification": np.array([2, 2, 6])},
        ...     crs=28992,
        ... )
        >>> to_geopackage(cloud, os.path.join(tempfile.mkdtemp(), "points.gpkg"))
    """
    try:
        import geopandas
        import shapely
    except ImportError as exc:
        raise ValueError(_GIS_HINT) from exc

    if cloud.crs is None:
        msg = (
            "Cannot write a GeoPackage without a coordinate reference system. "
            "Set the cloud's CRS first."
        )
        raise ValueError(msg)

    names = list(cloud.attribute_names) if attributes is None else list(attributes)
    missing = [name for name in names if name not in cloud.attribute_names]
    if missing:
        msg = (
            f"Attributes {missing} are not present. Available: {cloud.attribute_names}."
        )
        raise ValueError(msg)

    coordinates = cloud.xyz
    if include_z:
        geometry = shapely.points(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
    else:
        geometry = shapely.points(coordinates[:, 0], coordinates[:, 1])

    columns = {name: cloud.attribute(name) for name in names}
    frame = geopandas.GeoDataFrame(columns, geometry=geometry, crs=cloud.crs)
    frame.to_file(str(destination), driver="GPKG", layer=layer, engine="pyogrio")

    if cloud.provenance is not None:
        sidecar = Path(f"{destination}.provenance.json")
        sidecar.write_text(
            json.dumps(cloud.provenance.to_dict(), indent=2), encoding="utf-8"
        )
