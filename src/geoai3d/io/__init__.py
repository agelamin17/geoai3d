"""Reading and writing point clouds in geospatial file formats.

This subpackage wraps the file-format libraries (laspy for LAS/LAZ, pyarrow for
Parquet, and plain text for XYZ) behind small ``read_*`` and ``to_*`` functions
that return and accept the package's own
:class:`~geoai3d.core.pointcloud.PointCloud`, always carrying the coordinate
reference system through the round trip.
"""

from geoai3d.io.dispatch import read_lidar
from geoai3d.io.geotiff import read_geotiff, to_geotiff
from geoai3d.io.las import read_las, to_las
from geoai3d.io.parquet import read_parquet, to_parquet
from geoai3d.io.vector import to_geopackage
from geoai3d.io.xyz import read_xyz, to_xyz

__all__ = [
    "read_geotiff",
    "read_las",
    "read_lidar",
    "read_parquet",
    "read_xyz",
    "to_geopackage",
    "to_geotiff",
    "to_las",
    "to_parquet",
    "to_xyz",
]
