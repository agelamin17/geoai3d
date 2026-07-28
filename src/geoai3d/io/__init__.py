"""Reading and writing point clouds in geospatial file formats.

This subpackage wraps the file-format libraries (laspy for LAS/LAZ, and more to
come) behind small ``read_*`` and ``to_*`` functions that return and accept the
package's own :class:`~geoai3d.core.pointcloud.PointCloud`, always carrying the
coordinate reference system through the round trip.
"""

from geoai3d.io.las import read_las, to_las

__all__ = ["read_las", "to_las"]
