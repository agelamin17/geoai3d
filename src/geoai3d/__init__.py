"""GEOAI_3D: geospatial-first AI workflows for 3D data.

This package is at a pre-alpha stage. No public API is stable yet; see the
roadmap in the README for what is planned and in what order.

The first building blocks are the columnar :class:`PointCloud` representation
and the :class:`Provenance` lineage record it carries.
"""

from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import ProcessStep, Provenance
from geoai3d.crs import reproject, reproject_3d
from geoai3d.io.las import read_las, to_las
from geoai3d.io.parquet import read_parquet, to_parquet
from geoai3d.io.xyz import read_xyz, to_xyz

__version__ = "0.0.1"

__all__ = [
    "PointCloud",
    "ProcessStep",
    "Provenance",
    "__version__",
    "read_las",
    "read_parquet",
    "read_xyz",
    "reproject",
    "reproject_3d",
    "to_las",
    "to_parquet",
    "to_xyz",
]
