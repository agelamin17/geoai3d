"""GEOAI_3D: geospatial-first AI workflows for 3D data.

This package is at a pre-alpha stage. No public API is stable yet; see the
roadmap in the README for what is planned and in what order.

The first building blocks are the columnar :class:`PointCloud` representation
and the :class:`Provenance` lineage record it carries.
"""

from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import ProcessStep, Provenance
from geoai3d.crs import reproject, reproject_3d
from geoai3d.features import geometric_features, multiscale_features
from geoai3d.geometry import estimate_normals, remove_statistical_outliers
from geoai3d.index import build_kdtree
from geoai3d.intensity import normalize_intensity
from geoai3d.io.dispatch import read_lidar
from geoai3d.io.las import read_las, to_las
from geoai3d.io.parquet import read_parquet, to_parquet
from geoai3d.io.xyz import read_xyz, to_xyz
from geoai3d.outofcore import estimate_radius, geometric_features_tiled
from geoai3d.stream import geometric_features_stream
from geoai3d.subsample import subsample
from geoai3d.viz import view

__version__ = "0.1.0"

__all__ = [
    "PointCloud",
    "ProcessStep",
    "Provenance",
    "__version__",
    "build_kdtree",
    "estimate_normals",
    "estimate_radius",
    "geometric_features",
    "geometric_features_stream",
    "geometric_features_tiled",
    "multiscale_features",
    "normalize_intensity",
    "read_las",
    "read_lidar",
    "read_parquet",
    "read_xyz",
    "remove_statistical_outliers",
    "reproject",
    "reproject_3d",
    "subsample",
    "to_las",
    "to_parquet",
    "to_xyz",
    "view",
]
