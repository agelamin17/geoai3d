"""GEOAI_3D: geospatial-first AI workflows for 3D data.

This package is at a pre-alpha stage. No public API is stable yet; see the
roadmap in the README for what is planned and in what order.

The first building blocks are the columnar :class:`PointCloud` representation
and the :class:`Provenance` lineage record it carries.
"""

from geoai3d.classify import (
    Classifier,
    classify,
    feature_matrix,
    load_classifier,
    save_classifier,
    train_classifier,
)
from geoai3d.cluster import connected_components, dbscan, region_growing
from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import ProcessStep, Provenance
from geoai3d.core.raster import Raster
from geoai3d.crossval import spatial_block_split
from geoai3d.crs import reproject, reproject_3d
from geoai3d.dem import difference, to_dsm, to_dtm, volume
from geoai3d.features import geometric_features, multiscale_features
from geoai3d.geometry import estimate_normals, remove_statistical_outliers
from geoai3d.ground import ground
from geoai3d.index import build_kdtree
from geoai3d.intensity import normalize_intensity
from geoai3d.io.dispatch import read_lidar
from geoai3d.io.geotiff import read_geotiff, to_geotiff
from geoai3d.io.las import read_las, to_las
from geoai3d.io.parquet import read_parquet, to_parquet
from geoai3d.io.vector import to_geopackage
from geoai3d.io.xyz import read_xyz, to_xyz
from geoai3d.metrics import AccuracyReport, confusion_matrix, evaluate
from geoai3d.outofcore import estimate_radius, geometric_features_tiled
from geoai3d.primitives import Plane, fit_plane
from geoai3d.segment import segment
from geoai3d.stream import geometric_features_stream
from geoai3d.subsample import subsample
from geoai3d.viz import view

__version__ = "0.2.0"

__all__ = [
    "AccuracyReport",
    "Classifier",
    "Plane",
    "PointCloud",
    "ProcessStep",
    "Provenance",
    "Raster",
    "__version__",
    "build_kdtree",
    "classify",
    "confusion_matrix",
    "connected_components",
    "dbscan",
    "difference",
    "estimate_normals",
    "estimate_radius",
    "evaluate",
    "feature_matrix",
    "fit_plane",
    "geometric_features",
    "geometric_features_stream",
    "geometric_features_tiled",
    "ground",
    "load_classifier",
    "multiscale_features",
    "normalize_intensity",
    "read_geotiff",
    "read_las",
    "read_lidar",
    "read_parquet",
    "read_xyz",
    "region_growing",
    "remove_statistical_outliers",
    "reproject",
    "reproject_3d",
    "save_classifier",
    "segment",
    "spatial_block_split",
    "subsample",
    "to_dsm",
    "to_dtm",
    "to_geopackage",
    "to_geotiff",
    "to_las",
    "to_parquet",
    "to_xyz",
    "train_classifier",
    "view",
    "volume",
]
