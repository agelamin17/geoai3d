# GEOAI_3D

**Geospatial-first AI workflows for 3D data.** LiDAR point clouds, with
coordinate reference systems, vertical datums, provenance, and GIS export
treated as first-class concerns rather than afterthoughts.

Most 3D tooling forgets your data has a coordinate system. GEOAI_3D keeps a CRS
and a provenance record attached to every cloud and every output, handles the
ellipsoidal-versus-orthometric height distinction explicitly, and processes
clouds larger than memory with a tested tile-seam guarantee — then closes the
loop to a classified GeoPackage a GIS analyst can open.

## Install

```bash
pip install geoai3d
```

The base install is CPU-only and needs no compiler. Optional extras add
compressed-LAZ IO, the 3D viewer, and GIS raster/vector export:

```bash
pip install "geoai3d[laz,viz,gis]"
```

## A first workflow

Read a tile, describe its geometry, decompose the scene without any labels, and
write a GIS layer — a handful of lines, all on a CPU:

```python
import geoai3d as g3d

cloud = g3d.read_lidar("scan.laz")            # CRS read from the file header
cloud = g3d.geometric_features(cloud, radius=1.0)
segments = g3d.segment(cloud)                 # ground + objects, unsupervised
g3d.to_geopackage(segments, "segments.gpkg")  # CRS + lineage travel with it
```

See [Getting started](getting-started.md) for the supervised classification
workflow, and the [Tutorials](tutorials.md) for runnable notebooks on real data.

## Where it is

GEOAI_3D is built in stages, each shipping something installable and useful on
its own. The current release delivers the out-of-core georeferenced foundation,
frugal (label-free) segmentation, classical classification with honest
spatially-blocked evaluation, and GIS raster/vector export. Deep-learning models
and metric Gaussian splatting come later. See the
[roadmap](https://github.com/agelamin17/geoai3d#roadmap).

## Getting involved

The project is developed in the open and welcomes early involvement — especially
from anyone holding point-cloud data with independently surveyed control. See
[Contributing](contributing.md).
