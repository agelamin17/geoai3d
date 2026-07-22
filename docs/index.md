# GEOAI_3D

Geospatial-first AI workflows for 3D data.

!!! warning "Pre-alpha"
    Version 0.0.1 is a project skeleton. It installs and does nothing. This
    documentation site exists from the first commit so that it grows alongside
    the code rather than being assembled at the end.

## Installation

```bash
pip install geoai3d
```

Not yet published. Install from source in the meantime:

```bash
git clone https://github.com/agelamin17/geoai3d.git
cd geoai3d
pip install -e ".[dev]"
```

## What is planned

Stage 1 delivers the foundation: out-of-core reading and processing of Lidar point
clouds beyond available memory, spatial indexing, explicit handling of horizontal
and vertical coordinate reference systems, multi-scale geometric feature
computation with a tested tile-seam guarantee, and provenance recording on every
output.

Later stages add unsupervised geometric segmentation, GIS vector export, a QGIS
plugin, supervised semantic segmentation with pretrained models for geospatial
classes, and metric georeferenced Gaussian splatting.

See the [roadmap in the README](https://github.com/agelamin17/geoai3d#roadmap).

## Getting involved

The project is developed in the open and welcomes early involvement, particularly
from anyone holding point cloud data with independently surveyed control.
