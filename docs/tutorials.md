# Tutorials

Four notebooks work through the Stage-2 workflow end to end on a small, real
LiDAR sample — a ~150 m crop of the Dutch national dataset AHN4 (public domain,
CC0), included in the repository under `examples/data/`. Everything runs on a
CPU; no GPU and no training data are required.

Install the feature extras first:

```bash
pip install "geoai3d[laz,viz,gis]"
```

The notebooks live in
[`examples/notebooks/`](https://github.com/agelamin17/geoai3d/tree/main/examples/notebooks)
and render directly on GitHub:

1. **[Getting started](https://github.com/agelamin17/geoai3d/blob/main/examples/notebooks/01_getting_started.ipynb)**
   — read a tile, inspect its CRS and classes, reproject, compute geometric
   features, and view the cloud in 3D.
2. **[Terrain and volumes](https://github.com/agelamin17/geoai3d/blob/main/examples/notebooks/02_terrain_and_volumes.ipynb)**
   — filter the ground, build a DTM and DSM, difference them for object heights,
   compute a volume, and export GeoTIFFs.
3. **[Unsupervised segmentation](https://github.com/agelamin17/geoai3d/blob/main/examples/notebooks/03_unsupervised_segmentation.ipynb)**
   — decompose the scene into ground and objects with no labels.
4. **[Classification and GIS export](https://github.com/agelamin17/geoai3d/blob/main/examples/notebooks/04_classification_and_gis_export.ipynb)**
   — engineer features, train a random forest, evaluate it with
   spatially-blocked cross-validation, and export a classified GeoPackage.

To run them without Jupyter, the same pipelines are collected in a plain script
you can execute directly:

```bash
python examples/verify_notebooks.py
```
