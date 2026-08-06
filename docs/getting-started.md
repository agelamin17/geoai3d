# Getting started

## Installation

The base install is CPU-only and needs no compiler:

```bash
pip install geoai3d
```

Optional extras, added in brackets:

| Extra   | Adds                                                        |
|---------|-------------------------------------------------------------|
| `laz`   | Reading and writing compressed `.laz` files                 |
| `viz`   | The interactive Jupyter/Colab 3D viewer (`view`)            |
| `gis`   | Raster (GeoTIFF) and vector (GeoPackage) export via GDAL    |
| `dev`   | Test, lint, and type-check tooling                          |

For the tutorials, install all three feature extras:

```bash
pip install "geoai3d[laz,viz,gis]"
```

## Core ideas

- **Everything is georeferenced.** Reading a file recovers its CRS; every
  operation carries it through; writing an output stores it. A cloud without a
  CRS raises rather than guessing one.
- **Everything is provenanced.** Each operation appends a step to a lineage
  record, so an output knows what produced it, from which input, with which
  parameters.
- **The common case is a few lines.** High-level functions have sensible
  defaults; drop to the underlying library only when you need to.

## The frugal workflow (no labels)

Decompose a scene into ground and objects with no training data and no GPU:

```python
import geoai3d as g3d

cloud = g3d.read_lidar("scan.laz")
segments = g3d.segment(cloud)                  # cloth-filter ground, then grow regions
g3d.to_geopackage(segments, "segments.gpkg")
```

## The supervised workflow (with labels)

Train a random forest on per-point features and measure it honestly with
spatially-blocked cross-validation — which, unlike a random split, does not let
neighbouring points leak between training and test:

```python
import numpy as np
import geoai3d as g3d

cloud = g3d.read_lidar("labelled_scan.laz")
features = g3d.geometric_features(cloud, radius=1.0)

truths, predictions = [], []
for train, test in g3d.spatial_block_split(features, block_size=25.0, n_folds=5):
    model = g3d.train_classifier(features[train], label_attribute="classification")
    predicted = g3d.classify(features[test], model=model)
    truths.append(features[test].attribute("classification"))
    predictions.append(predicted.attribute("prediction"))

report = g3d.evaluate(np.concatenate(truths), np.concatenate(predictions))
print("overall accuracy:", report.overall_accuracy)
print("mean IoU:", report.mean_iou)
```

Then train on everything and export the classified layer:

```python
model = g3d.train_classifier(features, label_attribute="classification")
result = g3d.classify(features, model=model)
g3d.to_geopackage(result, "classified.gpkg")   # writes a provenance sidecar too
```

## Terrain products

```python
classified = g3d.ground(cloud)                 # adds an is_ground column
dtm = g3d.to_dtm(classified, resolution=0.5)   # bare earth
dsm = g3d.to_dsm(classified, resolution=0.5)   # top surface
heights = g3d.difference(dsm, dtm)             # object heights (nDSM)
g3d.to_geotiff(heights, "object_heights.tif")
print("above-ground volume:", g3d.volume(heights)["fill"])
```

Continue to the [Tutorials](tutorials.md) for these workflows as runnable
notebooks on a real LiDAR sample, or the [API reference](api.md) for every
function.
