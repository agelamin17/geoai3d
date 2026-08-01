# Benchmark: out-of-core geometric features on AHN

This measures that GEOAI_3D computes geometric features on a large public point
cloud within a stated memory ceiling (target: 16 GB). The dataset is the Dutch
national AHN LiDAR, which is **CC0** (public domain), so it is freely
redistributable and needs no login.

## 1. Get the data

Open the GeoTiles catalogue and pick an area:

- https://geotiles.citg.tudelft.nl/

Each map tile is split into 25 sub-tiles of 1 x 1.25 km with a 20 m overlap.
A single sub-tile LAZ is ~100-500 MB and ~10-15 million points (it fits in RAM,
good for viewing). Download the **AHN point cloud (LAZ)** link for one or more
sub-tiles. Its coordinate system is EPSG:7415 (RD New + NAP orthometric height).

## 2. View a tile (fits in memory)

```python
from geoai3d import read_las, view

cloud = read_las("sub_tile.laz")          # ~10-15M points, loads fine
view(cloud)                                # inline in Jupyter/Colab
# or, coloured by a geometric feature:
from geoai3d import geometric_features
feats = geometric_features(cloud, radius=1.0)
view(feats, color_by="planarity").show()   # roofs/roads bright, vegetation dark
```

`view` automatically thins the display to ~100k points, so it stays smooth.

## 3. (Optional) build a 100M+ point file

One sub-tile is ~12M points. To exceed 100M for the headline number, merge
several adjacent sub-tiles into one file (streamed, bounded memory):

```
python benchmarks/merge_laz.py merged.laz tileA.laz tileB.laz tileC.laz ... tileI.laz
```

Nine sub-tiles give ~100M+ points.

## 4. Run the benchmark

```
pip install psutil                    # for accurate peak-memory sampling
python benchmarks/benchmark.py merged.laz --radius 0.5 --tile-size 50 --workers 4 --verbose
```

It prints the point count, wall-clock time, throughput, **peak memory** (and a
PASS/OVER verdict against 16 GB), and with `--verbose`, the split between the
partition pass and the (parallel) feature pass. Copy those real numbers into
the paper / README — do not estimate them.

### Choosing radius and tile size

The cost is dominated by the number of neighbours each point has, which is
`density * pi * radius^2`. Pick a radius that captures ~20-30 neighbours:

- Check density from the benchmark output (points / area).
- AHN varies from ~10 to ~30 points/m^2. At ~28 points/m^2, `--radius 0.5`
  gives ~22 neighbours; at ~10 points/m^2, `--radius 0.8` is about right. A
  larger radius quadratically increases work and memory for little benefit.

`--tile-size 50` keeps each tile's working set small (bounded memory); a much
larger tile raises peak memory. `--workers` should match your core count: the
feature pass scales almost linearly with it.

## What "good" looks like

Streaming should peak far below what an in-memory copy would need (the script
prints that rough figure too). If peak memory grows with the *file* size rather
than staying near one tile's worth, something is holding the whole cloud
resident — that is the failure the seam/streaming design exists to prevent.
