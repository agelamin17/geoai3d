# GEOAI_3D

[![CI](https://github.com/agelamin17/geoai3d/actions/workflows/ci.yml/badge.svg)](https://github.com/agelamin17/geoai3d/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

**Geospatial-first AI workflows for 3D data.** Lidar point clouds, photogrammetry,
and Gaussian splatting, with coordinate reference systems, vertical datums, and
accuracy reporting treated as first-class concerns rather than afterthoughts.

> **Status: pre-alpha (v0.1.0).** The out-of-core georeferenced foundation is here:
> read and write LAS/LAZ, Parquet, and XYZ with coordinate-system and vertical-datum
> handling, geometric feature descriptors, and a streaming engine that computed
> features on a **722,964,406-point** LiDAR tile — which would need ~21.5 GB to hold
> in memory — within **8.9 GB of peak memory** on 4 CPU cores, bit-for-bit identical
> to the whole-cloud result. APIs may still change before 1.0. See the roadmap below
> and [CONTRIBUTING.md](CONTRIBUTING.md) if you want to be involved early.

## Why this exists

Working with 3D geospatial data today means assembling a pipeline by hand from
low-level libraries — PDAL or laspy for IO, Open3D for structures, COLMAP for
Structure-from-Motion, gsplat for splatting — each excellent, none aware of the
others, and none aware that your data has a coordinate system.

Three problems recur, and none of them are solved by any existing package:

**Scale.** A 200-million-point survey does not fit in memory. Chunked processing is
straightforward until you need a geometric feature computed near a tile boundary to
match the value it would have had on the unchunked cloud. GEOAI_3D treats that
seam contract as a tested guarantee, not an implementation detail.

**Vertical datums.** Most 3D tooling reduces "CRS handling" to horizontal
reprojection. Mixing GNSS ellipsoidal heights with an orthometric product silently
introduces errors of tens of metres. GEOAI_3D refuses to guess: geoid separation is
handled explicitly, and data without a declared vertical datum raises rather than
defaults.

**Accuracy.** Geomatics is the discipline that quantifies uncertainty; most 3D AI
tooling treats a coordinate as a fact. Per-point uncertainty, registration
covariance, and propagation into derived products are intended to be properties of
the data, not a separate analysis.

## Design commitments

These are constraints on the project, not aspirations:

- **Base install works on a CPU-only machine with no compiler.** Anything requiring
  CUDA or a build toolchain lives in an optional extra, and CI verifies the bare
  install in a bare container.
- **Cross-platform.** Linux, macOS, and Windows are tested on every commit, across
  Python 3.10 to 3.13. Colab-friendly throughout.
- **Spatial reference is never optional.** Any function returning spatial data
  returns it with a CRS attached. Missing georeferencing is an error naming the
  parameter that would fix it, never a silent default.
- **Build on the ecosystem, own the middle.** PDAL, laspy, Open3D, COLMAP, and
  gsplat are dependencies, not things to reimplement. What GEOAI_3D provides is the
  out-of-core execution model, the datum and CRS layer, uncertainty propagation, and
  provenance that sit between them.
- **Everything is provenanced.** Outputs record what produced them, from which
  input, with which parameters and versions.

## Installation

```bash
pip install geoai3d
```

Requires Python 3.10 or newer. The base install is CPU-only and needs no compiler.

Optional extras:

```bash
pip install "geoai3d[laz]"        # read/write compressed LAZ (lazrs backend)
pip install "geoai3d[viz]"        # interactive 3D viewer for Jupyter/Colab (plotly)
pip install "geoai3d[laz,viz]"    # both
```

From source, for development:

```bash
git clone https://github.com/agelamin17/geoai3d.git
cd geoai3d
pip install -e ".[dev,laz,viz]"
```

## Roadmap

Development is staged. Each stage ships something installable and useful on its own.

| Stage | Focus | Deliverable | Status |
|---|---|---|---|
| 0 | Project skeleton, CI, open development from day one | `v0.0.1` | ✅ Shipped |
| 1 | Out-of-core IO, spatial indexing, CRS and vertical datums, multi-scale geometric features, provenance | `v0.1.0` on PyPI | ✅ Shipped |
| 2 | Frugal segmentation, classical classification, GIS vector export | `v0.3.0` on PyPI and conda-forge | ⏭ Next |
| 3 | QGIS plugin, tutorial notebooks, tree segmentation, change detection | Plugin in the QGIS repository | Planned |
| 4 | Consolidation and documentation | Peer-reviewed software paper | Planned |
| 5 | Supervised segmentation and detection, pretrained geospatial models | `v1.0.0` and model zoo | Planned |
| 6 | Metric georeferenced Gaussian splatting validated against survey control | Methods paper | Planned |
| 7 | Annotation and labelling workflows | To be determined | Planned |

## Non-goals

Stating these early, because scope creep is the main risk to a project like this:

- Not competing with gsplat on rasterisation performance. GEOAI_3D wraps it and adds
  georeferencing.
- Not a desktop GUI application. Reach for non-coders comes via the QGIS plugin.
- Not an autonomous-driving toolkit. Aerial, terrestrial, and survey geometry are the
  target; automotive benchmarks and detectors are explicitly out of scope.
- Not a hosted service.

## Contributing

Early contributors are genuinely welcome, particularly anyone with survey control
data, benchmark scenes, or a workflow they would like to see supported. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [SUPPORT.md](SUPPORT.md).

## Citing

A software paper is planned. Until then, cite the repository via
[CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).

Code from proprietary or copyleft-incompatible sources is not accepted. See
[CONTRIBUTING.md](CONTRIBUTING.md#licensing) for the specific traps.
