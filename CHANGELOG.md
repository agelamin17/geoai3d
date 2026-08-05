# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This is the deliberate pre-conda-forge API-stabilisation window (targeting a
PyPI-only `0.2.0`). Breaking changes are batched here so the churn happens before
the conda-forge channel exists.

### Added
- `read_lidar`: a high-level reader that selects the format-specific reader from
  the file extension (`.las`/`.laz` → `read_las`; `.xyz`/`.txt`/`.asc`/`.pts` →
  `read_xyz`). This is now the documented entry point for loading point clouds.
  The format-specific readers stay public as escape hatches for known formats or
  format-specific options. Parquet is read with `read_parquet` and is
  deliberately not routed through `read_lidar`, being the package's own at-rest
  serialization format rather than an acquisition format.
- `geometric_features_stream` gained an `attributes` parameter selecting which
  source point dimensions to carry into the output. The default (`None`) carries
  all of them; pass a sequence of dimension names to carry only a subset and
  keep the output smaller.
- `evaluate` and `confusion_matrix` with an `AccuracyReport`: classification
  accuracy assessment (C9) -- confusion matrix, overall accuracy, and per-class
  precision, recall, F1, and IoU, plus mean IoU and macro F1. Pure NumPy.
- `spatial_block_split`: spatially-blocked cross-validation, grouping points into
  square blocks and dealing whole blocks into folds so test points never
  neighbour their own training points -- avoiding the inflated accuracy that
  random splits give on spatially autocorrelated data. Returns scikit-learn-style
  `(train, test)` index pairs.
- `feature_matrix`, `train_classifier`, `classify`, and a `Classifier` wrapper:
  classical point-wise semantic classification with a scikit-learn random forest
  on per-point features (C1, C2). `feature_matrix` assembles a matrix from named
  attributes (defaulting to the geometric features and normalised intensity
  present); `train_classifier` fits a forest on a labelled cloud; `classify`
  applies it, adding a `prediction` column. `save_classifier`/`load_classifier`
  persist a trained model (pickle). A fast CPU-only baseline, no GPU.
- `segment`: one-call unsupervised full-scene decomposition (B6), chaining
  ground filtering, surface normals and geometric features, and region growing
  into a single `segment` label per point (ground `0`, each object `1+`,
  unassigned `-1`). No labels, no training, no GPU -- the frugal laptop workflow.
- `dbscan`: density-based clustering (via scikit-learn), adding an integer
  `cluster` column with `-1` for unclustered points (B5).
- `region_growing`: normal-based unsupervised segmentation (after Rabbani et al.)
  growing smooth regions from seeds and adding an integer `segment` column,
  splitting a scene by surface orientation with no training data (B2). Needs
  surface normals from `estimate_normals`.
- scikit-learn added as a base dependency (CPU-only prebuilt wheels, no
  compiler), powering `dbscan` and the classical classifiers to come.
- `fit_plane`: RANSAC plane fitting, returning a `Plane` (unit normal, offset,
  inlier mask, `signed_distance`) refit to its inliers by least squares. The
  workhorse primitive for ground, walls, and roof facets, and a robust
  dominant-plane fallback for terrestrial scans (B1). Pure NumPy.
- `connected_components`: connectivity-based instance labelling, adding an integer
  `component` column (contiguous ids, `-1` for groups below `min_size`) via a
  KD-tree distance graph and SciPy sparse components (B8).
- `Raster`: a georeferenced 2D grid (values plus an affine transform, CRS, and
  nodata), the raster counterpart to `PointCloud`, keeping the CRS attached
  through every operation.
- `to_dtm` / `to_dsm`: rasterise a cloud to a Digital Terrain Model (bare earth,
  using a ground attribute when present) or Digital Surface Model (top surface),
  gridded on the shared cloud extent so the two align. `difference` subtracts two
  aligned rasters (DSM minus DTM for object heights, or two epochs for change),
  and `volume` integrates a height raster into cut, fill, and net volumes (G7).
  The raster maths is pure NumPy in the base install.
- `read_geotiff` / `to_geotiff`: GeoTIFF IO for rasters via rasterio, in the new
  optional `[gis]` extra (`pip install geoai3d[gis]`); the array maths needs no
  extra.
- `ground`: training-free ground filtering by cloth simulation (CSF, after
  Zhang et al. 2016), adding a boolean `is_ground` column. A pure-NumPy default
  backend (memory scales with the cloth grid, not the point count; no compiler)
  and an optional `backend="pdal"` delegating to PDAL's `filters.csf` for users
  who have PDAL installed. Assumes an airborne 2.5-D scene; terrestrial scans
  with walls or overhangs are a first pass (B7).
- `normalize_intensity`: radiometric normalisation of LiDAR intensity, applying
  the range (`(R / R_ref) ** 2`) and optional incidence-angle (`1 / cos(theta)`,
  from `nx`/`ny`/`nz` normals) corrections after Hofle & Pfeifer (2007). Takes an
  explicit sensor position, as a single `(3,)` origin (terrestrial) or a
  per-point `(N, 3)` trajectory (airborne/mobile), in the cloud's CRS; adds a
  `normalized_intensity` column and records the correction in the lineage.
  CPU-only, pure NumPy (B-series feature).

### Changed
- **Breaking (output shape):** `geometric_features_stream` now carries every
  source point attribute (intensity, returns, classification, GPS time, extra
  dimensions, and so on) through to its Parquet output, each with its original
  dtype, alongside the existing `index`, coordinate, and feature columns.
  Previously the streamed output dropped source attributes. This resolves the
  attribute carry-through item deferred from 0.1.0. Code that assumed a fixed
  streamed schema (index, x/y/z, and features only) must adjust; pass
  `attributes=[...]` to restrict the carried set.
- **Breaking:** `PointCloud` now coerces its `crs` argument to a `pyproj.CRS` at
  construction, so `PointCloud.crs` always returns a `pyproj.CRS` (or `None`)
  regardless of whether an EPSG code, a WKT/PROJ string, or a `pyproj.CRS` was
  passed. Two consequences: an unrecognisable `crs` now raises `ValueError` at
  construction rather than being stored verbatim, and the `crs` property is now
  typed `pyproj.CRS | None` rather than `object | None`. Code that relied on
  `PointCloud.crs` returning the exact object it was given must adjust.

## [0.1.0] - 2026-08-01

The out-of-core georeferenced foundation. Benchmark: multi-scale radius
geometric features on a 722,964,406-point AHN LiDAR tile — which would need
~21.5 GB to hold in memory, above a 16 GB budget — computed within **8.9 GB of
peak memory** in ~29 minutes on 4 CPU cores, bit-for-bit identical to the
whole-cloud result. Runs on a CPU-only laptop with no compiler.

### Added
- `core` subpackage holding the foundational, dependency-light data structures.
- `PointCloud`: the columnar (struct-of-arrays) in-memory representation, with
  `float64` coordinates, per-point attribute columns, attribute-preserving
  filtering and slicing, cheap `with_attribute`, axis-aligned `bounds`, and
  slots for a coordinate reference system and a provenance record.
- `Provenance` and `ProcessStep`: an ISO 19115-style lineage record that every
  operation appends to, capturing source, parameters, software version, and a
  UTC timestamp.
- NumPy added as the first base runtime dependency (CPU-only, no compiler).
- `read_las` and `to_las`: LAS/LAZ reading and writing via laspy, preserving
  standard and user-defined per-point dimensions and round-tripping the
  coordinate reference system. Reading a file without a CRS raises rather than
  returning an unreferenced cloud; writing requires a CRS.
- laspy and pyproj added as base dependencies (both CPU-only, no compiler).
- Optional `laz` extra (`pip install geoai3d[laz]`) for compressed-LAZ support
  via the lazrs backend.
- `to_parquet` / `read_parquet`: Parquet at-rest format storing every column
  losslessly, with the CRS and provenance record in the file's schema metadata.
- `Provenance.to_dict` / `from_dict` and `ProcessStep.to_dict` / `from_dict`
  for JSON-serialising lineage into file metadata.
- `read_xyz` / `to_xyz`: plain XYZ text IO, writing/reading the CRS via a
  sibling `.prj` file since the format cannot store it inline.
- `reproject`: horizontal reprojection between coordinate reference systems.
- `reproject_3d`: full 3D reprojection including the vertical datum, converting
  between ellipsoidal and orthometric (geoid-based) heights, with a clear error
  when a required geoid grid is unavailable offline.
- pyarrow added as a base dependency (CPU-only, no compiler).
- `build_kdtree`: a SciPy KD-tree over a cloud's coordinates (A4).
- `subsample`: random, voxel (keep-first), and farthest-point subsampling,
  all attribute-preserving (A6).
- `estimate_normals` and `remove_statistical_outliers`: PCA surface normals and
  statistical outlier removal over a KD-tree (A7).
- `geometric_features`: eigenvalue descriptors (linearity, planarity,
  sphericity, anisotropy, omnivariance, eigenentropy, surface variation,
  verticality, sum of eigenvalues) at a k- or radius-neighbourhood, after
  Weinmann et al. and Demantke et al. (B3).
- `multiscale_features`: the same descriptors at each point's eigenentropy-
  optimal neighbourhood size, with the chosen size as `optimal_k` (B4).
- scipy added as a base dependency (CPU-only, no compiler).
- `geometric_features_tiled`: out-of-core, tile-by-tile computation of
  fixed-radius geometric features, using a one-radius halo so that features
  near tile boundaries are **bit-for-bit identical** to the whole-cloud result
  (the seam contract). Neighbours are summed in global-index order to keep the
  two paths bitwise equal (A3, A5, and the tile-seam contract).
- `estimate_radius`: pick a feature radius from the cloud's own density so the
  out-of-core engine has a fixed halo width without the caller guessing.
- The per-point neighbourhood math now lives in `core._neighborhood`, shared
  verbatim by the in-memory feature functions and the tiled engine.
- `geometric_features_stream`: disk-based out-of-core features for a LAS/LAZ
  file that does not fit in memory. Streams the file once to partition points
  into per-tile temporary files (replicating each into neighbouring halos), then
  computes features tile by tile and writes a GEOAI_3D Parquet. Bounded memory,
  optional multi-process parallelism, and bit-identical to the whole-cloud
  result (the seam contract, on disk).
- `geoai3d` command-line interface (J5): `info`, `convert`, `reproject`,
  `subsample`, `features`, and `features-stream` subcommands, choosing input
  and output formats from the file extension.
- `view`: interactive Jupyter/Colab 3D viewer (I1, I4) via plotly, colouring
  by height, an attribute, or true RGB, with automatic thinning for display.
  In the optional `viz` extra (`pip install geoai3d[viz]`).
- `benchmarks/`: a runnable out-of-core benchmark on the public CC0 AHN dataset
  (memory + timing) plus a streaming tile-merge helper and instructions (J6).

### Deferred

- COPC spatial-index reads (to skip the partition pass on already-indexed files)
  and carrying the full set of source point attributes through the stream. The
  streaming engine and its bit-identical seam guarantee are done; these are
  performance and fidelity refinements on top.

- PLY/PCD/E57 IO and the format-conversion utility, which depend on Open3D (an
  optional `geometry` extra, kept out of the base install) and pye57. These
  need a live environment to validate and will land in a later increment.

## [0.0.1] - 2026-07-21

### Added
- Project skeleton: packaging, licence, and documentation scaffolding.
- Continuous integration across Linux, macOS, and Windows on Python 3.10 to 3.13.
- A dedicated CI job verifying that the base install succeeds in a container with
  no compiler and no GPU.
- Linting, formatting, and strict type checking via ruff and mypy.
- Contribution, support, governance, and AI usage documentation.

[Unreleased]: https://github.com/agelamin17/geoai3d/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agelamin17/geoai3d/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/agelamin17/geoai3d/releases/tag/v0.0.1
