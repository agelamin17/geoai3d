# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Deferred

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

[Unreleased]: https://github.com/agelamin17/geoai3d/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/agelamin17/geoai3d/releases/tag/v0.0.1
