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
