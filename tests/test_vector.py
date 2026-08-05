"""Tests for GIS vector export."""

from pathlib import Path

import numpy as np
import pytest

from geoai3d import PointCloud, Provenance, to_geopackage


def _classified_cloud() -> PointCloud:
    xyz = np.random.default_rng(0).uniform(0.0, 100.0, (200, 3))
    classification = np.random.default_rng(1).integers(0, 5, 200).astype(np.int64)
    provenance = Provenance(source="scan.laz")
    provenance.add_step("classify")
    return PointCloud(
        xyz,
        attributes={"classification": classification},
        crs=28992,
        provenance=provenance,
    )


def test_geopackage_round_trip(tmp_path: Path) -> None:
    geopandas = pytest.importorskip("geopandas")  # skipped without the [gis] extra
    cloud = _classified_cloud()
    path = tmp_path / "points.gpkg"
    to_geopackage(cloud, path)

    frame = geopandas.read_file(path)
    assert len(frame) == len(cloud)
    assert frame.crs.to_epsg() == 28992
    assert "classification" in frame.columns
    np.testing.assert_array_equal(
        np.sort(frame["classification"].to_numpy()),
        np.sort(cloud.attribute("classification")),
    )


def test_geopackage_writes_provenance_sidecar(tmp_path: Path) -> None:
    pytest.importorskip("geopandas")
    cloud = _classified_cloud()
    path = tmp_path / "points.gpkg"
    to_geopackage(cloud, path)
    sidecar = Path(f"{path}.provenance.json")
    assert sidecar.exists()
    assert "scan.laz" in sidecar.read_text(encoding="utf-8")


def test_geopackage_subset_of_attributes(tmp_path: Path) -> None:
    geopandas = pytest.importorskip("geopandas")
    cloud = _classified_cloud().with_attribute(
        "intensity", np.ones(200, dtype=np.uint16)
    )
    path = tmp_path / "points.gpkg"
    to_geopackage(cloud, path, attributes=["classification"])
    frame = geopandas.read_file(path)
    assert "classification" in frame.columns
    assert "intensity" not in frame.columns


def test_geopackage_requires_crs(tmp_path: Path) -> None:
    pytest.importorskip("geopandas")
    cloud = PointCloud(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="coordinate reference system"):
        to_geopackage(cloud, tmp_path / "points.gpkg")
