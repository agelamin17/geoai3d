"""Tests for CSF ground filtering."""

import numpy as np
import pyproj
import pytest

from geoai3d import PointCloud, ground


def _scene() -> tuple[PointCloud, np.ndarray]:
    """A gently sloped ground plane with a raised building and a stand of trees."""
    rng = np.random.default_rng(0)
    gx = rng.uniform(0.0, 40.0, 4000)
    gy = rng.uniform(0.0, 40.0, 4000)
    gz = 0.05 * gx + 0.03 * gy + rng.normal(0.0, 0.02, gx.size)
    ground_pts = np.column_stack([gx, gy, gz])
    bx = rng.uniform(10.0, 18.0, 800)
    by = rng.uniform(10.0, 18.0, 800)
    bz = 0.05 * bx + 0.03 * by + 6.0 + rng.normal(0.0, 0.02, bx.size)
    building = np.column_stack([bx, by, bz])
    tx = rng.uniform(25.0, 35.0, 800)
    ty = rng.uniform(25.0, 35.0, 800)
    tz = 0.05 * tx + 0.03 * ty + rng.uniform(2.0, 9.0, tx.size)
    trees = np.column_stack([tx, ty, tz])
    xyz = np.vstack([ground_pts, building, trees])
    truth = np.concatenate(
        [
            np.ones(len(ground_pts), dtype=bool),
            np.zeros(len(building) + len(trees), dtype=bool),
        ]
    )
    return PointCloud(xyz, crs=28992), truth


def test_csf_separates_ground_from_objects() -> None:
    cloud, truth = _scene()
    out = ground(cloud, cloth_resolution=1.0, class_threshold=0.5)
    predicted = out.attribute("is_ground")
    assert predicted.dtype == bool
    ground_recall = float(np.mean(predicted[truth]))
    object_recall = float(np.mean(~predicted[~truth]))
    assert ground_recall > 0.95
    assert object_recall > 0.95


def test_flat_plane_is_all_ground() -> None:
    rng = np.random.default_rng(1)
    flat = rng.uniform(0.0, 10.0, (500, 3))
    flat[:, 2] = 0.0
    out = ground(PointCloud(flat, crs=28992), cloth_resolution=1.0)
    assert bool(out.attribute("is_ground").all())


def test_preserves_crs_and_records_provenance() -> None:
    cloud, _ = _scene()
    out = ground(cloud, cloth_resolution=1.0)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992
    provenance = out.provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "ground"


def test_output_attribute_is_configurable() -> None:
    cloud, _ = _scene()
    out = ground(cloud, cloth_resolution=1.0, output_attribute="bare_earth")
    assert "bare_earth" in out.attribute_names


def test_empty_cloud_raises() -> None:
    with pytest.raises(ValueError, match="empty cloud"):
        ground(PointCloud(np.zeros((0, 3)), crs=28992))


def test_unknown_backend_raises() -> None:
    cloud, _ = _scene()
    with pytest.raises(ValueError, match="Unknown backend"):
        ground(cloud, backend="magic")


def test_non_positive_resolution_raises() -> None:
    cloud, _ = _scene()
    with pytest.raises(ValueError, match="cloth_resolution must be"):
        ground(cloud, cloth_resolution=0.0)


def test_rigidness_below_one_raises() -> None:
    cloud, _ = _scene()
    with pytest.raises(ValueError, match="rigidness must be at least 1"):
        ground(cloud, rigidness=0)


def test_pdal_backend_runs_when_available() -> None:
    pytest.importorskip("pdal")  # skipped when PDAL is not installed
    cloud, _ = _scene()
    out = ground(cloud, backend="pdal", cloth_resolution=1.0)
    predicted = out.attribute("is_ground")
    assert predicted.dtype == bool
    assert len(predicted) == len(cloud)
