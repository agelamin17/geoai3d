"""Tests for the Raster type and DTM/DSM/difference/volume operations."""

import numpy as np
import pyproj
import pytest

from geoai3d import (
    PointCloud,
    Raster,
    difference,
    read_geotiff,
    to_dsm,
    to_dtm,
    to_geotiff,
    volume,
)

_NORTH_UP = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)


def _cloud() -> PointCloud:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 10.0, (4000, 3))
    return PointCloud(xyz, crs=28992)


def test_raster_properties() -> None:
    raster = Raster(np.ones((4, 4)), _NORTH_UP, 28992, nodata=-9999.0)
    assert raster.shape == (4, 4)
    assert raster.height == 4
    assert raster.width == 4
    assert raster.resolution == (1.0, 1.0)
    assert raster.bounds == (0.0, 0.0, 4.0, 4.0)
    assert isinstance(raster.crs, pyproj.CRS)
    assert raster.crs.to_epsg() == 28992


def test_raster_requires_crs() -> None:
    with pytest.raises(ValueError, match="requires a coordinate reference"):
        Raster(np.ones((2, 2)), _NORTH_UP, None)


def test_raster_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError, match="data must be 2D"):
        Raster(np.ones(4), _NORTH_UP, 28992)


def test_dtm_of_flat_plane_is_constant() -> None:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 10.0, (5000, 3))
    xyz[:, 2] = 2.0
    dtm = to_dtm(PointCloud(xyz, crs=28992), resolution=1.0)
    assert np.allclose(np.nanmin(dtm.data), 2.0)
    assert np.allclose(np.nanmax(dtm.data), 2.0)
    assert dtm.crs.to_epsg() == 28992


def test_dsm_is_at_least_dtm() -> None:
    cloud = _cloud()
    dtm = to_dtm(cloud, resolution=1.0)
    dsm = to_dsm(cloud, resolution=1.0)
    both = ~np.isnan(dtm.data) & ~np.isnan(dsm.data)
    assert np.all(dsm.data[both] >= dtm.data[both])


def test_object_volume_from_difference() -> None:
    rng = np.random.default_rng(1)
    ground = np.column_stack(
        [rng.uniform(0.0, 10.0, 20000), rng.uniform(0.0, 10.0, 20000), np.zeros(20000)]
    )
    bx = rng.uniform(3.0, 7.0, 8000)
    by = rng.uniform(3.0, 7.0, 8000)
    box = np.column_stack([bx, by, np.ones(bx.size)])  # 4x4 m footprint, 1 m tall
    cloud = PointCloud(np.vstack([ground, box]), crs=28992)
    ndsm = difference(to_dsm(cloud, resolution=0.5), to_dtm(cloud, resolution=0.5))
    assert volume(ndsm)["fill"] == pytest.approx(16.0, abs=0.5)


def test_volume_cut_and_fill_and_net() -> None:
    data = np.array([[2.0, -1.0]])  # one 1 m^2 cell up 2, one down 1
    raster = Raster(data, (1.0, 0.0, 0.0, 0.0, -1.0, 1.0), 28992)
    result = volume(raster)
    assert result["fill"] == pytest.approx(2.0)
    assert result["cut"] == pytest.approx(1.0)
    assert result["net"] == pytest.approx(1.0)


def test_difference_shape_mismatch_raises() -> None:
    a = Raster(np.ones((2, 2)), (1.0, 0.0, 0.0, 0.0, -1.0, 2.0), 28992)
    b = Raster(np.ones((3, 3)), (1.0, 0.0, 0.0, 0.0, -1.0, 3.0), 28992)
    with pytest.raises(ValueError, match="not aligned: shapes"):
        difference(a, b)


def test_difference_transform_mismatch_raises() -> None:
    a = Raster(np.ones((2, 2)), (1.0, 0.0, 0.0, 0.0, -1.0, 2.0), 28992)
    b = Raster(np.ones((2, 2)), (2.0, 0.0, 0.0, 0.0, -2.0, 4.0), 28992)
    with pytest.raises(ValueError, match="affine transforms differ"):
        difference(a, b)


def test_difference_crs_mismatch_raises() -> None:
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    a = Raster(np.ones((2, 2)), transform, 28992)
    b = Raster(np.ones((2, 2)), transform, 4326)
    with pytest.raises(ValueError, match="different CRS"):
        difference(a, b)


def test_to_dtm_uses_ground_attribute() -> None:
    cloud = _cloud().with_attribute("is_ground", np.zeros(len(_cloud()), dtype=bool))
    with pytest.raises(ValueError, match="all false"):
        to_dtm(cloud, resolution=1.0)


def test_to_dtm_empty_cloud_raises() -> None:
    with pytest.raises(ValueError, match="empty cloud"):
        to_dtm(PointCloud(np.zeros((0, 3)), crs=28992), resolution=1.0)


def test_to_dtm_non_positive_resolution_raises() -> None:
    with pytest.raises(ValueError, match="resolution must be"):
        to_dtm(_cloud(), resolution=0.0)


def test_geotiff_round_trip() -> None:
    rasterio = pytest.importorskip("rasterio")  # skipped without the [gis] extra
    assert rasterio is not None
    import tempfile
    from pathlib import Path

    raster = Raster(np.arange(16.0).reshape(4, 4), _NORTH_UP, 28992, nodata=-9999.0)
    path = Path(tempfile.mkdtemp()) / "raster.tif"
    to_geotiff(raster, path)
    loaded = read_geotiff(path)
    assert loaded.shape == (4, 4)
    np.testing.assert_allclose(loaded.data, raster.data)
    assert loaded.crs.to_epsg() == 28992
