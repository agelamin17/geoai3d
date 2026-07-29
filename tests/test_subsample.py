"""Tests for subsampling."""

import numpy as np
import pytest

from geoai3d import PointCloud, subsample


def _cloud(n: int = 1000) -> PointCloud:
    rng = np.random.default_rng(0)
    return PointCloud(
        rng.random((n, 3)),
        attributes={"intensity": np.arange(n, dtype=np.uint16)},
        crs=28992,
    )


def test_random_keeps_requested_count() -> None:
    result = subsample(_cloud(), method="random", count=100, seed=0)
    assert len(result) == 100


def test_random_is_reproducible_with_seed() -> None:
    a = subsample(_cloud(), method="random", count=100, seed=42)
    b = subsample(_cloud(), method="random", count=100, seed=42)
    np.testing.assert_array_equal(a.xyz, b.xyz)


def test_voxel_reduces_and_is_a_subset() -> None:
    cloud = _cloud()
    result = subsample(cloud, method="voxel", voxel_size=0.2)
    assert 0 < len(result) < len(cloud)
    # Kept intensities are a subset of the originals (points, not averages).
    assert set(result.attribute("intensity").tolist()).issubset(
        set(cloud.attribute("intensity").tolist())
    )


def test_farthest_point_keeps_requested_count() -> None:
    result = subsample(_cloud(), method="farthest_point", count=50, seed=0)
    assert len(result) == 50


def test_subsample_preserves_crs_and_records_provenance() -> None:
    result = subsample(_cloud(), method="random", count=10, seed=0)
    assert result.crs is not None
    assert result.provenance is not None
    assert result.provenance.steps[-1].description == "subsample"


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        subsample(_cloud(), method="octree")


def test_random_without_count_raises() -> None:
    with pytest.raises(ValueError, match="requires count"):
        subsample(_cloud(), method="random")


def test_voxel_without_size_raises() -> None:
    with pytest.raises(ValueError, match="requires voxel_size"):
        subsample(_cloud(), method="voxel")


def test_random_count_too_large_raises() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        subsample(_cloud(n=10), method="random", count=100)
