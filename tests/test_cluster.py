"""Tests for connected-component clustering."""

import numpy as np
import pyproj
import pytest

from geoai3d import PointCloud, connected_components


def _two_clusters() -> PointCloud:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, (60, 3))
    b = rng.normal(10.0, 0.1, (60, 3))
    return PointCloud(np.vstack([a, b]), crs=28992)


def test_two_clusters_get_two_labels() -> None:
    out = connected_components(_two_clusters(), distance=0.5)
    labels = out.attribute("component")
    assert set(labels.tolist()) == {0, 1}
    assert np.count_nonzero(labels == 0) == 60
    assert np.count_nonzero(labels == 1) == 60


def test_min_size_marks_small_components_as_noise() -> None:
    rng = np.random.default_rng(0)
    big = rng.normal(0.0, 0.1, (60, 3))
    lone = np.array([[100.0, 100.0, 100.0]])
    out = connected_components(
        PointCloud(np.vstack([big, lone]), crs=28992), distance=0.5, min_size=2
    )
    labels = out.attribute("component")
    assert labels[-1] == -1
    assert set(labels[:60].tolist()) == {0}


def test_labels_are_contiguous_from_zero() -> None:
    out = connected_components(_two_clusters(), distance=0.5)
    kept = np.unique(out.attribute("component"))
    kept = kept[kept >= 0]
    assert kept.tolist() == list(range(len(kept)))


def test_preserves_crs_and_records_provenance() -> None:
    out = connected_components(_two_clusters(), distance=0.5)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992
    provenance = out.provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "connected_components"


def test_output_attribute_is_configurable() -> None:
    out = connected_components(
        _two_clusters(), distance=0.5, output_attribute="instance"
    )
    assert "instance" in out.attribute_names


def test_empty_cloud_raises() -> None:
    with pytest.raises(ValueError, match="empty cloud"):
        connected_components(PointCloud(np.zeros((0, 3)), crs=28992), distance=0.5)


def test_non_positive_distance_raises() -> None:
    with pytest.raises(ValueError, match="distance must be"):
        connected_components(_two_clusters(), distance=0.0)


def test_min_size_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_size must be at least 1"):
        connected_components(_two_clusters(), distance=0.5, min_size=0)
