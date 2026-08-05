"""Tests for the unsupervised full-scene decomposition."""

import numpy as np
import pyproj
import pytest

from geoai3d import PointCloud, segment


def _scene() -> PointCloud:
    """A flat ground plane with two spatially separated raised blocks."""
    rng = np.random.default_rng(0)
    ground = np.column_stack(
        [
            rng.uniform(0.0, 20.0, 6000),
            rng.uniform(0.0, 20.0, 6000),
            rng.normal(0.0, 0.02, 6000),
        ]
    )
    block_one = np.column_stack(
        [
            rng.uniform(3.0, 7.0, 800),
            rng.uniform(3.0, 7.0, 800),
            5.0 + rng.normal(0.0, 0.02, 800),
        ]
    )
    block_two = np.column_stack(
        [
            rng.uniform(13.0, 17.0, 800),
            rng.uniform(13.0, 17.0, 800),
            5.0 + rng.normal(0.0, 0.02, 800),
        ]
    )
    return PointCloud(np.vstack([ground, block_one, block_two]), crs=28992)


def test_segment_labels_ground_and_objects() -> None:
    out = segment(_scene(), ground_resolution=1.0, min_region_size=50)
    labels = out.attribute("segment")
    assert 0 in set(labels.tolist())  # ground is present as segment 0
    object_ids = {int(label) for label in labels if label > 0}
    assert len(object_ids) >= 2  # the two blocks are distinct segments


def test_segment_uses_existing_ground_when_not_detecting() -> None:
    cloud = _scene()
    is_ground = np.zeros(len(cloud), dtype=bool)
    is_ground[:6000] = True
    cloud = cloud.with_attribute("is_ground", is_ground)
    out = segment(cloud, detect_ground=False, min_region_size=50)
    labels = out.attribute("segment")
    assert np.all(labels[:6000] == 0)
    assert {int(label) for label in labels[6000:] if label > 0}


def test_segment_preserves_crs_and_provenance() -> None:
    out = segment(_scene(), ground_resolution=1.0, min_region_size=50)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992
    provenance = out.provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "segment"


def test_segment_empty_cloud_raises() -> None:
    with pytest.raises(ValueError, match="empty cloud"):
        segment(PointCloud(np.zeros((0, 3)), crs=28992))
