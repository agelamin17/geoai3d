"""Tests for the columnar PointCloud representation."""

import numpy as np
import pytest

from geoai3d import PointCloud, Provenance


def _simple_cloud() -> PointCloud:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 1.0]])
    return PointCloud(xyz, attributes={"intensity": np.array([10, 20, 30])})


def test_length_reports_point_count() -> None:
    assert len(_simple_cloud()) == 3


def test_coordinates_are_stored_as_float64() -> None:
    cloud = PointCloud(np.zeros((2, 3), dtype=np.float32))
    assert cloud.xyz.dtype == np.float64


def test_coordinates_are_contiguous() -> None:
    # A non-contiguous input (a transposed view) must be made contiguous.
    xyz = np.asfortranarray(np.ones((4, 3)))
    cloud = PointCloud(xyz)
    assert cloud.xyz.flags["C_CONTIGUOUS"]


def test_attribute_names_preserve_insertion_order() -> None:
    xyz = np.zeros((2, 3))
    cloud = PointCloud(
        xyz,
        attributes={"intensity": np.zeros(2), "classification": np.zeros(2)},
    )
    assert cloud.attribute_names == ["intensity", "classification"]


def test_attribute_returns_stored_array() -> None:
    cloud = _simple_cloud()
    assert cloud.attribute("intensity").tolist() == [10, 20, 30]


def test_bad_xyz_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"shape \(n_points, 3\)"):
        PointCloud(np.zeros((4, 2)))


def test_attribute_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="has length 2, but the cloud has 3"):
        PointCloud(np.zeros((3, 3)), attributes={"intensity": np.zeros(2)})


def test_attribute_wrong_dimensionality_raises() -> None:
    with pytest.raises(ValueError, match="must be one-dimensional"):
        PointCloud(np.zeros((3, 3)), attributes={"rgb": np.zeros((3, 3))})


def test_missing_attribute_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="No attribute named 'height'"):
        _simple_cloud().attribute("height")


def test_bounds_are_correct() -> None:
    assert _simple_cloud().bounds == (0.0, 0.0, 0.0, 2.0, 2.0, 1.0)


def test_bounds_of_empty_cloud_raises() -> None:
    empty = PointCloud(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty point cloud"):
        _ = empty.bounds


def test_with_attribute_returns_new_cloud_and_leaves_original() -> None:
    cloud = PointCloud(np.zeros((3, 3)))
    derived = cloud.with_attribute("planarity", np.ones(3))
    assert derived.attribute("planarity").tolist() == [1.0, 1.0, 1.0]
    assert cloud.attribute_names == []


def test_with_attribute_shares_coordinate_array() -> None:
    cloud = PointCloud(np.zeros((3, 3)))
    derived = cloud.with_attribute("planarity", np.ones(3))
    # The coordinates are shared, not copied, so appending a feature is cheap.
    assert derived.xyz is cloud.xyz


def test_boolean_mask_filters_points_and_attributes() -> None:
    cloud = _simple_cloud()
    subset = cloud[np.array([True, False, True])]
    assert len(subset) == 2
    assert subset.xyz.tolist() == [[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]]
    assert subset.attribute("intensity").tolist() == [10, 30]


def test_slice_selection_works() -> None:
    cloud = _simple_cloud()
    assert len(cloud[1:]) == 2


def test_crs_slot_is_carried_through_filtering() -> None:
    sentinel = object()
    cloud = PointCloud(np.zeros((3, 3)), crs=sentinel)
    assert cloud[np.array([True, True, False])].crs is sentinel


def test_provenance_is_copied_not_shared_on_filtering() -> None:
    prov = Provenance(source="scan.laz")
    prov.add_step("read LAS")
    cloud = PointCloud(np.zeros((3, 3)), provenance=prov)
    subset = cloud[np.array([True, False, True])]
    subset_prov = subset.provenance
    assert subset_prov is not None
    subset_prov.add_step("filter")
    # Filtering copies provenance, so the original record is untouched.
    assert len(prov.steps) == 1


def test_repr_mentions_point_count() -> None:
    assert "n_points=3" in repr(_simple_cloud())
