"""Tests for the out-of-core tiled engine, including the seam contract."""

import numpy as np
import pytest

from geoai3d import (
    PointCloud,
    estimate_radius,
    geometric_features,
    geometric_features_tiled,
)

_FEATURES = (
    "linearity",
    "planarity",
    "sphericity",
    "anisotropy",
    "omnivariance",
    "eigenentropy",
    "surface_variation",
    "verticality",
    "sum_eigenvalues",
)


def _scene(n: int = 3000, seed: int = 7) -> PointCloud:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 10.0, (n, 3))
    xyz[:, 2] *= 0.3  # a shallow slab, like a survey footprint
    return PointCloud(xyz, crs=28992)


def test_seam_contract_tiled_equals_whole_cloud_bitwise() -> None:
    # The defining Stage 1 guarantee: tiling with a halo must reproduce the
    # whole-cloud features exactly, bit for bit, even near tile boundaries.
    cloud = _scene()
    radius = 0.6
    whole = geometric_features(cloud, radius=radius)
    tiled = geometric_features_tiled(cloud, radius=radius, tile_size=2.5)
    for name in _FEATURES:
        np.testing.assert_array_equal(
            tiled.attribute(name),
            whole.attribute(name),
            err_msg=f"seam mismatch in {name!r}",
        )


def test_seam_contract_holds_for_small_tiles() -> None:
    # Smaller tiles mean more seams and more halo work; still must be identical.
    cloud = _scene()
    radius = 0.6
    whole = geometric_features(cloud, radius=radius)
    tiled = geometric_features_tiled(cloud, radius=radius, tile_size=1.2)
    np.testing.assert_array_equal(
        tiled.attribute("planarity"), whole.attribute("planarity")
    )


def test_single_tile_matches_whole_cloud() -> None:
    cloud = _scene()
    radius = 0.6
    whole = geometric_features(cloud, radius=radius)
    # A tile larger than the extent means one tile and no real seams.
    tiled = geometric_features_tiled(cloud, radius=radius, tile_size=1000.0)
    np.testing.assert_array_equal(
        tiled.attribute("verticality"), whole.attribute("verticality")
    )


def test_tiled_adds_features_and_records_provenance() -> None:
    tiled = geometric_features_tiled(_scene(), radius=0.6, tile_size=2.5)
    assert "planarity" in tiled.attribute_names
    assert tiled.provenance is not None
    assert tiled.provenance.steps[-1].description == "geometric_features_tiled"


def test_tiled_non_positive_radius_raises() -> None:
    with pytest.raises(ValueError, match="radius must be"):
        geometric_features_tiled(_scene(), radius=0.0, tile_size=2.5)


def test_tiled_non_positive_tile_size_raises() -> None:
    with pytest.raises(ValueError, match="tile_size must be"):
        geometric_features_tiled(_scene(), radius=0.6, tile_size=0.0)


def test_estimate_radius_is_positive_and_reasonable() -> None:
    cloud = _scene()
    radius = estimate_radius(cloud, target_neighbors=15)
    assert radius > 0.0
    # The radius should actually gather points; check it is not absurdly small.
    tiled = geometric_features_tiled(cloud, radius=radius, tile_size=2.5)
    assert np.isfinite(tiled.attribute("planarity")).mean() > 0.5


def test_estimate_radius_small_cloud_raises() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        estimate_radius(PointCloud(np.zeros((1, 3)), crs=28992))


def test_estimate_radius_bad_target_raises() -> None:
    with pytest.raises(ValueError, match="target_neighbors"):
        estimate_radius(_scene(), target_neighbors=0)
