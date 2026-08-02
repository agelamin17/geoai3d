"""Tests for radiometric intensity normalisation."""

import numpy as np
import pyproj
import pytest

from geoai3d import PointCloud, normalize_intensity


def _intensity_cloud(n: int = 5) -> PointCloud:
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.linspace(0.0, 10.0, n)
    return PointCloud(xyz, attributes={"intensity": np.ones(n)}, crs=28992)


def test_range_correction_removes_distance_falloff() -> None:
    n = 200
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.linspace(0.0, 40.0, n)
    sensor = [20.0, 0.0, 100.0]
    ranges = np.linalg.norm(xyz - np.asarray(sensor), axis=1)
    constant = 1.0e6
    raw = constant / ranges**2  # pure 1/R^2 falloff, no angle term
    cloud = PointCloud(xyz, attributes={"intensity": raw}, crs=28992)

    out = normalize_intensity(cloud, sensor_position=sensor)  # range only
    corrected = out.attribute("normalized_intensity")
    reference = float(np.median(ranges))
    np.testing.assert_allclose(corrected, constant / reference**2, rtol=1e-9)


def test_range_and_incidence_recover_constant_reflectance() -> None:
    rng = np.random.default_rng(0)
    n = 500
    xyz = np.zeros((n, 3))
    xyz[:, 0] = rng.uniform(0.0, 50.0, n)
    xyz[:, 1] = rng.uniform(0.0, 50.0, n)
    sensor = np.column_stack(
        [
            rng.uniform(0.0, 50.0, n),
            rng.uniform(0.0, 50.0, n),
            rng.uniform(100.0, 150.0, n),
        ]
    )
    beam = xyz - sensor
    ranges = np.linalg.norm(beam, axis=1)
    cos_true = np.abs(beam[:, 2]) / ranges  # normals are +z on flat ground
    constant, reflectance = 1.0e6, 0.5
    raw = constant * reflectance * cos_true / ranges**2
    cloud = PointCloud(
        xyz,
        attributes={
            "intensity": raw,
            "nx": np.zeros(n),
            "ny": np.zeros(n),
            "nz": np.ones(n),
        },
        crs=28992,
    )

    out = normalize_intensity(
        cloud,
        sensor_position=sensor,
        correct_range=True,
        correct_incidence_angle=True,
    )
    corrected = out.attribute("normalized_intensity")
    reference = float(np.median(ranges))
    np.testing.assert_allclose(
        corrected, constant * reflectance / reference**2, rtol=1e-9
    )


def test_origin_matches_broadcast_per_point() -> None:
    cloud = _intensity_cloud(50)
    origin = [10.0, 10.0, 80.0]
    per_point = np.tile(np.asarray(origin), (len(cloud), 1))
    a = normalize_intensity(cloud, sensor_position=origin)
    b = normalize_intensity(cloud, sensor_position=per_point)
    np.testing.assert_array_equal(
        a.attribute("normalized_intensity"), b.attribute("normalized_intensity")
    )


def test_explicit_reference_range_scales_output() -> None:
    cloud = _intensity_cloud(100)
    sensor = [5.0, 0.0, 60.0]
    near = normalize_intensity(cloud, sensor_position=sensor, reference_range=50.0)
    far = normalize_intensity(cloud, sensor_position=sensor, reference_range=100.0)
    ratio = far.attribute("normalized_intensity") / near.attribute(
        "normalized_intensity"
    )
    np.testing.assert_allclose(ratio, (50.0 / 100.0) ** 2)


def test_preserves_crs_and_records_provenance() -> None:
    out = normalize_intensity(_intensity_cloud(), sensor_position=[0.0, 0.0, 30.0])
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992
    provenance = out.provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "normalize_intensity"


def test_output_attribute_name_is_configurable() -> None:
    out = normalize_intensity(
        _intensity_cloud(), sensor_position=[0.0, 0.0, 30.0], output_attribute="i_norm"
    )
    assert "i_norm" in out.attribute_names


def test_missing_intensity_raises() -> None:
    cloud = PointCloud(np.zeros((5, 3)), crs=28992)
    with pytest.raises(ValueError, match="no 'intensity' attribute"):
        normalize_intensity(cloud, sensor_position=[0.0, 0.0, 10.0])


def test_incidence_without_normals_raises() -> None:
    with pytest.raises(ValueError, match="surface normals"):
        normalize_intensity(
            _intensity_cloud(),
            sensor_position=[0.0, 0.0, 10.0],
            correct_incidence_angle=True,
        )


def test_bad_sensor_shape_raises() -> None:
    with pytest.raises(ValueError, match="sensor_position must have shape"):
        normalize_intensity(
            _intensity_cloud(5), sensor_position=[[1.0, 2.0], [3.0, 4.0]]
        )


def test_zero_range_raises() -> None:
    cloud = PointCloud(
        np.zeros((3, 3)), attributes={"intensity": np.ones(3)}, crs=28992
    )
    with pytest.raises(ValueError, match="zero range"):
        normalize_intensity(cloud, sensor_position=[0.0, 0.0, 0.0])


def test_no_correction_selected_raises() -> None:
    with pytest.raises(ValueError, match="Nothing to normalise"):
        normalize_intensity(
            _intensity_cloud(), sensor_position=[0.0, 0.0, 10.0], correct_range=False
        )


def test_non_positive_reference_range_raises() -> None:
    with pytest.raises(ValueError, match="reference_range must be"):
        normalize_intensity(
            _intensity_cloud(), sensor_position=[0.0, 0.0, 10.0], reference_range=0.0
        )
