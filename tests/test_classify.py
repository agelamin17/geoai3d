"""Tests for classical point-wise classification."""

from pathlib import Path

import numpy as np
import pyproj
import pytest

from geoai3d import (
    PointCloud,
    classify,
    feature_matrix,
    load_classifier,
    save_classifier,
    train_classifier,
)


def _labelled_cloud() -> PointCloud:
    """Two classes cleanly separable by planarity and linearity."""
    rng = np.random.default_rng(0)
    n = 400
    planarity = np.concatenate([rng.normal(0.9, 0.05, n), rng.normal(0.2, 0.05, n)])
    linearity = np.concatenate([rng.normal(0.1, 0.05, n), rng.normal(0.8, 0.05, n)])
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    xyz = rng.uniform(0.0, 10.0, (2 * n, 3))
    return PointCloud(
        xyz,
        attributes={
            "planarity": planarity,
            "linearity": linearity,
            "classification": labels,
        },
        crs=28992,
    )


def test_feature_matrix_shape() -> None:
    cloud = _labelled_cloud()
    matrix = feature_matrix(cloud, ["planarity", "linearity"])
    assert matrix.shape == (800, 2)
    assert matrix.dtype == np.float64


def test_feature_matrix_missing_attribute_raises() -> None:
    with pytest.raises(ValueError, match="not present"):
        feature_matrix(_labelled_cloud(), ["planarity", "does_not_exist"])


def test_train_and_classify_recovers_labels() -> None:
    cloud = _labelled_cloud()
    model = train_classifier(cloud, feature_names=["planarity", "linearity"])
    assert model.feature_names == ("planarity", "linearity")
    out = classify(cloud, model=model)
    predicted = out.attribute("prediction")
    truth = cloud.attribute("classification")
    assert float(np.mean(predicted == truth)) > 0.95


def test_train_requires_label_attribute() -> None:
    cloud = PointCloud(
        np.zeros((10, 3)), attributes={"planarity": np.zeros(10)}, crs=28992
    )
    with pytest.raises(ValueError, match="label attribute"):
        train_classifier(cloud, feature_names=["planarity"])


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cloud = _labelled_cloud()
    model = train_classifier(cloud, feature_names=["planarity", "linearity"])
    path = tmp_path / "forest.pkl"
    save_classifier(model, path)
    loaded = load_classifier(path)
    assert loaded.feature_names == model.feature_names
    out = classify(cloud, model=loaded)
    predicted = out.attribute("prediction")
    truth = cloud.attribute("classification")
    assert float(np.mean(predicted == truth)) > 0.95


def test_classify_preserves_crs_and_provenance() -> None:
    cloud = _labelled_cloud()
    model = train_classifier(cloud, feature_names=["planarity", "linearity"])
    out = classify(cloud, model=model)
    crs = out.crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 28992
    provenance = out.provenance
    assert provenance is not None
    assert provenance.steps[-1].description == "classify"
