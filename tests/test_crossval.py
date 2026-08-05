"""Tests for spatially-blocked cross-validation."""

import numpy as np
import pytest

from geoai3d import PointCloud, spatial_block_split


def _cloud(n: int = 5000) -> PointCloud:
    xyz = np.random.default_rng(0).uniform(0.0, 100.0, (n, 3))
    return PointCloud(xyz, crs=28992)


def test_folds_cover_and_are_disjoint() -> None:
    cloud = _cloud()
    folds = spatial_block_split(cloud, block_size=10.0, n_folds=5)
    assert len(folds) == 5
    covered = np.zeros(len(cloud), dtype=int)
    for train, test in folds:
        assert set(train.tolist()).isdisjoint(test.tolist())
        assert len(train) + len(test) == len(cloud)
        covered[test] += 1
    # Every point is in exactly one fold's test set.
    np.testing.assert_array_equal(covered, np.ones(len(cloud), dtype=int))


def test_each_block_stays_in_one_fold() -> None:
    cloud = _cloud()
    block_size = 10.0
    folds = spatial_block_split(cloud, block_size=block_size, n_folds=5)
    point_fold = np.empty(len(cloud), dtype=int)
    for fold_id, (_, test) in enumerate(folds):
        point_fold[test] = fold_id
    xyz = cloud.xyz
    block_col = np.floor((xyz[:, 0] - xyz[:, 0].min()) / block_size).astype(int)
    block_row = np.floor((xyz[:, 1] - xyz[:, 1].min()) / block_size).astype(int)
    _, block_index = np.unique(
        np.column_stack([block_col, block_row]), axis=0, return_inverse=True
    )
    block_index = np.asarray(block_index).ravel()
    for block in np.unique(block_index):
        assert len(set(point_fold[block_index == block].tolist())) == 1


def test_empty_cloud_raises() -> None:
    with pytest.raises(ValueError, match="empty cloud"):
        spatial_block_split(PointCloud(np.zeros((0, 3)), crs=28992), block_size=10.0)


def test_non_positive_block_size_raises() -> None:
    with pytest.raises(ValueError, match="block_size must be"):
        spatial_block_split(_cloud(), block_size=0.0)


def test_too_few_folds_raises() -> None:
    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        spatial_block_split(_cloud(), block_size=10.0, n_folds=1)


def test_too_few_blocks_raises() -> None:
    with pytest.raises(ValueError, match="Use a smaller"):
        spatial_block_split(_cloud(), block_size=1000.0, n_folds=5)
