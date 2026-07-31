"""Tests for the interactive viewer (skipped when plotly is not installed)."""

import numpy as np
import pytest

from geoai3d import PointCloud, view


def test_view_returns_scatter3d_figure() -> None:
    pytest.importorskip("plotly.graph_objects")
    cloud = PointCloud(np.random.default_rng(0).random((500, 3)), crs=28992)
    figure = view(cloud)
    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter3d"
    assert len(figure.data[0].x) == 500


def test_view_thins_large_cloud() -> None:
    pytest.importorskip("plotly.graph_objects")
    cloud = PointCloud(np.random.default_rng(0).random((5000, 3)), crs=28992)
    figure = view(cloud, max_points=1000, seed=0)
    assert len(figure.data[0].x) == 1000


def test_view_color_by_attribute() -> None:
    pytest.importorskip("plotly.graph_objects")
    cloud = PointCloud(
        np.zeros((10, 3)),
        attributes={"intensity": np.arange(10, dtype=np.uint16)},
        crs=28992,
    )
    figure = view(cloud, color_by="intensity")
    assert len(figure.data[0].marker.color) == 10


def test_view_bad_color_raises() -> None:
    pytest.importorskip("plotly.graph_objects")
    cloud = PointCloud(np.zeros((10, 3)), crs=28992)
    with pytest.raises(ValueError, match="Cannot colour by"):
        view(cloud, color_by="nope")
