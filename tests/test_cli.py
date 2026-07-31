"""Tests for the geoai3d command-line interface."""

from pathlib import Path

import laspy
import numpy as np
import pyproj
import pytest

from geoai3d import read_parquet
from geoai3d.cli import main


def _make_las(path: Path, n: int = 800, *, with_crs: bool = True) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 10.0, (n, 3))
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = np.full(3, 0.001)
    if with_crs:
        header.add_crs(pyproj.CRS.from_epsg(28992))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(str(path))


def test_info_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    assert main(["info", str(las)]) == 0
    out = capsys.readouterr().out
    assert "points:     800" in out
    assert "crs:" in out


def test_convert_las_to_parquet(tmp_path: Path) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    out = tmp_path / "out.parquet"
    assert main(["convert", str(las), str(out)]) == 0
    assert out.exists()
    assert len(read_parquet(out)) == 800


def test_reproject_command(tmp_path: Path) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    out = tmp_path / "out.parquet"
    main(["reproject", str(las), str(out), "--to", "4326"])
    crs = read_parquet(out).crs
    assert isinstance(crs, pyproj.CRS)
    assert crs.to_epsg() == 4326


def test_subsample_command(tmp_path: Path) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    out = tmp_path / "out.parquet"
    main(["subsample", str(las), str(out), "--method", "random", "--count", "100"])
    assert len(read_parquet(out)) == 100


def test_features_command(tmp_path: Path) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    out = tmp_path / "out.parquet"
    main(["features", str(las), str(out), "--radius", "0.8"])
    assert "planarity" in read_parquet(out).attribute_names


def test_features_stream_command(tmp_path: Path) -> None:
    las = tmp_path / "in.las"
    _make_las(las)
    out = tmp_path / "out.parquet"
    main(
        [
            "features-stream",
            str(las),
            str(out),
            "--radius",
            "0.8",
            "--tile-size",
            "3.0",
        ]
    )
    assert "planarity" in read_parquet(out).attribute_names


def test_unsupported_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported input format"):
        main(["info", str(tmp_path / "cloud.e57")])
