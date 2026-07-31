"""Command-line interface for common GEOAI_3D batch tasks.

Exposes the high-level functions as a ``geoai3d`` command so point-cloud
pipelines can run from a terminal or a shell script without writing Python:
inspect a file, convert between formats, reproject, subsample, and compute
geometric features (in memory or out-of-core). Input and output formats are
chosen from the file extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from geoai3d.crs import reproject
from geoai3d.features import geometric_features
from geoai3d.io.las import read_las, to_las
from geoai3d.io.parquet import read_parquet, to_parquet
from geoai3d.io.xyz import read_xyz, to_xyz
from geoai3d.stream import geometric_features_stream
from geoai3d.subsample import subsample

if TYPE_CHECKING:
    from geoai3d.core.pointcloud import PointCloud

_LAS_SUFFIXES = (".las", ".laz")
_TEXT_SUFFIXES = (".xyz", ".txt", ".csv")


def _read(path: str, crs: str | None) -> PointCloud:
    """Read a cloud, choosing the reader from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _LAS_SUFFIXES:
        return read_las(path, crs=crs)
    if suffix == ".parquet":
        return read_parquet(path, crs=crs)
    if suffix in _TEXT_SUFFIXES:
        return read_xyz(path, crs=crs)
    msg = f"Unsupported input format {suffix!r} for {path!r}."
    raise ValueError(msg)


def _write(cloud: PointCloud, path: str) -> None:
    """Write a cloud, choosing the writer from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _LAS_SUFFIXES:
        to_las(cloud, path)
    elif suffix == ".parquet":
        to_parquet(cloud, path)
    elif suffix in _TEXT_SUFFIXES:
        to_xyz(cloud, path)
    else:
        msg = f"Unsupported output format {suffix!r} for {path!r}."
        raise ValueError(msg)


def _cmd_info(args: argparse.Namespace) -> None:
    """Print a summary of a point-cloud file."""
    cloud = _read(args.input, args.crs)
    minx, miny, minz, maxx, maxy, maxz = cloud.bounds
    print(f"points:     {len(cloud)}")
    print(f"crs:        {cloud.crs}")
    print(f"bounds x:   {minx} .. {maxx}")
    print(f"bounds y:   {miny} .. {maxy}")
    print(f"bounds z:   {minz} .. {maxz}")
    print(f"attributes: {cloud.attribute_names}")


def _cmd_convert(args: argparse.Namespace) -> None:
    """Convert a cloud between formats, preserving the CRS."""
    _write(_read(args.input, args.crs), args.output)


def _cmd_reproject(args: argparse.Namespace) -> None:
    """Reproject a cloud to a target CRS."""
    _write(reproject(_read(args.input, args.crs), args.to), args.output)


def _cmd_subsample(args: argparse.Namespace) -> None:
    """Subsample a cloud."""
    result = subsample(
        _read(args.input, args.crs),
        method=args.method,
        voxel_size=args.voxel_size,
        count=args.count,
        seed=args.seed,
    )
    _write(result, args.output)


def _cmd_features(args: argparse.Namespace) -> None:
    """Compute in-memory geometric features."""
    result = geometric_features(
        _read(args.input, args.crs), k=args.k, radius=args.radius
    )
    _write(result, args.output)


def _cmd_features_stream(args: argparse.Namespace) -> None:
    """Compute geometric features out-of-core, streaming from disk."""
    geometric_features_stream(
        args.input,
        args.output,
        radius=args.radius,
        tile_size=args.tile_size,
        crs=args.crs,
        workers=args.workers,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``geoai3d`` command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="geoai3d", description="Geospatial-first tools for 3D point clouds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Summarise a point-cloud file.")
    info.add_argument("input")
    info.add_argument("--crs", default=None, help="Override/supply the CRS.")
    info.set_defaults(func=_cmd_info)

    convert = subparsers.add_parser("convert", help="Convert between formats.")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--crs", default=None, help="Override/supply the CRS.")
    convert.set_defaults(func=_cmd_convert)

    reproject_cmd = subparsers.add_parser("reproject", help="Reproject a cloud.")
    reproject_cmd.add_argument("input")
    reproject_cmd.add_argument("output")
    reproject_cmd.add_argument("--to", required=True, help="Target CRS (e.g. 4326).")
    reproject_cmd.add_argument("--crs", default=None, help="Source CRS override.")
    reproject_cmd.set_defaults(func=_cmd_reproject)

    sub = subparsers.add_parser("subsample", help="Subsample a cloud.")
    sub.add_argument("input")
    sub.add_argument("output")
    sub.add_argument(
        "--method", default="voxel", choices=("random", "voxel", "farthest_point")
    )
    sub.add_argument("--voxel-size", type=float, default=None, dest="voxel_size")
    sub.add_argument("--count", type=int, default=None)
    sub.add_argument("--seed", type=int, default=None)
    sub.add_argument("--crs", default=None, help="Override/supply the CRS.")
    sub.set_defaults(func=_cmd_subsample)

    features = subparsers.add_parser(
        "features", help="Compute geometric features in memory."
    )
    features.add_argument("input")
    features.add_argument("output")
    features.add_argument("--k", type=int, default=None)
    features.add_argument("--radius", type=float, default=None)
    features.add_argument("--crs", default=None, help="Override/supply the CRS.")
    features.set_defaults(func=_cmd_features)

    stream = subparsers.add_parser(
        "features-stream", help="Compute geometric features out-of-core."
    )
    stream.add_argument("input")
    stream.add_argument("output")
    stream.add_argument("--radius", type=float, required=True)
    stream.add_argument("--tile-size", type=float, required=True, dest="tile_size")
    stream.add_argument("--workers", type=int, default=1)
    stream.add_argument("--crs", default=None, help="Override/supply the CRS.")
    stream.set_defaults(func=_cmd_features_stream)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``geoai3d`` command line.

    Args:
        argv: Argument list, or ``None`` to use ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0
