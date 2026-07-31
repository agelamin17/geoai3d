"""Streaming, disk-based out-of-core geometric features.

Computes fixed-radius geometric features on a LAS/LAZ file that does not fit in
memory, in two passes that never hold the whole cloud at once:

1. Stream the file once and partition points into per-tile temporary Parquet
   files. Each point is written to its own tile and, if it lies within one
   radius of a neighbouring tile, replicated into that neighbour's halo -- but
   never twice into the same tile, which would corrupt a covariance.
2. For each tile, read its temp file (core points plus halo), compute features
   for the core points, and append them to the output Parquet.

Because neighbours are summed in global (file) order, the result is identical --
bit for bit -- to computing on the whole cloud in memory, so the seam contract
extends to disk. Memory stays bounded to one tile at a time, and the per-tile
feature pass can run across several processes.

The output is a GEOAI_3D Parquet (readable by :func:`~geoai3d.read_parquet`)
with an ``index`` column giving each point's row position in the source file,
its coordinates, and the geometric feature columns.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import laspy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from geoai3d.core._crs import coerce_crs
from geoai3d.core._neighborhood import FEATURE_NAMES, features_from_eigen, radius_eigen
from geoai3d.core.provenance import Provenance

if TYPE_CHECKING:
    import os

# Metadata keys mirror geoai3d.io.parquet so the output loads with read_parquet.
_CRS_KEY = b"geoai3d:crs"
_PROVENANCE_KEY = b"geoai3d:provenance"
_SCHEMA_VERSION_KEY = b"geoai3d:schema_version"
_SCHEMA_VERSION = b"1"

_TEMP_SCHEMA = pa.schema(
    [
        ("index", pa.int64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("z", pa.float64()),
    ]
)

_TileResult = tuple[
    NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], dict[str, NDArray[Any]]
]


def _tile_worker(item: tuple[str, int, int, float, float, float, float]) -> _TileResult:
    """Compute core-point features for one tile from its temp Parquet file."""
    temp_path, tile_i, tile_j, origin_x, origin_y, tile_size, radius = item
    table = pq.read_table(temp_path)
    index = np.asarray(table.column("index"))
    x = np.asarray(table.column("x"))
    y = np.asarray(table.column("y"))
    z = np.asarray(table.column("z"))
    xyz = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    tree = cKDTree(xyz)

    # Decide core membership with the same floor() used to build the tiles, so
    # each point is the core of exactly one tile with no boundary ambiguity
    # (comparing against reconstructed tile edges can double-count a point that
    # lands exactly on a seam, because adjacent edges differ by a floating bit).
    core_i = np.floor((x - origin_x) / tile_size).astype(np.int64)
    core_j = np.floor((y - origin_y) / tile_size).astype(np.int64)
    core = (core_i == tile_i) & (core_j == tile_j)
    core_positions = np.flatnonzero(core)

    eigenvalues, normals = radius_eigen(
        xyz, index.astype(np.intp), tree, core_positions, radius
    )
    features = features_from_eigen(eigenvalues, normals)
    return (
        index[core_positions],
        x[core_positions],
        y[core_positions],
        z[core_positions],
        features,
    )


def _get_writer(
    writers: dict[tuple[int, int], Any],
    paths: dict[tuple[int, int], Path],
    temp_dir: Path,
    tile_i: int,
    tile_j: int,
) -> Any:
    """Return (creating if needed) the temp Parquet writer for a tile."""
    key = (tile_i, tile_j)
    if key not in writers:
        path = temp_dir / f"tile_{tile_i}_{tile_j}.parquet"
        writers[key] = pq.ParquetWriter(str(path), _TEMP_SCHEMA)
        paths[key] = path
    return writers[key]


def _partition(
    reader: Any,
    temp_dir: Path,
    origin_x: float,
    origin_y: float,
    tile_size: float,
    radius: float,
    chunk_size: int,
) -> dict[tuple[int, int], Path]:
    """Stream the file, writing each point to its tile and neighbouring halos."""
    writers: dict[tuple[int, int], Any] = {}
    paths: dict[tuple[int, int], Path] = {}
    base = 0
    for points in reader.chunk_iterator(chunk_size):
        count = len(points)
        x = np.asarray(points.x, dtype=np.float64)
        y = np.asarray(points.y, dtype=np.float64)
        z = np.asarray(points.z, dtype=np.float64)
        index = np.arange(base, base + count, dtype=np.int64)
        base += count

        u = x - origin_x
        v = y - origin_y
        i_low = np.floor((u - radius) / tile_size).astype(np.int64)
        i_high = np.floor((u + radius) / tile_size).astype(np.int64)
        j_low = np.floor((v - radius) / tile_size).astype(np.int64)
        j_high = np.floor((v + radius) / tile_size).astype(np.int64)
        combos = (
            (i_low, j_low, np.ones(count, dtype=bool)),
            (i_high, j_low, i_high != i_low),
            (i_low, j_high, j_high != j_low),
            (i_high, j_high, (i_high != i_low) & (j_high != j_low)),
        )
        for tile_i_array, tile_j_array, mask in combos:
            selected = np.flatnonzero(mask)
            if selected.size == 0:
                continue
            pairs = np.column_stack([tile_i_array[selected], tile_j_array[selected]])
            unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            for group, (tile_i, tile_j) in enumerate(unique_pairs):
                members = selected[inverse == group]
                writer = _get_writer(
                    writers, paths, temp_dir, int(tile_i), int(tile_j)
                )
                writer.write_table(
                    pa.table(
                        {
                            "index": index[members],
                            "x": x[members],
                            "y": y[members],
                            "z": z[members],
                        }
                    )
                )
    for writer in writers.values():
        writer.close()
    return paths


def _output_metadata(
    crs: Any, source: str, radius: float, tile_size: float
) -> dict[bytes, bytes]:
    """Build the schema metadata (CRS + provenance) for the output file."""
    provenance = Provenance(source=source)
    provenance.add_step(
        "geometric_features_stream", {"radius": radius, "tile_size": tile_size}
    )
    return {
        _SCHEMA_VERSION_KEY: _SCHEMA_VERSION,
        _CRS_KEY: crs.to_wkt().encode("utf-8"),
        _PROVENANCE_KEY: json.dumps(provenance.to_dict()).encode("utf-8"),
    }


def _write_result(writer: Any, schema: Any, result: _TileResult) -> None:
    """Append one tile's core-point features to the output writer."""
    index_core, x_core, y_core, z_core, features = result
    if len(index_core) == 0:
        return
    columns: dict[str, NDArray[Any]] = {
        "index": index_core,
        "x": x_core,
        "y": y_core,
        "z": z_core,
    }
    columns.update(features)
    writer.write_table(pa.table(columns, schema=schema))


def geometric_features_stream(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    radius: float,
    tile_size: float,
    crs: object | None = None,
    chunk_size: int = 1_000_000,
    workers: int = 1,
) -> None:
    """Compute fixed-radius geometric features on a LAS/LAZ file out-of-core.

    Streams the file in two passes (partition to per-tile temp files, then
    per-tile feature computation) so the whole cloud is never resident. The
    output matches :func:`~geoai3d.geometric_features` with the same radius,
    bit for bit.

    Args:
        source: Path to the input ``.las`` or ``.laz`` file.
        destination: Path to the output ``.parquet`` file.
        radius: Neighbourhood radius and halo width, in the file's CRS units.
        tile_size: Edge length of the square (x, y) tiles. Choose it so the
            number of tiles is modest (dozens to a few hundred); it must be at
            least ``radius``.
        crs: CRS to assign, overriding the file header. Required if the file
            has none.
        chunk_size: Number of points read per streaming chunk in pass one.
        workers: Number of processes for the per-tile feature pass. 1 runs
            serially.

    Raises:
        ValueError: If ``radius`` or ``tile_size`` is not positive,
            ``tile_size`` is smaller than ``radius``, ``workers`` is below 1,
            or the file has no CRS and none is supplied.
    """
    if radius <= 0:
        msg = "radius must be a positive number."
        raise ValueError(msg)
    if tile_size <= 0:
        msg = "tile_size must be a positive number."
        raise ValueError(msg)
    if tile_size < radius:
        msg = "tile_size must be at least radius, or halos would span whole tiles."
        raise ValueError(msg)
    if workers < 1:
        msg = "workers must be at least 1."
        raise ValueError(msg)

    with laspy.open(str(source)) as reader:
        header = reader.header
        file_crs = header.parse_crs()
        resolved_crs = coerce_crs(crs) if crs is not None else file_crs
        if resolved_crs is None:
            msg = (
                f"{str(source)!r} has no coordinate reference system and none "
                "was supplied. Pass crs= to set it explicitly."
            )
            raise ValueError(msg)
        origin_x = float(header.mins[0])
        origin_y = float(header.mins[1])

        temp_dir = Path(tempfile.mkdtemp(prefix="geoai3d_tiles_"))
        try:
            paths = _partition(
                reader, temp_dir, origin_x, origin_y, tile_size, radius, chunk_size
            )
            out_schema = pa.schema(
                [
                    ("index", pa.int64()),
                    ("x", pa.float64()),
                    ("y", pa.float64()),
                    ("z", pa.float64()),
                    *[(name, pa.float64()) for name in FEATURE_NAMES],
                ]
            ).with_metadata(
                _output_metadata(resolved_crs, str(source), radius, tile_size)
            )
            items = [
                (str(path), tile_i, tile_j, origin_x, origin_y, tile_size, radius)
                for (tile_i, tile_j), path in paths.items()
            ]
            writer = pq.ParquetWriter(str(destination), out_schema)
            try:
                if workers > 1:
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        for result in pool.map(_tile_worker, items):
                            _write_result(writer, out_schema, result)
                else:
                    for item in items:
                        _write_result(writer, out_schema, _tile_worker(item))
            finally:
                writer.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
