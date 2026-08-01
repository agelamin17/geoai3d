"""Benchmark: out-of-core geometric features on a large LAS/LAZ file.

Runs geoai3d's streaming feature engine on a file and reports point count,
wall-clock time, throughput, and peak memory (RSS), to check that a cloud is
processed within a stated memory ceiling.

Peak memory is sampled across this process and any worker processes, so the
number is the true footprint even with --workers > 1. Needs psutil:

    pip install psutil

Usage:

    python benchmarks/benchmark.py INPUT.laz --radius 1.0 --tile-size 100
    python benchmarks/benchmark.py INPUT.laz --radius 1.0 --tile-size 100 --workers 4

For AHN (~10 points/m^2), radius 1.0 m captures ~30 neighbours; a tile size of
50-100 m keeps the tile count modest.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import laspy
import psutil

from geoai3d import geometric_features_stream


class PeakMemory:
    """Context manager sampling peak RSS of this process plus its children."""

    def __init__(self, interval: float = 0.1) -> None:
        self._process = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak_bytes = 0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _current(self) -> int:
        total = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._current())
            self._stop.wait(self._interval)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--tile-size", type=float, required=True, dest="tile_size")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ceiling", type=float, default=16.0, help="Memory target (GB).")
    parser.add_argument("--verbose", action="store_true", help="Show pass timings.")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_suffix(".features.parquet")

    with laspy.open(str(source)) as reader:
        n_points = reader.header.point_count
    size_gb = source.stat().st_size / (1024**3)
    naive_gb = n_points * 32 / (1024**3)

    print(f"input:        {source.name}  ({size_gb:.2f} GB on disk)")
    print(f"points:       {n_points:,}")
    print(f"radius:       {args.radius}   tile_size: {args.tile_size}   workers: {args.workers}")
    print(f"in-memory copy would need roughly ~{naive_gb:.1f} GB just for the columns")
    print("running streaming feature computation ...\n")

    start = time.perf_counter()
    with PeakMemory() as memory:
        geometric_features_stream(
            source,
            output,
            radius=args.radius,
            tile_size=args.tile_size,
            workers=args.workers,
            verbose=args.verbose,
        )
    elapsed = time.perf_counter() - start

    print("--- result ---")
    print(f"output:       {output.name}")
    print(f"wall time:    {elapsed:.1f} s")
    print(f"throughput:   {n_points / elapsed / 1e6:.2f} million points/s")
    print(f"PEAK MEMORY:  {memory.peak_gb:.2f} GB   (ceiling: {args.ceiling:.0f} GB)")
    verdict = "PASS" if memory.peak_gb <= args.ceiling else "OVER CEILING"
    print(f"ceiling:      {verdict}")


if __name__ == "__main__":
    main()
