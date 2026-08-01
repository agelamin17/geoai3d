"""Stream-merge several LAS/LAZ files into one, keeping only coordinates.

Builds a single large cloud from adjacent tiles (for example several AHN
sub-tiles) so the out-of-core benchmark has a 100M+ point file to chew on.
Coordinates are re-quantised to a common origin; only x/y/z are carried, since
the benchmark computes features from coordinates. Memory stays bounded because
the inputs are streamed in chunks.

Usage:
    python benchmarks/merge_laz.py merged.laz tile1.laz tile2.laz [tile3.laz ...]
"""

from __future__ import annotations

import sys

import laspy
import numpy as np


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python benchmarks/merge_laz.py OUT.laz IN1.laz IN2.laz ...")
        raise SystemExit(2)
    output = sys.argv[1]
    inputs = sys.argv[2:]

    mins = np.array([np.inf, np.inf, np.inf])
    crs = None
    for path in inputs:
        with laspy.open(path) as reader:
            mins = np.minimum(mins, np.asarray(reader.header.mins))
            if crs is None:
                crs = reader.header.parse_crs()

    header = laspy.LasHeader(version="1.4", point_format=0)
    header.offsets = np.floor(mins)
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs is not None:
        header.add_crs(crs)

    total = 0
    with laspy.open(output, mode="w", header=header) as writer:
        for path in inputs:
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(2_000_000):
                    record = laspy.ScaleAwarePointRecord.zeros(
                        len(chunk), header=header
                    )
                    record.x = chunk.x
                    record.y = chunk.y
                    record.z = chunk.z
                    writer.write_points(record)
                    total += len(chunk)
            print(f"merged {path} (running total {total:,} points)")
    print(f"wrote {output} with {total:,} points")


if __name__ == "__main__":
    main()
