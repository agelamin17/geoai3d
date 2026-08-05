"""Reading and writing rasters as GeoTIFF, via rasterio.

GeoTIFF is the standard interchange format for georeferenced rasters, so DTMs,
DSMs, and difference surfaces are written and read here as single-band GeoTIFFs
carrying the CRS, affine transform, and nodata value. This is the one raster
operation that needs GDAL: it uses rasterio, which ships as the optional
``[gis]`` extra (``pip install geoai3d[gis]``). The raster maths in
:mod:`geoai3d.dem` stays pure NumPy and needs nothing extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoai3d.core.raster import Raster

if TYPE_CHECKING:
    import os

_GIS_HINT = (
    "Reading and writing GeoTIFFs needs rasterio, which is not installed. "
    "Install the GIS extra with 'pip install geoai3d[gis]'."
)


def to_geotiff(raster: Raster, destination: str | os.PathLike[str]) -> None:
    """Write a :class:`~geoai3d.Raster` to a single-band GeoTIFF.

    Args:
        raster: The raster to write. Its CRS, transform, and nodata value are
            stored in the file.
        destination: Output ``.tif`` path.

    Raises:
        ValueError: If rasterio (the ``[gis]`` extra) is not installed.

    Example:
        >>> import numpy as np
        >>> import os, tempfile
        >>> from geoai3d import Raster, to_geotiff
        >>> raster = Raster(np.ones((4, 4)), (1.0, 0.0, 0.0, 0.0, -1.0, 4.0), 28992)
        >>> to_geotiff(raster, os.path.join(tempfile.mkdtemp(), "out.tif"))
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ValueError(_GIS_HINT) from exc

    data = np.asarray(raster.data)
    with rasterio.open(
        str(destination),
        "w",
        driver="GTiff",
        height=raster.height,
        width=raster.width,
        count=1,
        dtype=str(data.dtype),
        crs=raster.crs.to_wkt(),
        transform=rasterio.Affine(*raster.transform),
        nodata=raster.nodata,
    ) as dataset:
        dataset.write(data, 1)


def read_geotiff(
    source: str | os.PathLike[str],
    *,
    band: int = 1,
) -> Raster:
    """Read one band of a GeoTIFF into a :class:`~geoai3d.Raster`.

    Args:
        source: Path to a ``.tif`` file.
        band: 1-based band index to read.

    Returns:
        A :class:`~geoai3d.Raster` with the band's values, transform, CRS, and
        nodata value.

    Raises:
        ValueError: If rasterio (the ``[gis]`` extra) is not installed, or the
            file carries no CRS.

    Example:
        >>> import numpy as np
        >>> import os, tempfile
        >>> from geoai3d import Raster, to_geotiff, read_geotiff
        >>> path = os.path.join(tempfile.mkdtemp(), "out.tif")
        >>> to_geotiff(
        ...     Raster(np.ones((4, 4)), (1.0, 0.0, 0.0, 0.0, -1.0, 4.0), 28992), path
        ... )
        >>> read_geotiff(path).shape
        (4, 4)
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ValueError(_GIS_HINT) from exc

    with rasterio.open(str(source)) as dataset:
        data = dataset.read(band)
        transform = tuple(dataset.transform)[:6]
        file_crs = dataset.crs
        nodata = dataset.nodata
    if file_crs is None:
        msg = (
            f"{str(source)!r} has no coordinate reference system, so it cannot "
            "be read as a georeferenced raster."
        )
        raise ValueError(msg)
    resolved_nodata = float("nan") if nodata is None else float(nodata)
    return Raster(data, transform, file_crs.to_wkt(), nodata=resolved_nodata)
