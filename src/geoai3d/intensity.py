"""Radiometric normalisation of LiDAR intensity.

Raw LiDAR intensity is not a stable surface property: it falls off with the
square of the sensor-to-target range and with the cosine of the incidence angle
between the beam and the surface. Feeding uncalibrated intensity into a
classifier is the most common way the field misuses it. This module applies the
standard data-driven corrections (after Hofle & Pfeifer, 2007) so intensity
better reflects target reflectance and is comparable across a survey.

Both corrections follow from the LiDAR range equation, in which the received
power is proportional to ``rho * cos(theta) / R ** 2`` for a diffuse (Lambertian)
target of reflectance ``rho`` at range ``R`` and incidence angle ``theta``:

* Range: multiply by ``(R / R_ref) ** 2`` to undo the ``1 / R ** 2`` falloff,
  where ``R_ref`` is a reference range (by default the median range of the
  cloud), so a corrected value is what the sensor would have recorded at
  ``R_ref``.
* Incidence angle: divide by ``cos(theta)`` to undo the obliquity falloff,
  where ``theta`` is the angle between the beam and the surface normal. Grazing
  angles are clamped (``max_incidence_angle``) so the correction cannot blow up.

Both need the sensor position, which LAS does not store, so it is passed
explicitly and must be in the same coordinate reference system as the cloud.
A single ``(3,)`` origin suits a terrestrial scan; a per-point ``(N, 3)``
trajectory suits an airborne or mobile survey.

Reference:
    Hofle, B. & Pfeifer, N. (2007). Correction of laser scanning intensity
    data: Data and model-driven approaches. ISPRS Journal of Photogrammetry
    and Remote Sensing, 62(6), 415-433.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from geoai3d.core._derive import attach_attributes
from geoai3d.core.pointcloud import PointCloud

if TYPE_CHECKING:
    from numpy.typing import NDArray

_NORMAL_ATTRIBUTES = ("nx", "ny", "nz")


def normalize_intensity(
    cloud: PointCloud,
    *,
    sensor_position: NDArray[Any] | Sequence[float] | Sequence[Sequence[float]],
    intensity_attribute: str = "intensity",
    reference_range: float | None = None,
    correct_range: bool = True,
    correct_incidence_angle: bool = False,
    max_incidence_angle: float = 80.0,
    output_attribute: str = "normalized_intensity",
) -> PointCloud:
    """Normalise LiDAR intensity for range and, optionally, incidence angle.

    Corrects intensity so it better reflects target reflectance and is
    comparable across a survey, using the sensor geometry (see the module
    docstring for the model). The result is added as a new ``float64`` column
    and the CRS and provenance are carried through.

    Args:
        cloud: The cloud to normalise. Must carry the intensity attribute and,
            for incidence-angle correction, surface normals (``nx``, ``ny``,
            ``nz`` from :func:`~geoai3d.estimate_normals`).
        sensor_position: The scanner position, in the cloud's CRS. A ``(3,)``
            array for a single origin (terrestrial), or an ``(N, 3)`` array of
            per-point positions (airborne or mobile trajectory).
        intensity_attribute: Name of the raw intensity attribute to read.
        reference_range: Range ``R_ref`` the range correction normalises to. If
            omitted, the median sensor-to-point range of the cloud is used.
        correct_range: If true, apply the ``(R / R_ref) ** 2`` range correction.
        correct_incidence_angle: If true, also divide by ``cos(theta)`` using
            the surface normals. Off by default because it needs normals.
        max_incidence_angle: Grazing-angle clamp in degrees for the incidence
            correction; incidence angles beyond it are treated as this angle so
            the ``1 / cos(theta)`` factor stays bounded.
        output_attribute: Name of the normalised intensity column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the normalised intensity column
        added.

    Raises:
        ValueError: If neither correction is enabled; if the intensity or (for
            incidence correction) the normal attributes are missing;
            if ``sensor_position`` is not ``(3,)`` or ``(N, 3)``; if any point
            coincides with the sensor (zero range); if ``reference_range`` is
            not positive; or if ``max_incidence_angle`` is not in ``(0, 90)``.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, normalize_intensity
        >>> pts = np.zeros((100, 3))
        >>> pts[:, 0] = np.linspace(0.0, 10.0, 100)
        >>> cloud = PointCloud(
        ...     pts, attributes={"intensity": np.full(100, 100.0)}, crs=28992
        ... )
        >>> out = normalize_intensity(cloud, sensor_position=[5.0, 0.0, 50.0])
        >>> "normalized_intensity" in out.attribute_names
        True
    """
    if not correct_range and not correct_incidence_angle:
        msg = (
            "Nothing to normalise: enable correct_range and/or correct_incidence_angle."
        )
        raise ValueError(msg)
    if intensity_attribute not in cloud.attribute_names:
        msg = (
            f"Cloud has no {intensity_attribute!r} attribute to normalise. "
            f"Available attributes: {cloud.attribute_names}. Pass "
            "intensity_attribute= if it is stored under another name."
        )
        raise ValueError(msg)

    n_points = len(cloud)
    sensor = np.asarray(sensor_position, dtype=np.float64)
    if sensor.shape == (3,):
        sensor = np.broadcast_to(sensor, (n_points, 3))
    elif sensor.shape != (n_points, 3):
        msg = (
            "sensor_position must have shape (3,) for a single scanner origin "
            f"or ({n_points}, 3) for a per-point trajectory; got {sensor.shape}."
        )
        raise ValueError(msg)

    beam = cloud.xyz - sensor
    ranges = np.linalg.norm(beam, axis=1)
    if np.any(ranges <= 0.0):
        msg = (
            "Some points coincide with the sensor position (zero range), so "
            "intensity cannot be normalised there. Check sensor_position."
        )
        raise ValueError(msg)

    if reference_range is None:
        resolved_reference = float(np.median(ranges))
    elif reference_range <= 0.0:
        msg = "reference_range must be a positive distance."
        raise ValueError(msg)
    else:
        resolved_reference = float(reference_range)

    corrected = cloud.attribute(intensity_attribute).astype(np.float64)
    if correct_range:
        corrected = corrected * (ranges / resolved_reference) ** 2

    if correct_incidence_angle:
        if not 0.0 < max_incidence_angle < 90.0:
            msg = "max_incidence_angle must be between 0 and 90 degrees."
            raise ValueError(msg)
        missing = [n for n in _NORMAL_ATTRIBUTES if n not in cloud.attribute_names]
        if missing:
            msg = (
                "Incidence-angle correction needs surface normals "
                f"{list(_NORMAL_ATTRIBUTES)}; missing {missing}. Run "
                "estimate_normals on the cloud first."
            )
            raise ValueError(msg)
        normals = np.column_stack(
            [cloud.attribute(name).astype(np.float64) for name in _NORMAL_ATTRIBUTES]
        )
        normal_lengths = np.linalg.norm(normals, axis=1)
        normal_lengths = np.where(normal_lengths == 0.0, 1.0, normal_lengths)
        unit_normals = normals / normal_lengths[:, None]
        unit_beam = beam / ranges[:, None]
        # abs(): a surface normal's sign is arbitrary, so fold the beam onto the
        # front face rather than letting an upward normal give a negative cosine.
        cos_incidence = np.abs(np.einsum("ij,ij->i", unit_beam, unit_normals))
        cos_floor = float(np.cos(np.radians(max_incidence_angle)))
        cos_incidence = np.maximum(cos_incidence, cos_floor)
        corrected = corrected / cos_incidence

    return attach_attributes(
        cloud,
        {output_attribute: corrected},
        "normalize_intensity",
        {
            "intensity_attribute": intensity_attribute,
            "reference_range": resolved_reference,
            "correct_range": correct_range,
            "correct_incidence_angle": correct_incidence_angle,
            "max_incidence_angle": (
                max_incidence_angle if correct_incidence_angle else None
            ),
            "sensor_position": (
                "per_point" if np.asarray(sensor_position).ndim == 2 else "origin"
            ),
        },
    )
