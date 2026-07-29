"""Internal helpers for producing derived point clouds with lineage.

Private module. Feature and filtering operations either add attribute columns to
the same points or select a subset of points; both must carry the CRS through
and append a provenance step. These two helpers keep that bookkeeping in one
place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoai3d.core.pointcloud import PointCloud
from geoai3d.core.provenance import Provenance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def attach_attributes(
    cloud: PointCloud,
    new_attributes: dict[str, NDArray[Any]],
    description: str,
    parameters: dict[str, Any],
) -> PointCloud:
    """Return a new cloud with extra attribute columns and a provenance step.

    Coordinates and existing attributes are carried through; the CRS is
    preserved and a step recording the operation is appended to the lineage.

    Args:
        cloud: The source cloud.
        new_attributes: Attribute columns to add (or replace), each of length
            ``len(cloud)``.
        description: Provenance step description.
        parameters: Provenance step parameters.

    Returns:
        The derived :class:`PointCloud`.
    """
    attributes: dict[str, NDArray[Any]] = {
        name: cloud.attribute(name) for name in cloud.attribute_names
    }
    attributes.update(new_attributes)
    provenance = (
        cloud.provenance.copy() if cloud.provenance is not None else Provenance()
    )
    provenance.add_step(description, parameters)
    return PointCloud(
        cloud.xyz, attributes=attributes, crs=cloud.crs, provenance=provenance
    )


def select(
    cloud: PointCloud,
    indices: NDArray[Any],
    description: str,
    parameters: dict[str, Any],
) -> PointCloud:
    """Return the selected points as a new cloud with a provenance step.

    Args:
        cloud: The source cloud.
        indices: Integer indices (or a boolean mask) of the points to keep.
        description: Provenance step description.
        parameters: Provenance step parameters.

    Returns:
        The subset :class:`PointCloud`, with the CRS carried through.
    """
    subset = cloud[indices]
    provenance = (
        subset.provenance.copy() if subset.provenance is not None else Provenance()
    )
    provenance.add_step(description, parameters)
    attributes: dict[str, NDArray[Any]] = {
        name: subset.attribute(name) for name in subset.attribute_names
    }
    return PointCloud(
        subset.xyz, attributes=attributes, crs=subset.crs, provenance=provenance
    )
