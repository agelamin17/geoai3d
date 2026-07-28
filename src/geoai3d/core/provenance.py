"""Provenance and lineage records for GEOAI_3D data products.

Every operation that produces or transforms spatial data should append a
:class:`ProcessStep` to the object's :class:`Provenance`, so that an output
carries a machine-readable record of what produced it, from which inputs, with
which software and parameters. This mirrors the lineage model of ISO 19115
(geographic information metadata) and underpins the reproducibility story the
project roadmap calls for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _default_software() -> str:
    """Return the running ``geoai3d`` name and version, e.g. ``"geoai3d 0.1.0"``."""
    from geoai3d import __version__

    return f"geoai3d {__version__}"


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ProcessStep:
    """A single recorded step in a data product's processing history.

    Attributes:
        description: Human-readable summary of what the step did, for example
            ``"voxel subsampling"``.
        parameters: The parameters that controlled the step, as a plain mapping
            of names to values, for example ``{"voxel_size": 0.5}``.
        software: Name and version of the software that ran the step, for
            example ``"geoai3d 0.1.0"``.
        timestamp: When the step ran, as a timezone-aware UTC datetime.
    """

    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    software: str = field(default_factory=_default_software)
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass
class Provenance:
    """An ordered lineage record for a data product.

    A :class:`Provenance` bundles an optional description of the original data
    source with an ordered list of :class:`ProcessStep` entries, oldest first.

    Args:
        source: Description of the original input, for example a file name or a
            dataset DOI. ``None`` if unknown.
        steps: The processing steps applied so far, oldest first.

    Example:
        >>> prov = Provenance(source="scan.laz")
        >>> _ = prov.add_step("voxel subsampling", {"voxel_size": 0.5})
        >>> len(prov.steps)
        1
    """

    source: str | None = None
    steps: list[ProcessStep] = field(default_factory=list)

    def add_step(
        self,
        description: str,
        parameters: dict[str, Any] | None = None,
        software: str | None = None,
    ) -> ProcessStep:
        """Append a processing step to the lineage and return it.

        Args:
            description: Human-readable summary of the step.
            parameters: Parameters that controlled the step. A copy is stored,
                so later mutation of the passed mapping does not affect the
                record. Defaults to an empty mapping.
            software: Software name and version. Defaults to the running
                ``geoai3d`` version.

        Returns:
            The :class:`ProcessStep` that was appended.

        Example:
            >>> prov = Provenance()
            >>> step = prov.add_step("read LAS", {"path": "scan.laz"})
            >>> step.description
            'read LAS'
        """
        step = ProcessStep(
            description=description,
            parameters=dict(parameters) if parameters is not None else {},
            software=software if software is not None else _default_software(),
        )
        self.steps.append(step)
        return step

    def copy(self) -> Provenance:
        """Return an independent copy of this lineage record.

        Returns:
            A new :class:`Provenance` whose step list can be extended without
            affecting the original. Individual :class:`ProcessStep` entries are
            immutable and therefore safe to share.
        """
        return Provenance(source=self.source, steps=list(self.steps))
