"""Tests for the provenance and lineage records."""

from datetime import timezone

from geoai3d import ProcessStep, Provenance


def test_add_step_appends_and_returns() -> None:
    prov = Provenance(source="scan.laz")
    step = prov.add_step("voxel subsampling", {"voxel_size": 0.5})
    assert isinstance(step, ProcessStep)
    assert prov.steps == [step]
    assert step.description == "voxel subsampling"
    assert step.parameters == {"voxel_size": 0.5}


def test_source_is_stored() -> None:
    prov = Provenance(source="dataset-doi")
    assert prov.source == "dataset-doi"


def test_default_source_is_none() -> None:
    assert Provenance().source is None


def test_software_defaults_to_geoai3d_version() -> None:
    step = Provenance().add_step("read LAS")
    assert step.software.startswith("geoai3d ")


def test_explicit_software_is_kept() -> None:
    step = Provenance().add_step("run model", software="ptv3 1.2")
    assert step.software == "ptv3 1.2"


def test_timestamp_is_timezone_aware_utc() -> None:
    step = Provenance().add_step("read LAS")
    assert step.timestamp.tzinfo is timezone.utc


def test_parameters_are_copied_defensively() -> None:
    params = {"voxel_size": 0.5}
    step = Provenance().add_step("voxel subsampling", params)
    params["voxel_size"] = 999.0
    assert step.parameters == {"voxel_size": 0.5}


def test_copy_is_independent() -> None:
    original = Provenance(source="scan.laz")
    original.add_step("read LAS")
    duplicate = original.copy()
    duplicate.add_step("reproject")
    assert len(original.steps) == 1
    assert len(duplicate.steps) == 2
    assert duplicate.source == "scan.laz"
