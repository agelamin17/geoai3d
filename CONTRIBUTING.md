# Contributing to GEOAI_3D

Contributions are welcome at any scale, from a typo fix to a new module. This
document covers how to work on the code and the few rules that are not negotiable.

## Ways to help that are not code

The project needs these at least as much as it needs pull requests:

- **Test data.** Point clouds with known, independently surveyed control. This is
  the scarcest resource in the project.
- **Workflow reports.** Tell us what you do with 3D geospatial data and where the
  current tooling hurts. Open an issue with the `workflow` label.
- **Documentation and tutorials**, especially from a discipline we have not thought
  about.

## Development setup

```bash
git clone https://github.com/agelamin17/geoai3d.git
cd geoai3d
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Run the checks locally before pushing:

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not gpu"
```

## Code conventions

**API design.** The common case takes one to five lines. Expose a well-named
high-level function with sensible defaults, and always provide an escape hatch to
the underlying object for advanced users.

```python
cloud = geoai3d.read_lidar("scan.laz")
labels = geoai3d.segment(cloud, model="ptv3-outdoor")
geoai3d.to_geopackage(labels, "trees.gpkg")
```

**Naming.** Modules are lowercase, singular, and domain-named (`lidar`, `sfm`,
`splat`, `annotate`, `viz`). Readers and writers are `read_*` and `to_*`, not
`load_*` and `save_*`. Booleans are affirmative: `include_ground`, not
`skip_ground`. Public names are never abbreviated: `point_cloud`, not `pc`.

**Docstrings.** Google style, on every public function: one-line summary, Args,
Returns, Raises, and a runnable Example. Examples must use data the package can
fetch automatically, so they are copy-pasteable.

**Type hints** on everything public. `mypy --strict` must pass.

**Georeferencing.** Any function returning spatial data returns it with a CRS
attached. If an input lacks one, raise an error naming the parameter that would fix
it. Never assume a default CRS, and never silently mix ellipsoidal and orthometric
heights.

**Devices.** Auto-detect with `device: str | None = None`. Never hard-code `.cuda()`.

**Errors.** Specific exceptions with actionable messages: what was wrong, what was
expected, how to fix it. No bare `except:`, no `assert` for validation.

## Testing

pytest. Every public function gets a test. Fixture point clouds live in the repo,
under 1 MB each. Unit tests never touch the network. Mark GPU-dependent tests with
`@pytest.mark.gpu` and slow ones with `@pytest.mark.slow` so CI can select.

## Dependencies

The base install must succeed on a CPU-only machine with no compiler — CI enforces
this in a bare container. Anything requiring CUDA, a build toolchain, or a
platform-specific binary goes in an optional extra. Adding a base dependency needs
justification in the pull request.

## Licensing

This is the one area where a mistake is expensive and hard to undo.

GEOAI_3D is MIT. Contributions must be compatible with that.

- **Do not contribute code you do not have the right to relicense**, including code
  from paid courses or commercial training material. Capabilities may be
  reimplemented from published literature; source may not be copied. If you learned
  a technique from a proprietary course, implement it from the underlying paper and
  cite the paper.
- **Copyleft dependencies are not acceptable in the core.** The known trap:
  OpenDroneMap's fork of OpenSfM is AGPLv3 and incompatible with an MIT core. Use
  Mapillary's BSD-licensed OpenSfM or COLMAP instead.
- **Verify model weights separately from model code.** A permissively licensed
  architecture may ship weights under a restrictive licence. Both need checking.
- **Check dataset redistribution terms** before adding a loader that downloads data.

If you are unsure, open an issue before writing the code.

## AI assistance

Using AI tools to help write contributions is fine. Disclose it in the pull request
description — which tools, and for what. You remain responsible for correctness,
originality, and licensing of anything you submit. See [AI_USAGE.md](AI_USAGE.md).

## Pull requests

Keep them focused. Reference an issue where one exists. Update the CHANGELOG under
`Unreleased`. CI must be green.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
