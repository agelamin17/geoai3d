# Releasing GEOAI_3D

The Stage-2 release lands in two steps by design: a breaking revision to PyPI
first (`0.2.0`), then — once it has settled and installed cleanly for at least
one person — a release to both PyPI and conda-forge (`0.3.0`). Doing the API
churn on PyPI before the conda-forge feedstock exists is deliberate: fixing a
mistake on PyPI is cheap, fixing it after a feedstock is live is not.

## Before any release

1. The gate is green on your machine and in CI:

   ```bash
   ruff format . && ruff check . && ruff format --check . && mypy && pytest -q
   ```

2. `CHANGELOG.md` has a dated section for the version, and the compare links at
   the bottom are updated.
3. `__version__` in `src/geoai3d/__init__.py` matches the version being released
   (the build reads it from there).

## Step 1 — `0.2.0` to PyPI (breaking revision)

```bash
# clean build
rm -rf dist
python -m build            # builds sdist + wheel from pyproject
twine check dist/*         # metadata sanity

# publish
twine upload dist/*        # needs a PyPI API token (or a trusted publisher)

# tag the release
git tag -a v0.2.0 -m "0.2.0"
git push origin v0.2.0
```

Then create a GitHub Release from the `v0.2.0` tag, pasting the CHANGELOG
section. Install it fresh in a clean environment to confirm it works:

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install "geoai3d[laz,viz,gis]"
/tmp/check/bin/python -c "import geoai3d; print(geoai3d.__version__)"
```

Ideally get one other person to install it too. Leave it on PyPI for a few days
and fix anything that surfaces (each fix is a `0.2.x` patch release).

## Step 2 — `0.3.0` to PyPI and conda-forge

When `0.2.0` has settled:

1. Bump `__version__` to `0.3.0`, move the CHANGELOG `Unreleased` heading to
   `0.3.0` with today's date, update the compare links, commit.
2. Repeat the PyPI build/upload/tag from Step 1 with `v0.3.0`.
3. Get the sdist SHA-256 (conda-forge pins it):

   ```bash
   python -c "import hashlib,urllib.request; \
     u='https://pypi.org/packages/source/g/geoai3d/geoai3d-0.3.0.tar.gz'; \
     print(hashlib.sha256(urllib.request.urlopen(u).read()).hexdigest())"
   ```

4. Submit to conda-forge:
   - Fork <https://github.com/conda-forge/staged-recipes>.
   - Copy `conda-recipe/meta.yaml` from this repo to
     `recipes/geoai3d/meta.yaml` in the fork.
   - Set `version` to `0.3.0` and paste the SHA-256 into `source.sha256`.
   - Open a pull request. The conda-forge CI builds and lints the recipe;
     address any review comments.
   - On merge, a `geoai3d-feedstock` repository is created and the package
     appears on conda-forge. From then on, a bot opens an update PR
     automatically whenever a new version is published to PyPI.

## GitHub Pages (documentation site)

The `Docs` workflow builds and deploys the MkDocs site to GitHub Pages on every
push to `main`. Enable it once, in the repository settings:

> Settings → Pages → Build and deployment → Source: **GitHub Actions**

The site then publishes to `https://agelamin17.github.io/geoai3d/`.
