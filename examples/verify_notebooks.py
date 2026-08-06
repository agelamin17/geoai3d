"""Run every tutorial notebook's pipeline as a plain script.

This proves the notebook code works against the real sample without needing
Jupyter. Install first::

    pip install -e ".[dev,laz,viz,gis]"

then run from the repo root::

    python examples/verify_notebooks.py

It prints the key results of each notebook (notably notebook 4's
spatially-blocked accuracy). Outputs are written to a temporary folder.
"""

import tempfile
from pathlib import Path

import numpy as np

from geoai3d import (
    classify,
    difference,
    evaluate,
    geometric_features,
    ground,
    read_lidar,
    reproject,
    segment,
    spatial_block_split,
    to_dsm,
    to_dtm,
    to_geopackage,
    to_geotiff,
    to_parquet,
    train_classifier,
    volume,
)

SAMPLE = Path(__file__).parent / "data" / "ahn_sample.laz"
OUT = Path(tempfile.mkdtemp(prefix="geoai3d_notebooks_"))


def notebook_1() -> None:
    print("\n=== 1. Getting started ===")
    cloud = read_lidar(SAMPLE)
    print(f"{len(cloud):,} points, CRS EPSG:{cloud.crs.to_epsg()}")
    labels, counts = np.unique(cloud.attribute("classification"), return_counts=True)
    print("classes:", {int(v): f"{100 * c / len(cloud):.1f}%" for v, c in zip(labels, counts)})
    print("WGS84 bounds:", tuple(round(b, 4) for b in reproject(cloud, 4326).bounds))
    features = geometric_features(cloud, radius=1.0)
    classes = features.attribute("classification")
    planarity = features.attribute("planarity")
    for name, code in [("ground", 2), ("building", 6), ("veg", 1)]:
        median = float(np.nanmedian(planarity[classes == code]))
        print(f"median planarity {name:>8}: {median:.3f}")
    try:
        from geoai3d import view

        view(features, color_by="planarity")
        print("view() built a figure OK")
    except ImportError:
        print("view() skipped (install the viz extra to use it)")
    to_parquet(features, OUT / "features.parquet")
    print("saved features.parquet")


def notebook_2() -> None:
    print("\n=== 2. Terrain and volumes ===")
    cloud = read_lidar(SAMPLE)
    classified = ground(cloud, cloth_resolution=1.0, class_threshold=0.5)
    print(f"ground: {100 * classified.attribute('is_ground').mean():.1f}%")
    dtm = to_dtm(classified, resolution=0.5)
    dsm = to_dsm(classified, resolution=0.5)
    ndsm = difference(dsm, dtm)
    print("DTM grid:", dtm.shape, "| tallest object:", round(float(np.nanmax(ndsm.data)), 1), "m")
    print("above-ground volume (m^3):", round(volume(ndsm)["fill"], 1))
    to_geotiff(dtm, OUT / "dtm.tif")
    to_geotiff(ndsm, OUT / "object_heights.tif")
    print("wrote dtm.tif and object_heights.tif")
    try:
        import plotly.express as px

        px.imshow(dtm.data, title="DTM")
        px.imshow(ndsm.data, title="Height above ground")
        print("raster previews built OK")
    except ImportError:
        print("raster previews skipped (install the viz extra)")


def notebook_3() -> None:
    print("\n=== 3. Unsupervised segmentation ===")
    cloud = read_lidar(SAMPLE)
    labelled = segment(
        cloud,
        ground_resolution=1.0,
        normal_k=16,
        smoothness_degrees=15.0,
        min_region_size=100,
    )
    segments = labelled.attribute("segment")
    object_ids = np.unique(segments[segments > 0])
    print(f"ground {100 * (segments == 0).mean():.1f}% | "
          f"{len(object_ids)} object segments | "
          f"{100 * (segments == -1).mean():.1f}% unassigned")


def notebook_4() -> None:
    print("\n=== 4. Classification and GIS export ===")
    cloud = read_lidar(SAMPLE)
    features = geometric_features(cloud, radius=1.0)

    classified = ground(cloud, cloth_resolution=1.0)
    dtm = to_dtm(classified, resolution=1.0)
    a, _, c, _, e, f = dtm.transform
    col = np.clip(((cloud.xyz[:, 0] - c) / a).astype(int), 0, dtm.width - 1)
    row = np.clip(((cloud.xyz[:, 1] - f) / e).astype(int), 0, dtm.height - 1)
    height = cloud.xyz[:, 2] - dtm.data[row, col]
    height = np.where(np.isnan(height), 0.0, height)
    features = features.with_attribute("height_above_ground", height)

    feature_names = [
        "planarity", "linearity", "sphericity", "verticality", "height_above_ground"
    ]
    truths, predictions = [], []
    for train_idx, test_idx in spatial_block_split(features, block_size=25.0, n_folds=5):
        model = train_classifier(
            features[train_idx], label_attribute="classification", feature_names=feature_names
        )
        predicted = classify(features[test_idx], model=model)
        truths.append(features[test_idx].attribute("classification"))
        predictions.append(predicted.attribute("prediction"))
    report = evaluate(np.concatenate(truths), np.concatenate(predictions))
    print(f"blocked-CV overall accuracy: {report.overall_accuracy:.3f}")
    print(f"blocked-CV mean IoU        : {report.mean_iou:.3f}")
    names = {1: "vegetation", 2: "ground", 6: "building", 9: "water"}
    for label in report.labels:
        m = report.per_class[label]
        print(f"  {names.get(label, label):>10}: IoU {m['iou']:.2f}  n={int(m['support']):,}")

    model = train_classifier(
        features, label_attribute="classification", feature_names=feature_names
    )
    result = classify(features, model=model)
    to_geopackage(
        result,
        OUT / "classified.gpkg",
        attributes=["prediction", "classification", "height_above_ground"],
    )
    print("wrote classified.gpkg")
    try:
        import plotly.express as px

        from geoai3d import view

        view(result, color_by="prediction")
        ticks = [names.get(label, str(label)) for label in report.labels]
        px.imshow(report.confusion, x=ticks, y=ticks, text_auto=True)
        print("classification plots built OK")
    except ImportError:
        print("classification plots skipped (install the viz extra)")


if __name__ == "__main__":
    if not SAMPLE.exists():
        raise SystemExit(f"Sample not found at {SAMPLE}. Create it first (see data/README.md).")
    notebook_1()
    notebook_2()
    notebook_3()
    notebook_4()
    print(f"\nAll four notebook pipelines ran. Outputs in {OUT}")
