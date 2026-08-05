"""Classical point-wise classification on geometric features.

A random forest on per-point features (eigenvalue descriptors, verticality,
normalised intensity, and so on) is a fast, CPU-only baseline for semantic
classification that needs no GPU and little tuning. On aerial LiDAR it is often
competitive with far heavier deep models, which makes it the right first rung.

The workflow is: assemble a feature matrix from named attributes
(:func:`feature_matrix`), train a forest on a labelled cloud
(:func:`train_classifier`), then apply the resulting :class:`Classifier` to new
clouds (:func:`classify`). A classifier remembers the feature columns it was
trained on, so applying it uses the same columns in the same order.
:func:`save_classifier` and :func:`load_classifier` persist a trained model.

scikit-learn provides the forest; it is a base dependency, imported lazily so it
loads only when a classifier is actually built.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.core._derive import attach_attributes
from geoai3d.core._neighborhood import FEATURE_NAMES

if TYPE_CHECKING:
    import os

    from geoai3d.core.pointcloud import PointCloud

# Attributes tried, in order, when the caller does not name features explicitly.
_DEFAULT_FEATURE_CANDIDATES = (*FEATURE_NAMES, "normalized_intensity")


@dataclass(frozen=True)
class Classifier:
    """A trained classifier and the feature columns it expects.

    Attributes:
        estimator: The fitted scikit-learn estimator.
        feature_names: The attribute names used as features, in order.
    """

    estimator: Any
    feature_names: tuple[str, ...]


def _resolve_features(
    cloud: PointCloud, feature_names: tuple[str, ...] | list[str] | None
) -> list[str]:
    """Resolve which attribute columns to use as features."""
    if feature_names is None:
        available = cloud.attribute_names
        present = [name for name in _DEFAULT_FEATURE_CANDIDATES if name in available]
        if not present:
            msg = (
                "No feature columns found. Run geometric_features first, or pass "
                "feature_names explicitly."
            )
            raise ValueError(msg)
        return present
    names = list(feature_names)
    missing = [name for name in names if name not in cloud.attribute_names]
    if missing:
        msg = (
            f"Feature attributes {missing} are not present. Available: "
            f"{cloud.attribute_names}."
        )
        raise ValueError(msg)
    return names


def feature_matrix(
    cloud: PointCloud,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> NDArray[Any]:
    """Assemble a ``(n_points, n_features)`` matrix from named attributes.

    Args:
        cloud: The cloud to read features from.
        feature_names: Attribute names to stack, in order. If omitted, the
            geometric feature columns (and ``normalized_intensity``) present on
            the cloud are used.

    Returns:
        A ``float64`` feature matrix.

    Raises:
        ValueError: If no features can be resolved, or a named attribute is
            absent.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, feature_matrix
        >>> cloud = PointCloud(
        ...     np.zeros((4, 3)),
        ...     attributes={"planarity": np.ones(4), "linearity": np.zeros(4)},
        ...     crs=28992,
        ... )
        >>> feature_matrix(cloud, ["planarity", "linearity"]).shape
        (4, 2)
    """
    names = _resolve_features(cloud, feature_names)
    columns = [cloud.attribute(name).astype(np.float64) for name in names]
    return np.asarray(np.column_stack(columns), dtype=np.float64)


def train_classifier(
    cloud: PointCloud,
    *,
    label_attribute: str = "classification",
    feature_names: tuple[str, ...] | list[str] | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 0,
) -> Classifier:
    """Train a random-forest classifier on a labelled cloud.

    Args:
        cloud: The labelled training cloud.
        label_attribute: Attribute holding the integer class label per point.
        feature_names: Feature attributes to train on. If omitted, resolved as
            in :func:`feature_matrix`.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth, or ``None`` for unlimited.
        random_state: Seed for reproducible training.

    Returns:
        A :class:`Classifier` wrapping the fitted forest and its feature names.

    Raises:
        ValueError: If the label attribute is absent, or features cannot be
            resolved.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, train_classifier, classify
        >>> rng = np.random.default_rng(0)
        >>> planarity = np.concatenate([rng.normal(0.9, 0.05, 100),
        ...                             rng.normal(0.1, 0.05, 100)])
        >>> labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
        >>> cloud = PointCloud(
        ...     np.zeros((200, 3)),
        ...     attributes={"planarity": planarity, "classification": labels},
        ...     crs=28992,
        ... )
        >>> model = train_classifier(cloud, feature_names=["planarity"])
        >>> model.feature_names
        ('planarity',)
    """
    if label_attribute not in cloud.attribute_names:
        msg = (
            f"Cloud has no {label_attribute!r} label attribute to train on. "
            "Pass label_attribute= to name the training labels."
        )
        raise ValueError(msg)
    names = _resolve_features(cloud, feature_names)
    features = feature_matrix(cloud, names)
    labels = np.asarray(cloud.attribute(label_attribute))

    from sklearn.ensemble import RandomForestClassifier

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    estimator.fit(features, labels)
    return Classifier(estimator=estimator, feature_names=tuple(names))


def classify(
    cloud: PointCloud,
    *,
    model: Classifier,
    output_attribute: str = "prediction",
) -> PointCloud:
    """Apply a trained classifier to a cloud.

    Uses the same feature columns the model was trained on and adds the
    predicted class as a new column, leaving any existing ``classification``
    field untouched unless ``output_attribute`` names it.

    Args:
        cloud: The cloud to classify. Must carry the model's feature columns.
        model: A :class:`Classifier` from :func:`train_classifier` or
            :func:`load_classifier`.
        output_attribute: Name of the predicted-class column to add.

    Returns:
        A new :class:`~geoai3d.PointCloud` with the prediction column added, the
        CRS and provenance carried through.

    Raises:
        ValueError: If a feature column the model expects is absent.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud, train_classifier, classify
        >>> rng = np.random.default_rng(0)
        >>> planarity = np.concatenate([rng.normal(0.9, 0.05, 100),
        ...                             rng.normal(0.1, 0.05, 100)])
        >>> labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
        >>> cloud = PointCloud(
        ...     np.zeros((200, 3)),
        ...     attributes={"planarity": planarity, "classification": labels},
        ...     crs=28992,
        ... )
        >>> model = train_classifier(cloud, feature_names=["planarity"])
        >>> "prediction" in classify(cloud, model=model).attribute_names
        True
    """
    features = feature_matrix(cloud, model.feature_names)
    predictions = np.asarray(model.estimator.predict(features))
    return attach_attributes(
        cloud,
        {output_attribute: predictions},
        "classify",
        {"feature_names": list(model.feature_names)},
    )


def save_classifier(model: Classifier, destination: str | os.PathLike[str]) -> None:
    """Persist a trained classifier to disk (pickle).

    Args:
        model: The classifier to save.
        destination: Output path.
    """
    with open(destination, "wb") as handle:
        pickle.dump(model, handle)


def load_classifier(source: str | os.PathLike[str]) -> Classifier:
    """Load a classifier saved by :func:`save_classifier`.

    Only load models from a source you trust: the file is unpickled.

    Args:
        source: Path to a saved classifier.

    Returns:
        The loaded :class:`Classifier`.

    Raises:
        ValueError: If the file does not contain a GEOAI_3D classifier.
    """
    with open(source, "rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, Classifier):
        msg = f"{str(source)!r} does not contain a GEOAI_3D Classifier."
        raise ValueError(msg)
    return loaded
