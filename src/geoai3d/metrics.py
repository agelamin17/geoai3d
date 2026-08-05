"""Accuracy assessment for point classification.

Reports the standard metrics for a set of predicted labels against ground
truth: the confusion matrix, overall accuracy, and per-class precision, recall,
F1, and intersection-over-union (IoU), plus the mean IoU and macro F1 that
summarise a multi-class result. IoU is the metric the semantic-segmentation
literature reports, so it is included alongside the classical ones.

Everything here is pure NumPy, so accuracy assessment needs nothing beyond the
base install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AccuracyReport:
    """A classification accuracy assessment.

    Attributes:
        labels: The class labels, in the row/column order of ``confusion``.
        confusion: The confusion matrix, rows = truth, columns = prediction.
        overall_accuracy: Fraction of points classified correctly.
        per_class: Per-label ``precision``, ``recall``, ``f1``, ``iou``, and
            ``support`` (the number of truth points of that class).
        mean_iou: Mean of the per-class IoU values.
        macro_f1: Mean of the per-class F1 values.
    """

    labels: tuple[int, ...]
    confusion: NDArray[Any]
    overall_accuracy: float
    per_class: dict[int, dict[str, float]]
    mean_iou: float
    macro_f1: float


def confusion_matrix(
    truth: NDArray[Any],
    prediction: NDArray[Any],
    labels: NDArray[Any] | list[int] | tuple[int, ...] | None = None,
) -> tuple[NDArray[Any], tuple[int, ...]]:
    """Build a confusion matrix (rows = truth, columns = prediction).

    Args:
        truth: True integer labels, one per point.
        prediction: Predicted integer labels, one per point.
        labels: The labels to include, in order. If omitted, the sorted union of
            the values present is used. Points whose truth or prediction falls
            outside ``labels`` are ignored.

    Returns:
        The ``(k, k)`` integer confusion matrix and the tuple of labels giving
        its row and column order.

    Raises:
        ValueError: If ``truth`` and ``prediction`` have different lengths, or
            are empty.
    """
    true_labels = np.asarray(truth).ravel()
    predicted_labels = np.asarray(prediction).ravel()
    if true_labels.shape != predicted_labels.shape:
        msg = (
            f"truth and prediction must have the same length; got "
            f"{true_labels.shape[0]} and {predicted_labels.shape[0]}."
        )
        raise ValueError(msg)
    if true_labels.size == 0:
        msg = "Cannot assess accuracy on empty label arrays."
        raise ValueError(msg)

    if labels is None:
        label_values = np.unique(np.concatenate([true_labels, predicted_labels]))
    else:
        label_values = np.array(sorted({int(label) for label in labels}))
    n_labels = len(label_values)

    keep = np.isin(true_labels, label_values) & np.isin(predicted_labels, label_values)
    row = np.searchsorted(label_values, true_labels[keep])
    column = np.searchsorted(label_values, predicted_labels[keep])
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(matrix, (row, column), 1)
    return matrix, tuple(int(label) for label in label_values)


def evaluate(
    truth: NDArray[Any],
    prediction: NDArray[Any],
    *,
    labels: NDArray[Any] | list[int] | tuple[int, ...] | None = None,
) -> AccuracyReport:
    """Assess predicted labels against ground truth.

    Args:
        truth: True integer labels, one per point.
        prediction: Predicted integer labels, one per point.
        labels: The labels to include, in order. If omitted, the sorted union of
            the values present is used.

    Returns:
        An :class:`AccuracyReport` with the confusion matrix, overall accuracy,
        per-class metrics, mean IoU, and macro F1.

    Raises:
        ValueError: If ``truth`` and ``prediction`` have different lengths, or
            are empty.

    Example:
        >>> import numpy as np
        >>> from geoai3d import evaluate
        >>> truth = np.array([0, 0, 1, 1, 1])
        >>> prediction = np.array([0, 1, 1, 1, 1])
        >>> report = evaluate(truth, prediction)
        >>> round(report.overall_accuracy, 2)
        0.8
    """
    matrix, label_values = confusion_matrix(truth, prediction, labels)
    total = int(matrix.sum())
    overall_accuracy = float(np.trace(matrix)) / total if total > 0 else 0.0

    per_class: dict[int, dict[str, float]] = {}
    iou_values: list[float] = []
    f1_values: list[float] = []
    for index, label in enumerate(label_values):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum()) - true_positive
        false_negative = float(matrix[index, :].sum()) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        union = true_positive + false_positive + false_negative
        iou = true_positive / union if union > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": float(matrix[index, :].sum()),
        }
        iou_values.append(iou)
        f1_values.append(f1)

    return AccuracyReport(
        labels=label_values,
        confusion=matrix,
        overall_accuracy=overall_accuracy,
        per_class=per_class,
        mean_iou=float(np.mean(iou_values)) if iou_values else 0.0,
        macro_f1=float(np.mean(f1_values)) if f1_values else 0.0,
    )
