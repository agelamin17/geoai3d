"""Tests for accuracy assessment."""

import numpy as np
import pytest

from geoai3d import confusion_matrix, evaluate


def test_confusion_matrix_counts() -> None:
    truth = np.array([0] * 10 + [1] * 5)
    prediction = np.array([0] * 8 + [1] * 2 + [1] * 5)
    matrix, labels = confusion_matrix(truth, prediction)
    assert labels == (0, 1)
    np.testing.assert_array_equal(matrix, np.array([[8, 2], [0, 5]]))


def test_evaluate_matches_hand_computation() -> None:
    truth = np.array([0] * 10 + [1] * 5)
    prediction = np.array([0] * 8 + [1] * 2 + [1] * 5)
    report = evaluate(truth, prediction)
    assert report.overall_accuracy == pytest.approx(13 / 15)
    assert report.per_class[0]["recall"] == pytest.approx(0.8)
    assert report.per_class[0]["iou"] == pytest.approx(0.8)
    assert report.per_class[1]["precision"] == pytest.approx(5 / 7)
    assert report.per_class[0]["support"] == pytest.approx(10.0)
    assert report.mean_iou == pytest.approx((0.8 + 5 / 7) / 2)


def test_evaluate_perfect_prediction() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    report = evaluate(labels, labels.copy())
    assert report.overall_accuracy == pytest.approx(1.0)
    assert report.mean_iou == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)


def test_labels_subset_ignores_other_classes() -> None:
    truth = np.array([0, 1, 2, 0])
    prediction = np.array([0, 1, 1, 0])
    report = evaluate(truth, prediction, labels=[0, 1])
    assert report.labels == (0, 1)
    assert int(report.confusion.sum()) == 3  # the class-2 point is dropped


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(np.array([0, 1]), np.array([0]))


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty label arrays"):
        evaluate(np.array([], dtype=int), np.array([], dtype=int))
