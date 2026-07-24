"""Validation analytics: box IoU + confusion matrix (numpy-only core)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from dfine.train.metrics import ConfusionMatrix, box_iou  # noqa: E402


def test_box_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[20, 20, 30, 30]], dtype=float)
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)
    # half-overlap: [0,0,10,10] vs [5,0,15,10] → inter 50, union 150 → 1/3
    c = np.array([[5, 0, 15, 10]], dtype=float)
    assert box_iou(a, c)[0, 0] == pytest.approx(1 / 3)


def test_box_iou_empty():
    assert box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1.0]])).shape == (0, 1)


def test_confusion_matrix_tp_fp_fn_and_misclass():
    cm = ConfusionMatrix(num_classes=3, conf=0.25, iou_thresh=0.5)
    box = [0, 0, 10, 10]
    far = [100, 100, 110, 110]
    # GT: class 0 @ box, class 1 @ far.  Preds: class 0 @ box (TP), class 2 spurious (FP),
    # class 1 @ far is *missing* → FN.
    cm.process_batch(
        det_boxes=np.array([box, [50, 50, 60, 60]], float),
        det_scores=np.array([0.9, 0.9]),
        det_classes=np.array([0, 2]),
        gt_boxes=np.array([box, far], float),
        gt_classes=np.array([0, 1]),
    )
    m = cm.matrix
    bg = 3
    assert m[0, 0] == 1  # true positive (pred 0 ↔ gt 0)
    assert m[2, bg] == 1  # false positive (pred 2, no GT)
    assert m[bg, 1] == 1  # false negative (gt 1 missed)
    assert m.sum() == 3


def test_confusion_matrix_misclassification_goes_off_diagonal():
    cm = ConfusionMatrix(num_classes=2, iou_thresh=0.5)
    box = [0, 0, 10, 10]
    cm.process_batch(
        det_boxes=np.array([box], float),
        det_scores=np.array([0.9]),
        det_classes=np.array([1]),  # predicts class 1
        gt_boxes=np.array([box], float),
        gt_classes=np.array([0]),  # truth is class 0
    )
    assert cm.matrix[1, 0] == 1  # predicted 1 where truth was 0


def test_confusion_matrix_conf_filter_drops_low_score():
    cm = ConfusionMatrix(num_classes=2, conf=0.5)
    cm.process_batch(
        det_boxes=np.array([[0, 0, 10, 10]], float),
        det_scores=np.array([0.1]),  # below conf → ignored
        det_classes=np.array([0]),
        gt_boxes=np.zeros((0, 4)),
        gt_classes=np.array([]),
    )
    assert cm.matrix.sum() == 0


def test_confusion_matrix_plot_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    cm = ConfusionMatrix(num_classes=2)
    cm.process_batch(
        np.array([[0, 0, 10, 10]], float),
        np.array([0.9]),
        np.array([0]),
        np.array([[0, 0, 10, 10]], float),
        np.array([0]),
    )
    out = cm.plot(tmp_path / "cm.png", names={0: "cat", 1: "dog"})
    assert out.exists() and out.stat().st_size > 0
