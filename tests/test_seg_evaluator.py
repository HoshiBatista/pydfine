"""Seg val evaluation: mIoU (sem_seg) confusion matrix + mask AP (instance segment).

The confusion-matrix math needs only torch; the end-to-end evaluators run the real native
seg model (scipy matcher for the postprocessor); mask AP additionally needs torchmetrics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import DFINE as NativeDFINE  # noqa: E402
from dfine.backends.native import DFINEPostProcessor  # noqa: E402
from dfine.train.seg_evaluator import (  # noqa: E402
    SemSegConfusionMatrix,
    evaluate_mask_ap,
    evaluate_sem_seg,
    seg_val_fn,
)

IMGSZ = 320  # >= the nano decoder's 300-query top-k needs (matches the other seg tests)


def _cfg(**kw):
    return DFINEConfig.preset(
        "n", imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1, **kw
    )


# --- confusion matrix / mIoU --------------------------------------------------


def test_confusion_matrix_perfect_prediction():
    cm = SemSegConfusionMatrix(num_classes=3)
    gt = torch.tensor([[0, 1], [2, 2]])
    cm.update(gt.clone(), gt)
    m = cm.compute()
    assert m["mIoU"] == 1.0 and m["pixel_acc"] == 1.0


def test_confusion_matrix_iou_and_ignore_index():
    cm = SemSegConfusionMatrix(num_classes=2, ignore_index=255)
    gt = torch.tensor([0, 0, 1, 1, 255])
    pred = torch.tensor([0, 1, 1, 1, 0])  # last pixel ignored (gt==255)
    cm.update(pred, gt)
    m = cm.compute()
    # IoU_0 = 1/2, IoU_1 = 2/3 -> mIoU = 0.5833; pixel_acc = 3/4.
    assert abs(m["mIoU"] - (0.5 + 2 / 3) / 2) < 1e-4
    assert abs(m["pixel_acc"] - 0.75) < 1e-4


def test_confusion_matrix_rejects_out_of_range_class():
    cm = SemSegConfusionMatrix(num_classes=2)
    with pytest.raises(ValueError, match="num_classes"):
        cm.update(torch.tensor([0, 1]), torch.tensor([0, 5]))


# --- end-to-end sem_seg mIoU --------------------------------------------------


def _sem_loader(batch=2, num_classes=4):
    samples = torch.rand(batch, 3, IMGSZ, IMGSZ)
    sem = torch.zeros(IMGSZ, IMGSZ, dtype=torch.int64)
    sem[:, IMGSZ // 2 :] = 1
    return [(samples, [{"sem_mask": sem.clone()} for _ in range(batch)])]


def test_evaluate_sem_seg_returns_bounded_metrics():
    cfg = _cfg(task="sem_seg", num_classes=4)
    model = NativeDFINE.from_config(cfg)
    metrics = evaluate_sem_seg(
        model, _sem_loader(), torch.device("cpu"), num_classes=4, ignore_index=255
    )
    assert set(metrics) == {"mIoU", "pixel_acc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# --- end-to-end instance mask AP ----------------------------------------------


def _seg_loader(batch=1, n=2, num_classes=80):
    samples = torch.rand(batch, 3, IMGSZ, IMGSZ)
    targets = []
    for _ in range(batch):
        boxes = torch.rand(n, 4) * 0.3 + 0.35
        masks = torch.zeros(n, IMGSZ, IMGSZ, dtype=torch.uint8)
        for i, (cx, cy, w, h) in enumerate(boxes):
            x0, y0 = int((cx - w / 2) * IMGSZ), int((cy - h / 2) * IMGSZ)
            x1, y1 = int((cx + w / 2) * IMGSZ), int((cy + h / 2) * IMGSZ)
            masks[i, y0:y1, x0:x1] = 1
        targets.append(
            {"labels": torch.randint(0, num_classes, (n,)), "boxes": boxes, "masks": masks}
        )
    return [(samples, targets)]


def test_evaluate_mask_ap_returns_bounded_metrics():
    pytest.importorskip("torchmetrics")
    cfg = _cfg(task="segment")
    model = NativeDFINE.from_config(cfg)
    pp = DFINEPostProcessor.from_config(cfg)
    metrics = evaluate_mask_ap(model, pp, _seg_loader(), torch.device("cpu"))
    assert set(metrics) == {"mAP_50_95_mask", "mAP_50_mask", "mAP_75_mask"}
    # untrained -> AP is near zero, but must be a finite value in the valid COCO range.
    assert all(-1.0 <= v <= 1.0 for v in metrics.values())


# --- val_fn factory -----------------------------------------------------------


def test_seg_val_fn_selects_by_task():
    cpu = torch.device("cpu")
    assert callable(seg_val_fn("sem_seg", device=cpu, num_classes=4))
    pp = DFINEPostProcessor.from_config(_cfg(task="segment"))
    assert callable(seg_val_fn("segment", postprocessor=pp, device=cpu, num_classes=80))
    with pytest.raises(ValueError, match="postprocessor"):
        seg_val_fn("segment", device=cpu, num_classes=80)  # segment needs a postprocessor
    with pytest.raises(ValueError, match="segment, sem_seg"):
        seg_val_fn("detect", device=cpu, num_classes=80)
