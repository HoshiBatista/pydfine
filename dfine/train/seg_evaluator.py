"""Segmentation validation: mIoU (``sem_seg``) + mask AP (instance ``segment``).

Two self-contained evaluators for the seg val path, plus a :func:`seg_val_fn` factory that
picks the right one by task for ``Trainer.fit(val_fn=…)`` / ``DFINE.train(val_loader=…)``.
Both score at the loader's (square ``imgsz``) resolution — predictions and the loader's GT
masks are already aligned there, so no original-size resize is needed.

- ``sem_seg`` → :class:`SemSegConfusionMatrix`: a dep-free ``[C, C]`` pixel confusion matrix
  (rows = GT, cols = pred), ``ignore_index`` pixels excluded → ``{"mIoU", "pixel_acc"}``.
  Follows D-FINE-seg's ``SemSegValidator`` (© ArgoHA, Apache-2.0).
- ``segment`` → :func:`evaluate_mask_ap`: COCO mask AP over the decoded top-k instance
  masks via ``torchmetrics`` ``MeanAveragePrecision(iou_type="segm")`` (the same metric
  D-FINE-seg uses), keyed ``{"mAP_50_95_mask", "mAP_50_mask", "mAP_75_mask"}``.

Needs ``pip install pydfine[train]`` (``torchmetrics`` for mask AP).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..log import LOGGER, metrics_line, rule

__all__ = [
    "SemSegConfusionMatrix",
    "evaluate_sem_seg",
    "evaluate_mask_ap",
    "seg_val_fn",
]


class SemSegConfusionMatrix:
    """Streaming ``[C, C]`` pixel confusion matrix (rows = GT, cols = pred) for mIoU.

    ``ignore_index`` GT pixels never enter the matrix. Nothing dense is stored.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """Accumulate one image; ``pred``/``gt`` are ``(H, W)`` int tensors at the same size."""
        valid = gt != self.ignore_index
        gt_v = gt[valid].long()
        if gt_v.numel() and int(gt_v.max()) >= self.num_classes:
            raise ValueError(
                f"GT mask has class id {int(gt_v.max())} >= num_classes={self.num_classes} "
                f"(ignore_index={self.ignore_index}); masks must use contiguous 0..num_classes-1"
            )
        idx = gt_v * self.num_classes + pred[valid].long().cpu()
        cm = torch.bincount(idx, minlength=self.num_classes**2)
        self.cm += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        """Return ``{"mIoU", "pixel_acc"}`` (mIoU averaged over classes present in GT)."""
        cm = self.cm.double()
        diag = cm.diag()
        union = cm.sum(1) + cm.sum(0) - diag
        present = cm.sum(1) > 0  # classes with GT pixels
        iou = diag / union.clamp(min=1)
        miou = iou[present].mean().item() if present.any() else 0.0
        acc = (diag.sum() / cm.sum().clamp(min=1)).item()
        return {"mIoU": miou, "pixel_acc": acc}


@torch.no_grad()
def evaluate_sem_seg(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Run ``model`` over a sem_seg val loader and return ``{"mIoU", "pixel_acc"}``.

    ``model`` runs in eval mode (restored afterwards); the argmax label map is compared to
    each target's ``sem_mask`` at the loader resolution.
    """
    cm = SemSegConfusionMatrix(num_classes, ignore_index)
    was_training = model.training
    model.eval()
    try:
        for samples, targets in data_loader:
            logits = model(samples.to(device))["sem_seg_logits"]
            preds = logits.argmax(1)
            for pred, t in zip(preds, targets):
                cm.update(pred, t["sem_mask"].to(pred.device))
    finally:
        if was_training:
            model.train()

    metrics = cm.compute()
    LOGGER.info(f"{rule('eval · sem_seg', 'cyan')}  {metrics_line(metrics)}")
    return metrics


@torch.no_grad()
def evaluate_mask_ap(
    model: nn.Module,
    postprocessor: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    conf: float = 0.05,
    mask_thresh: float = 0.5,
) -> dict[str, float]:
    """Run ``model`` over an instance-seg val loader and return COCO mask AP.

    Decodes each image's top-k queries, upsamples their mask maps to the input resolution,
    thresholds at ``mask_thresh``, keeps detections scoring above ``conf``, and scores them
    against the loader's GT masks with ``torchmetrics`` (segm). Returns
    ``{"mAP_50_95_mask", "mAP_50_mask", "mAP_75_mask"}``.
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval")
    was_training = model.training
    model.eval()
    try:
        for samples, targets in data_loader:
            samples = samples.to(device)
            imgsz = samples.shape[-2:]
            outputs = model(samples)
            pred_masks = outputs["pred_masks"]  # sigmoid [B, Q, h, w]
            sizes = torch.tensor([[imgsz[1], imgsz[0]]] * len(targets), device=device)
            results = postprocessor(outputs, sizes)

            preds, gts = [], []
            for b, (det, t) in enumerate(zip(results, targets)):
                keep = det["scores"] > conf
                m = pred_masks[b][det["query_index"][keep]]
                if m.numel():
                    m = F.interpolate(
                        m.unsqueeze(0).float(), size=imgsz, mode="bilinear", align_corners=False
                    ).squeeze(0)
                    binm = (m >= mask_thresh).to(torch.uint8)
                else:
                    binm = torch.zeros((0, *imgsz), dtype=torch.uint8, device=device)
                preds.append(
                    {"masks": binm, "scores": det["scores"][keep], "labels": det["labels"][keep]}
                )
                gts.append(
                    {
                        "masks": t["masks"].to(device).to(torch.uint8),
                        "labels": t["labels"].to(device),
                    }
                )
            metric.update(preds, gts)
    finally:
        if was_training:
            model.train()

    out = metric.compute()
    metrics = {
        "mAP_50_95_mask": float(out["map"]),
        "mAP_50_mask": float(out["map_50"]),
        "mAP_75_mask": float(out["map_75"]),
    }
    LOGGER.info(f"{rule('eval · segm', 'cyan')}  {metrics_line(metrics)}")
    return metrics


def seg_val_fn(
    task: str,
    *,
    postprocessor: nn.Module | None = None,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    conf: float = 0.05,
    mask_thresh: float = 0.5,
) -> Callable[[nn.Module, Iterable], dict[str, float]]:
    """Build a ``(module, loader) -> metrics`` closure for the seg task.

    ``task="sem_seg"`` scores mIoU; ``task="segment"`` scores mask AP (needs
    ``postprocessor``). Slots into ``Trainer.fit(val_fn=…)`` / ``DFINE.train(val_fn=…)``.
    """
    if task == "sem_seg":

        def _sem(module: nn.Module, loader: Iterable) -> dict[str, float]:
            return evaluate_sem_seg(
                module, loader, device, num_classes=num_classes, ignore_index=ignore_index
            )

        return _sem

    if task == "segment":
        if postprocessor is None:
            raise ValueError("segment mask AP needs a postprocessor.")

        def _ins(module: nn.Module, loader: Iterable) -> dict[str, float]:
            return evaluate_mask_ap(
                module, postprocessor, loader, device, conf=conf, mask_thresh=mask_thresh
            )

        return _ins

    raise ValueError(f"seg_val_fn supports task in (segment, sem_seg), got {task!r}.")
