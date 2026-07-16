"""COCO detection evaluation for ``DFINE.val`` (Phase 4).

Ports upstream D-FINE's ``det_engine.evaluate`` down to the single-process detection
path: run the model over a COCO val loader, decode with the postprocessor, and score
predictions against the loader's ground-truth ``.coco`` with ``faster-coco-eval``
(the same evaluator upstream wraps as ``CocoEvaluator``). Returns the 12 standard COCO
summary metrics as a named ``dict[str, float]``.

The result slots straight into ``Trainer.fit(val_fn=…)`` via :func:`coco_val_fn`.

Label spaces must line up: the postprocessor's ``remap_mscoco_category`` decides whether
predicted labels are contiguous ``0..N-1`` or the sparse MS-COCO ids, and those must
match the ``category_id`` values in the ground-truth annotations JSON (stock MS-COCO =
sparse ids → build the model with ``remap_mscoco_category=True``).

Needs ``pip install dfine[train]`` (``faster-coco-eval``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["evaluate", "coco_val_fn", "COCO_STAT_NAMES"]

# Order of the classic 12-element COCO ``summarize()`` stats array.
COCO_STAT_NAMES = (
    "AP",  # AP @ IoU=0.50:0.95 (primary mAP)
    "AP50",  # AP @ IoU=0.50
    "AP75",  # AP @ IoU=0.75
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",  # AR given 1 detection per image
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def _coco_gt(data_loader: Iterable):
    """Pull the ground-truth ``COCO`` object off a COCO val loader's dataset."""
    dataset = getattr(data_loader, "dataset", None)
    coco = getattr(dataset, "coco", None)
    if coco is None:
        raise ValueError(
            "val loader's dataset has no `.coco` ground truth — pass a loader built by "
            "`build_coco_dataloader`/`build_coco_dataloaders` (a `CocoDetection`)."
        )
    return coco


@torch.no_grad()
def evaluate(
    model: nn.Module,
    postprocessor: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    iou_type: str = "bbox",
) -> dict[str, float]:
    """Evaluate ``model`` over ``data_loader`` and return COCO metrics.

    ``model`` runs in eval mode (restored afterwards); each batch is decoded by
    ``postprocessor`` to original-scale ``xyxy`` boxes, then scored against the loader's
    ground truth. Returns a dict keyed by :data:`COCO_STAT_NAMES` (``AP`` is the primary
    mAP@[.50:.95]).
    """
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

    evaluator = FasterCocoEvaluator(_coco_gt(data_loader), [iou_type])

    was_training = model.training
    model.eval()
    try:
        for samples, targets in data_loader:
            samples = samples.to(device)
            orig_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
            outputs = model(samples)
            results = postprocessor(outputs, orig_sizes)
            evaluator.update({int(t["image_id"].item()): r for t, r in zip(targets, results)})
    finally:
        if was_training:
            model.train()

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.coco_eval[iou_type].stats.tolist()
    return {name: float(v) for name, v in zip(COCO_STAT_NAMES, stats)}


def coco_val_fn(
    postprocessor: nn.Module,
    device: torch.device,
    *,
    iou_type: str = "bbox",
) -> Callable[[nn.Module, Iterable], dict[str, float]]:
    """Build a ``(module, loader) -> metrics`` closure for ``Trainer.fit(val_fn=…)``.

    Captures the ``postprocessor``/``device`` so the trainer can score the (EMA) module
    each epoch: ``trainer.fit(train_loader, val_loader=…, val_fn=coco_val_fn(pp, dev))``.
    """

    def _val_fn(module: nn.Module, loader: Iterable) -> dict[str, float]:
        return evaluate(module, postprocessor, loader, device, iou_type=iou_type)

    return _val_fn
