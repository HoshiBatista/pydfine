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

Needs ``pip install pydfine[train]`` (``faster-coco-eval``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..log import LOGGER, metrics_line, rule

__all__ = ["evaluate", "coco_val_fn", "COCO_STAT_NAMES"]

COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
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
    plots: bool = False,
    output_dir: str | None = None,
    names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` over ``data_loader`` and return COCO metrics.

    ``model`` runs in eval mode (restored afterwards); each batch is decoded by
    ``postprocessor`` to original-scale ``xyxy`` boxes, then scored against the loader's
    ground truth. Returns a dict keyed by :data:`COCO_STAT_NAMES` (``AP`` is the primary
    mAP@[.50:.95]).

    With ``plots=True`` also writes ``confusion_matrix.png`` + ``pr_curve.png`` under
    ``output_dir`` (default ``runs/val``) and logs the per-class AP table — see
    :mod:`dfine.train.metrics`. ``names`` maps class id → label for the plots. The
    confusion matrix assumes contiguous labels (``remap_mscoco_category=False``).
    """
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

    evaluator = FasterCocoEvaluator(_coco_gt(data_loader), [iou_type])

    # Analytics plots (confusion matrix / PR curves / worst frames) accumulate locally per
    # process, so under multi-GPU each rank sees only its data shard AND every rank would race
    # writing the same PNGs. Disable them there; the COCO metrics below are still gathered
    # across ranks by FasterCocoEvaluator and remain correct.
    from .distributed import get_world_size

    if plots and get_world_size() > 1:
        LOGGER.warning(
            "val analytics plots are disabled under multi-GPU (per-rank shards would be partial "
            "and ranks would race writing the same files); COCO metrics are unaffected."
        )
        plots = False

    cm = prm = worst = None
    if plots:
        from .metrics import ConfusionMatrix, PRCurveMetrics, WorstPredictions

        cat_ids = [int(c) for c in evaluator.coco_eval[iou_type].params.catIds]
        nc = (max(names) + 1) if names else (max(cat_ids, default=-1) + 1)
        cm, prm, worst = ConfusionMatrix(nc), PRCurveMetrics(nc), WorstPredictions()

    was_training = model.training
    model.eval()
    try:
        for samples, targets in data_loader:
            samples = samples.to(device)
            orig_sizes = torch.stack([t["orig_size"].to(device) for t in targets], dim=0)
            outputs = model(samples)
            results = postprocessor(outputs, orig_sizes)
            evaluator.update({int(t["image_id"].item()): r for t, r in zip(targets, results)})
            if cm is not None:
                _update_analytics(cm, prm, worst, results, targets)
    finally:
        if was_training:
            model.train()

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.coco_eval[iou_type].stats.tolist()
    metrics = {name: float(v) for name, v in zip(COCO_STAT_NAMES, stats)}
    LOGGER.info(f"{rule(f'eval · {iou_type}', 'cyan')}  {metrics_line(metrics)}")

    if cm is not None:
        from .metrics import save_val_analytics

        save_val_analytics(
            evaluator.coco_eval[iou_type],
            cm,
            output_dir or "runs/val",
            names,
            prm=prm,
            worst=worst,
        )
    return metrics


def _update_analytics(cm, prm, worst, results, targets) -> None:
    """Feed one batch's predictions + GT (as original-scale ``xyxy``) to the accumulators."""
    for t, r in zip(targets, results):
        w, h = (float(v) for v in t["orig_size"].tolist())
        g = t["boxes"].cpu().numpy().reshape(-1, 4)  # cxcywh, normalized
        if len(g):
            cx, cy, bw, bh = g[:, 0] * w, g[:, 1] * h, g[:, 2] * w, g[:, 3] * h
            gt_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        else:
            gt_xyxy = np.zeros((0, 4))
        db = r["boxes"].cpu().numpy()
        ds = r["scores"].cpu().numpy()
        dl = r["labels"].cpu().numpy()
        gl = t["labels"].cpu().numpy()
        cm.process_batch(db, ds, dl, gt_xyxy, gl)
        prm.process_batch(db, ds, dl, gt_xyxy, gl)
        worst.add(t.get("image_path"), db, ds, dl, gt_xyxy, gl)


def coco_val_fn(
    postprocessor: nn.Module,
    device: torch.device,
    *,
    iou_type: str = "bbox",
    plots: bool = False,
    plots_dir: str | Path | None = None,
    names: dict[int, str] | None = None,
) -> Callable[[nn.Module, Iterable], dict[str, float]]:
    """Build a ``(module, loader) -> metrics`` closure for ``Trainer.fit(val_fn=…)``.

    Captures the ``postprocessor``/``device`` so the trainer can score the (EMA) module
    each epoch: ``trainer.fit(train_loader, val_loader=…, val_fn=coco_val_fn(pp, dev))``.

    With ``plots=True`` the analytics bundle (confusion matrix, P/R/F1 curves, per-class AP,
    worst-predictions gallery) is also written **every epoch** under
    ``plots_dir/epoch{N}/`` (``plots_dir`` defaults to ``runs/train/val``), where ``N``
    counts validation calls from 0 — so each epoch's plots are kept side by side rather than
    overwritten. ``names`` maps class id → label for the plots. The analytics assume
    contiguous labels (``remap_mscoco_category=False``).
    """
    base = Path(plots_dir) if plots_dir is not None else Path("runs/train") / "val"
    state = {"call": 0}

    def _val_fn(module: nn.Module, loader: Iterable) -> dict[str, float]:
        out_dir = str(base / f"epoch{state['call']}") if plots else None
        metrics = evaluate(
            module,
            postprocessor,
            loader,
            device,
            iou_type=iou_type,
            plots=plots,
            output_dir=out_dir,
            names=names,
        )
        state["call"] += 1
        return metrics

    return _val_fn
