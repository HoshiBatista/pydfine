"""COCO evaluation tests (Phase 4).

Builds a tiny COCO val set on disk and checks `dfine.train.evaluator.evaluate`:
the metric plumbing (box scaling, image_id keying, label space) is validated by
replaying the ground truth back as *perfect* predictions -> AP == 1.0, and the
returned dict carries the 12 named COCO stats. Needs faster-coco-eval (train extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_coco_eval")
import torch.nn as nn  # noqa: E402

from dfine.train.dataset import build_coco_val_dataloader  # noqa: E402
from dfine.train.evaluator import COCO_STAT_NAMES, coco_val_fn, evaluate  # noqa: E402
from tests.test_dataset import _write_split  # noqa: E402

IMGSZ = 320


def _make_val_root(tmp_path):
    _write_split(
        tmp_path / "val",
        tmp_path / "annotations" / "instances_val.json",
        ((200, 150), (120, 90)),
    )
    return str(tmp_path)


class _ReplayModel(nn.Module):
    """Return precomputed per-image results, in dataset (loader) order."""

    def __init__(self, per_image):
        super().__init__()
        self._per_image = per_image
        self._cursor = 0

    def forward(self, samples):
        n = samples.shape[0]
        out = self._per_image[self._cursor : self._cursor + n]
        self._cursor += n
        return out


class _IdentityPost(nn.Module):
    """Postprocessor stub: predictions already decoded upstream."""

    def forward(self, outputs, orig_sizes):
        return outputs


def _perfect_predictions(loader):
    """Build GT-as-prediction dicts (xyxy px, GT labels, score 1.0) in dataset order."""
    dataset = loader.dataset
    coco = dataset.coco
    per_image = []
    for image_id in dataset.ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        per_image.append(
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "scores": torch.ones(len(labels)),
            }
        )
    return per_image


def test_evaluate_perfect_predictions_scores_ap_one(tmp_path):
    root = _make_val_root(tmp_path)
    loader = build_coco_val_dataloader(root, imgsz=IMGSZ, batch_size=2, num_workers=0)

    model = _ReplayModel(_perfect_predictions(loader))
    metrics = evaluate(model, _IdentityPost(), loader, torch.device("cpu"))

    assert set(metrics) == set(COCO_STAT_NAMES)
    assert all(isinstance(v, float) for v in metrics.values())
    # GT replayed as predictions -> perfect detection.
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["AP50"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_restores_training_mode(tmp_path):
    root = _make_val_root(tmp_path)
    loader = build_coco_val_dataloader(root, imgsz=IMGSZ, batch_size=2, num_workers=0)
    model = _ReplayModel(_perfect_predictions(loader))
    model.train()
    evaluate(model, _IdentityPost(), loader, torch.device("cpu"))
    assert model.training  # eval() is restored afterwards


def test_coco_val_fn_closure(tmp_path):
    root = _make_val_root(tmp_path)
    loader = build_coco_val_dataloader(root, imgsz=IMGSZ, batch_size=2, num_workers=0)
    fn = coco_val_fn(_IdentityPost(), torch.device("cpu"))
    metrics = fn(_ReplayModel(_perfect_predictions(loader)), loader)
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_rejects_non_coco_loader():
    class _Plain:
        dataset = object()

    with pytest.raises(ValueError, match="no `.coco`"):
        evaluate(_IdentityPost(), _IdentityPost(), _Plain(), torch.device("cpu"))
