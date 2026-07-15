"""Training augmentations (Phase 4) — the D-FINE two-phase augment pipeline.

Ports the training-transform recipe from upstream ``configs/.../dataloader.yml`` +
``src/data/transforms``:

    RandomPhotometricDistort(p=0.5) → RandomZoomOut → RandomIoUCrop(p=0.8)
    → SanitizeBoundingBoxes → RandomHorizontalFlip → Resize(imgsz)
    → SanitizeBoundingBoxes → ConvertPILImage → ConvertBoxes(cxcywh, normalized)

Most ops are torchvision ``transforms.v2`` built-ins; only ``RandomIoUCrop`` needs a
probability wrapper (ported verbatim). The final tensor/box conversions are shared with
``dataset.py``.

Two-phase schedule (``TrainCompose`` + ``stop_epoch``): D-FINE trains most epochs with
the heavy geometry/photometric augs, then turns them **off** for the trailing
``no_aug_epoch`` epochs so the model settles on clean, full-frame images. Once the epoch
reaches ``stop_epoch`` the advanced ops (photometric distort, zoom-out, IoU-crop) are
skipped while the deterministic resize/convert tail keeps running. The dataloader
forwards ``set_epoch`` down to this compose, so it just works with ``DFINE.train``.

Plug in via ``build_coco_dataloader(..., transforms=train_transforms(imgsz, stop_epoch=…))``.
Needs ``pip install dfine[train]``.
"""

from __future__ import annotations

from typing import Any

import torch
import torchvision
import torchvision.transforms.v2 as T

from .dataset import _ConvertBoxes, _ConvertPILImage

torchvision.disable_beta_transforms_warning()

__all__ = ["RandomIoUCrop", "TrainCompose", "train_transforms", "ADVANCED_OPS"]

# Ops disabled during the trailing no-aug epochs (matches upstream's stop_epoch policy).
ADVANCED_OPS = ("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop")


class RandomIoUCrop(T.RandomIoUCrop):
    """``torchvision`` ``RandomIoUCrop`` with an apply-probability ``p`` (ported)."""

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


class TrainCompose:
    """Sequential transform with D-FINE's ``stop_epoch`` two-phase policy.

    Applies ``ops`` in order to the ``(image, target)`` sample. From ``stop_epoch``
    onward, any op whose class name is in ``stop_ops`` is skipped (the no-aug tail);
    everything else always runs. Call :meth:`set_epoch` each epoch — the dataloader
    forwards it automatically.
    """

    def __init__(self, ops, stop_ops=ADVANCED_OPS, stop_epoch: int | None = None):
        self.ops = list(ops)
        self.stop_ops = set(stop_ops)
        self.stop_epoch = stop_epoch if stop_epoch is not None else 10**9
        self._epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        drop_advanced = self._epoch >= self.stop_epoch
        for op in self.ops:
            if drop_advanced and type(op).__name__ in self.stop_ops:
                continue
            sample = op(sample)
        return sample


def train_transforms(
    imgsz: int = 640,
    *,
    photometric_p: float = 0.5,
    iou_crop_p: float = 0.8,
    hflip: bool = True,
    zoom_out: bool = True,
    stop_epoch: int | None = None,
) -> TrainCompose:
    """Build D-FINE's training augmentation pipeline (see module docstring).

    ``stop_epoch`` is the epoch at which the advanced augs switch off — pass
    ``cfg.epochs - cfg.no_aug_epoch`` to match the released recipe. The individual
    probabilities/toggles let you dial the augmentation strength for small datasets.
    """
    ops: list[Any] = [T.RandomPhotometricDistort(p=photometric_p)]
    if zoom_out:
        ops.append(T.RandomZoomOut(fill=0))
    ops.append(RandomIoUCrop(p=iou_crop_p))
    ops.append(T.SanitizeBoundingBoxes(min_size=1))
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    ops += [
        T.Resize((imgsz, imgsz)),
        T.SanitizeBoundingBoxes(min_size=1),
        _ConvertPILImage(),
        _ConvertBoxes(),
    ]
    return TrainCompose(ops, stop_ops=ADVANCED_OPS, stop_epoch=stop_epoch)
