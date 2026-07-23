"""SemSegCriterion — native port of D-FINE-seg's semantic-segmentation loss.

Ported from ``D-FINE-seg/src/d_fine/sem_seg_criterion.py`` (Apache-2.0, © ArgoHA,
https://github.com/ArgoHA/D-FINE-seg — an independent from-scratch framework).
Supervises the dense per-pixel (``task="sem_seg"``) head: pixel cross-entropy +
multi-class soft Dice, plus an auxiliary CE on the deep-supervision branch when the
decoder emits ``sem_seg_logits_aux``. ``ignore_index`` pixels are excluded from both
terms. Changes from upstream: added :meth:`SemSegCriterion.from_config`; loss math is
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["SemSegCriterion"]


class SemSegCriterion(nn.Module):
    """CE + multi-class soft Dice (+ auxiliary CE) over the sem_seg logits.

    ``forward(outputs, targets)`` reads ``outputs['sem_seg_logits']`` ``[B, C, H, W]``
    and each target's ``sem_mask`` ``[H, W]`` long label map; returns a weighted loss
    dict (``loss_ce``/``loss_dice`` and, when present, ``loss_aux``).
    """

    def __init__(
        self,
        weight_dict,
        num_classes,
        ignore_index=255,
        class_weights=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.class_weights = (
            torch.tensor(list(class_weights), dtype=torch.float32) if class_weights else None
        )

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> SemSegCriterion:
        """Build the sem_seg criterion from a :class:`DFINEConfig`.

        Uses upstream's fixed ``{loss_ce, loss_dice, loss_aux}`` weight dict and the
        ``ignore_index`` / label-smoothing knobs from the config.
        """
        return cls(
            weight_dict={
                "loss_ce": cfg.loss_ce_weight,
                "loss_dice": cfg.loss_dice_weight,
                "loss_aux": cfg.loss_aux_weight,
            },
            num_classes=cfg.num_classes,
            ignore_index=cfg.sem_seg_ignore_index,
            label_smoothing=cfg.sem_seg_label_smoothing,
        )

    def _dice(self, logits, target, valid):
        # multi-class soft dice; ignored pixels are masked out of both prob and one-hot
        prob = logits.softmax(1)
        one_hot = F.one_hot(torch.where(valid, target, 0), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(prob.dtype)
        v = valid.unsqueeze(1).to(prob.dtype)
        prob, one_hot = prob * v, one_hot * v
        inter = (prob * one_hot).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)  # absent classes -> dice 1 -> zero loss
        return 1.0 - dice.mean()

    def forward(self, outputs, targets):
        logits = outputs["sem_seg_logits"].float()
        target = torch.stack([t["sem_mask"] for t in targets])  # (B, H, W) long
        valid = target != self.ignore_index

        if not valid.any():  # all-ignore batch (e.g. fully padded) contributes zero loss
            zero = logits.sum() * 0.0
            losses = {"loss_ce": zero, "loss_dice": zero}
            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = outputs["sem_seg_logits_aux"].float().sum() * 0.0
        else:
            weight = (
                self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            losses = {
                "loss_ce": F.cross_entropy(
                    logits,
                    target,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    label_smoothing=self.label_smoothing,
                ),
                "loss_dice": self._dice(logits, target, valid),
            }
            if "sem_seg_logits_aux" in outputs:
                losses["loss_aux"] = F.cross_entropy(
                    outputs["sem_seg_logits_aux"].float(),
                    target,
                    ignore_index=self.ignore_index,
                )
        return {k: v * self.weight_dict[k] for k, v in losses.items()}
