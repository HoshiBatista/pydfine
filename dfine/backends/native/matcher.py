"""HungarianMatcher — native D-FINE port.

Ported from ``D-FINE/src/zoo/dfine/matcher.py`` (Apache-2.0; © Facebook for the
original DETR matcher, © 2024 The D-FINE Authors). Solves the 1-to-1
prediction↔target assignment (LSAP via ``scipy``) that the criterion supervises.
Changes from upstream:

- Dropped ``@register()`` / ``__share__`` / the ``src.core`` import; box helpers
  come from ``.box_ops``.
- Added :meth:`HungarianMatcher.from_config`.
- Instance-mask matching costs (``dice_cost`` + ``sigmoid_focal_cost`` and the
  ``pred_masks`` branch in :meth:`forward`) are ported from
  ``D-FINE-seg/src/d_fine/matcher.py`` (Apache-2.0, © ArgoHA — an independent
  from-scratch framework). They only activate for ``task="segment"`` (``cost_mask`` /
  ``cost_mask_dice`` > 0 and the outputs carry ``pred_masks``); detection is
  byte-identical.

Matching math is unchanged. ``scipy`` is a training-only dependency (install the
``train`` extra); it's imported lazily so inference never needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["HungarianMatcher", "dice_cost", "sigmoid_focal_cost"]


def dice_cost(pred_masks: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise Dice cost ``[Q, T]`` (1 - Dice) between query mask probs and GT masks.

    ``pred_masks`` is ``[Q, H, W]`` sigmoid probabilities, ``gt_masks`` ``[T, H, W]`` binary.
    """
    pred_masks = pred_masks.flatten(1).float()  # [Q, H*W]
    gt_masks = gt_masks.flatten(1).float()  # [T, H*W]
    numerator = 2 * torch.einsum("qp,tp->qt", pred_masks, gt_masks)  # [Q, T]
    denominator = pred_masks.sum(dim=1, keepdim=True) + gt_masks.sum(dim=1)  # [Q, T]
    return 1 - (numerator + eps) / (denominator + eps)


def sigmoid_focal_cost(
    pred_logits: torch.Tensor, gt_labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Pairwise pixel-wise sigmoid-focal cost ``[Q, T]``, normalized by pixel count.

    ``pred_logits`` is ``[Q, H*W]`` mask logits, ``gt_labels`` ``[T, H*W]`` binary.
    """
    pred_logits = pred_logits.float()
    gt_labels = gt_labels.float()
    pred_prob = pred_logits.sigmoid()
    neg_cost = (1 - alpha) * (pred_prob**gamma) * (-(1 - pred_prob + 1e-8).log())
    pos_cost = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())
    cost = torch.einsum("qp,tp->qt", pos_cost, gt_labels) + torch.einsum(
        "qp,tp->qt", neg_cost, (1 - gt_labels)
    )
    return cost / pred_logits.shape[1]


class HungarianMatcher(nn.Module):
    """Computes a 1-to-1 assignment between predictions and targets."""

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_mask = weight_dict.get("cost_mask", 0)  # focal mask cost (segment only)
        self.cost_mask_dice = weight_dict.get("cost_mask_dice", 0)  # dice mask cost (segment only)
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, (
            "all costs cant be 0"
        )

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> HungarianMatcher:
        weight_dict = {
            "cost_class": cfg.cost_class,
            "cost_bbox": cfg.cost_bbox,
            "cost_giou": cfg.cost_giou,
        }
        if cfg.enable_mask_head:
            weight_dict["cost_mask"] = cfg.cost_mask
            weight_dict["cost_mask_dice"] = cfg.cost_mask_dice
        return cls(
            weight_dict=weight_dict,
            use_focal_loss=True,
            alpha=cfg.matcher_alpha,
            gamma=cfg.matcher_gamma,
        )

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        from scipy.optimize import linear_sum_assignment

        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)
        self._add_mask_cost(C, outputs, targets, bs, num_queries)
        C = C.cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}

    def _add_mask_cost(self, C, outputs, targets, bs, num_queries):
        """Add the instance-mask cost (dice + focal) into ``C`` in place (segment only).

        No-op unless a mask cost weight is set, the outputs carry ``pred_masks``, and some
        target has masks — so detection matching is untouched. GT masks are bilinear-resized
        to the prediction's ``(Hm, Wm)``. Mirrors D-FINE-seg's per-batch accumulation.
        """
        if self.cost_mask <= 0 and self.cost_mask_dice <= 0:
            return
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            return
        if not any((m := t.get("masks")) is not None and m.numel() > 0 for t in targets):
            return

        # Drop leading denoising queries if present, so Q lines up with the match queries.
        if pred_masks.shape[1] != num_queries:
            dn_num = pred_masks.shape[1] - num_queries
            if dn_num > 0:
                pred_masks = pred_masks[:, dn_num:]

        Hm, Wm = pred_masks.shape[-2:]
        sizes = [len(v["boxes"]) for v in targets]
        offset = 0
        for b in range(bs):
            n_tgt = sizes[b]
            t = targets[b]
            m = t.get("masks")
            if n_tgt == 0 or m is None or m.numel() == 0:
                offset += n_tgt
                continue

            tgt_m = m.float().to(pred_masks.device)  # [Nb, H, W]
            if tgt_m.shape[-2:] != (Hm, Wm):
                tgt_m = F.interpolate(
                    tgt_m.unsqueeze(1), size=(Hm, Wm), mode="bilinear", align_corners=False
                ).squeeze(1)

            cost_mc = torch.zeros(num_queries, n_tgt, device=pred_masks.device)
            if self.cost_mask_dice > 0:
                cost_mc = cost_mc + self.cost_mask_dice * dice_cost(pred_masks[b].sigmoid(), tgt_m)
            if self.cost_mask > 0:
                cost_mc = cost_mc + self.cost_mask * sigmoid_focal_cost(
                    pred_masks[b].flatten(1), tgt_m.flatten(1), alpha=self.alpha, gamma=self.gamma
                )
            C[b, :, offset : offset + n_tgt] = C[b, :, offset : offset + n_tgt] + cost_mc
            offset += n_tgt

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        from scipy.optimize import linear_sum_assignment

        indices_list = []
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        return indices_list
