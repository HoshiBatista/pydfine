"""HungarianMatcher — native D-FINE port.

Ported from ``D-FINE/src/zoo/dfine/matcher.py`` (Apache-2.0; © Facebook for the
original DETR matcher, © 2024 The D-FINE Authors). Solves the 1-to-1
prediction↔target assignment (LSAP via ``scipy``) that the criterion supervises.
Changes from upstream:

- Dropped ``@register()`` / ``__share__`` / the ``src.core`` import; box helpers
  come from ``.box_ops``.
- Added :meth:`HungarianMatcher.from_config`.

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

__all__ = ["HungarianMatcher"]


class HungarianMatcher(nn.Module):
    """Computes a 1-to-1 assignment between predictions and targets."""

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, (
            "all costs cant be 0"
        )

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> HungarianMatcher:
        return cls(
            weight_dict={
                "cost_class": cfg.cost_class,
                "cost_bbox": cfg.cost_bbox,
                "cost_giou": cfg.cost_giou,
            },
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
        C = C.view(bs, num_queries, -1).cpu()

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
