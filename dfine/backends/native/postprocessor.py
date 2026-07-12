"""DFINEPostProcessor — native D-FINE port.

Ported from ``D-FINE/src/zoo/dfine/postprocessor.py`` (Apache-2.0, © 2024 The
D-FINE Authors; copied from RT-DETR, © 2023 lyuwenyu). Decodes the decoder's
``pred_logits``/``pred_boxes`` into top-k detections in the *original* image
scale. Changes from upstream:

- Dropped ``@register()``, ``__share__``, and the ``src.core`` import.
- The ``remap_mscoco_category`` branch pulls its lookup table from
  ``.coco`` instead of ``src.data.dataset`` (kept lazy so a non-COCO model
  never imports it).
- Added :meth:`DFINEPostProcessor.from_config`.

Behaviour and tensor math are unchanged so results match upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    """``a % b`` written without the modulo op (older TensorRT lacks ``%``)."""
    return a - a // b * b


class DFINEPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
        remap_mscoco_category: bool = False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> DFINEPostProcessor:
        """Build the postprocessor from a :class:`DFINEConfig`.

        D-FINE always trains with focal loss, so ``use_focal_loss`` is fixed to
        ``True`` (upstream sets it globally in ``dfine_hgnetv2.yml``).
        """
        return cls(
            num_classes=cfg.num_classes,
            use_focal_loss=True,
            num_top_queries=cfg.num_top_queries,
            remap_mscoco_category=cfg.remap_mscoco_category,
        )

    def extra_repr(self) -> str:
        return (
            f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, "
            f"num_top_queries={self.num_top_queries}"
        )

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        # onnx export path: return raw tensors, no python-side postproc
        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from .coco import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            results.append(dict(labels=lab, boxes=box, scores=sco))

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
