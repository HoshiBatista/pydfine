"""SemSegDecoder — native port of D-FINE-seg's dense per-pixel (``task="sem_seg"``) head.

Ported from ``D-FINE-seg/src/d_fine/arch/dfine_decoder.py`` (Apache-2.0, © ArgoHA,
https://github.com/ArgoHA/D-FINE-seg). Replaces the whole :class:`DFINETransformer`
decoder slot for semantic segmentation, bypassing the query/matcher/NMS path entirely.

It reuses the same :class:`MaskDecoder` fuser as the instance-mask branch — the attribute
name ``mask_decoder`` is preserved so the trained fuser weights inside
``dfine_seg_<size>_coco.pt`` transfer; the ``neck``/``classifier``/``aux_head`` train from
scratch on a sem_seg dataset. Layer/parameter names (``mask_decoder``/``neck``/``dropout``/
``classifier``/``aux_head``) are kept verbatim for checkpoint parity.

Logits are produced at 1/4 resolution and bilinearly upsampled ×4 to the input size; the
``aux_head`` deep-supervision branch (finest PAN feature) is train-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .mask_decoder import MaskDecoder

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["SemSegDecoder"]


def conv_gn_act(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 conv → GroupNorm(32) → ReLU, the sem_seg neck's building block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


class SemSegDecoder(nn.Module):
    """Dense per-pixel head: :class:`MaskDecoder` fuser → seg neck → 1×1 classifier.

    Args:
        num_classes: number of semantic classes (every pixel class, background included).
        feat_channels: HybridEncoder output channels per level (finest first).
        mask_dim: fuser output dim (128 for nano, 256 otherwise); the fuser weights
            transfer from ``dfine_seg_<size>_coco.pt``.
        mask_low_level_ch: backbone stride-8 channels prepended to the fuser inputs for
            models without a native 1/8 encoder level (nano); ``None`` otherwise.
        neck_dim: channel width of the two conv-GN-ReLU neck blocks.
        dropout: ``Dropout2d`` rate before the classifier (and in the aux head).
        aux: build the train-only deep-supervision head on the finest PAN feature.
    """

    def __init__(
        self,
        num_classes: int,
        feat_channels: list[int],
        mask_dim: int = 256,
        mask_low_level_ch: int | None = None,
        neck_dim: int = 128,
        dropout: float = 0.1,
        aux: bool = True,
    ) -> None:
        super().__init__()
        in_chs = list(feat_channels)
        if mask_low_level_ch is not None:
            in_chs = [mask_low_level_ch] + in_chs
        self.mask_decoder = MaskDecoder(in_chs=in_chs, out_ch=mask_dim)
        self.neck = nn.Sequential(conv_gn_act(mask_dim, neck_dim), conv_gn_act(neck_dim, neck_dim))
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(neck_dim, num_classes, 1)
        self.aux_head = (
            nn.Sequential(
                conv_gn_act(feat_channels[0], neck_dim),
                nn.Dropout2d(dropout),
                nn.Conv2d(neck_dim, num_classes, 1),
            )
            if aux
            else None
        )

    def forward(
        self,
        feats: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        low_level_feat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_feats = list(feats) if low_level_feat is None else [low_level_feat] + list(feats)
        x = self.mask_decoder(mask_feats)
        logits = self.classifier(self.dropout(self.neck(x)))
        logits = F.interpolate(logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        out = {"sem_seg_logits": logits}
        if self.training and self.aux_head is not None:
            aux = self.aux_head(feats[0])
            out["sem_seg_logits_aux"] = F.interpolate(
                aux, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )
        return out

    @classmethod
    def from_config(
        cls,
        cfg: DFINEConfig,
        *,
        mask_low_level_ch: int | None = None,
    ) -> SemSegDecoder:
        """Build the sem_seg decoder from a :class:`DFINEConfig`.

        ``neck_dim``/``dropout``/``aux`` follow upstream's fixed defaults (not config
        knobs). ``mask_low_level_ch`` is derived by the assembled model's seg wiring
        (non-``None`` only for nano), mirroring the instance-mask branch.
        """
        return cls(
            num_classes=cfg.num_classes,
            feat_channels=cfg.feat_channels,
            mask_dim=cfg.mask_dim,
            mask_low_level_ch=mask_low_level_ch,
        )
