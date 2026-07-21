"""MaskDecoder — native port of D-FINE-seg's instance-mask feature fuser.

Ported from ``D-FINE-seg/src/d_fine/arch/dfine_decoder.py`` (Apache-2.0, © ArgoHA,
https://github.com/ArgoHA/D-FINE-seg). The detection core there follows the D-FINE
paper; this mask head is D-FINE-seg's own addition. Changes from upstream: none of
substance — layer/parameter names (``lateral``/``bn``/``fusion_conv``/``fusion_norm``/
``up_conv``/``bn1``) are preserved verbatim so ``dfine_seg_<size>_coco.pt`` loads unchanged.

Takes the HybridEncoder PAN outputs (stride 8/16/32, or a backbone stride-8 feature
prepended for models without a native 1/8 level, e.g. nano) and fuses them into a
single 1/4-resolution mask-feature map. A per-query mask embedding (built in the
decoder) is then dot-producted against these features to yield per-instance masks.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

__all__ = ["MaskDecoder"]


class MaskDecoder(nn.Module):
    """Fuse multi-scale PAN features into 1/4-res mask features ``(B, out_ch, H/4, W/4)``.

    Args:
        in_chs: channel count of each input level, finest first (``feats[0]`` is the
            base resolution; coarser levels are bilinearly upsampled onto it).
        out_ch: output (mask) channel dim — ``mask_dim`` (128 for nano, 256 otherwise).
    """

    def __init__(self, in_chs: list[int], out_ch: int = 256) -> None:
        super().__init__()
        n_groups = 32
        # 1x1 proj + GroupNorm for each input level.
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1, bias=False) for c in in_chs])
        self.bn = nn.ModuleList([nn.GroupNorm(n_groups, out_ch) for _ in in_chs])

        # 3x3 conv after fusion to smooth aliasing and add spatial context.
        self.fusion_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.fusion_norm = nn.GroupNorm(n_groups, out_ch)

        # Upsample 1/8 -> 1/4 with bilinear + conv (avoids checkerboard artifacts).
        self.up_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.kaiming_normal_(self.up_conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # Take the finest level as the base, fuse upsampled coarser levels into it.
        f0 = self.bn[0](self.lateral[0](feats[0]))  # (B, out_ch, H/8, W/8)
        x = f0
        for i in range(1, len(feats)):
            t = self.bn[i](self.lateral[i](feats[i]))
            x = x + F.interpolate(t, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        x = self.act(self.fusion_norm(self.fusion_conv(x)))

        # 1/8 -> 1/4 (bilinear x2 + conv, smoother than ConvTranspose).
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act(self.bn1(self.up_conv(x)))
        return x  # (B, out_ch, H/4, W/4)
