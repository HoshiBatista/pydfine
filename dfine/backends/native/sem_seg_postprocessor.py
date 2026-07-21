"""SemSegPostProcessor — turn sem_seg logits into original-scale label maps.

The dense head (:class:`SemSegDecoder`) emits ``sem_seg_logits`` ``[B, C, H, W]`` at the
input resolution. This postprocessor takes the per-pixel argmax and NEAREST-resizes each
image's label map to its original size, mirroring D-FINE-seg's ``process_sem_seg``
(``D-FINE-seg/src/infer/torch_model.py``, Apache-2.0, © ArgoHA). Like pydfine's detection
path it assumes a plain (non-letterbox) resize, so there is no pad cropping.

Kept as a separate module from the model (like :class:`DFINEPostProcessor`); pair it with
a ``task="sem_seg"`` model for decoded label maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["SemSegPostProcessor"]


class SemSegPostProcessor(nn.Module):
    """argmax over classes → per-image NEAREST resize to original size → uint8 ``[H, W]``."""

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> SemSegPostProcessor:
        """Build from a :class:`DFINEConfig` (no tunable state; matches the other heads)."""
        return cls()

    def forward(self, outputs, orig_target_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Decode ``outputs["sem_seg_logits"]`` to a list of ``[H0, W0]`` uint8 label maps.

        ``orig_target_sizes`` is ``[B, 2]`` as ``(W, H)`` per image (the same convention
        as :class:`DFINEPostProcessor`).
        """
        maps = outputs["sem_seg_logits"].argmax(1, keepdim=True).float()
        results = []
        for b in range(maps.shape[0]):
            w0, h0 = int(orig_target_sizes[b][0]), int(orig_target_sizes[b][1])
            m = F.interpolate(maps[b : b + 1], size=(h0, w0), mode="nearest")[0, 0]
            results.append(m.to(torch.uint8))
        return results
