"""Assembled DFINE model — native D-FINE port.

Ported from ``D-FINE/src/zoo/dfine/dfine.py`` (Apache-2.0, © 2024 The D-FINE
Authors). Wires the ported backbone (:class:`HGNetv2`), encoder
(:class:`HybridEncoder`) and decoder (:class:`DFINETransformer`) into one
``nn.Module`` whose forward is ``decoder(encoder(backbone(x)))``. Changes from
upstream:

- Dropped ``@register()`` / ``__inject__``; submodules are built from a
  :class:`DFINEConfig` via :meth:`from_config`.
- Added :meth:`load` for loading upstream ``.pth`` checkpoints.

The submodule attribute names (``backbone``/``encoder``/``decoder``) match
upstream so a released checkpoint's ``state_dict`` loads with ``strict=True``.
The postprocessor is intentionally *not* part of this module (upstream keeps it
separate); pair this with :class:`DFINEPostProcessor` for decoded detections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from .dfine_decoder import DFINETransformer
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .loader import load_checkpoint

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["DFINE"]


class DFINE(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> DFINE:
        """Build the full model (backbone + encoder + decoder) from a config."""
        return cls(
            backbone=HGNetv2.from_config(cfg),
            encoder=HybridEncoder.from_config(cfg),
            decoder=DFINETransformer.from_config(cfg),
        )

    @classmethod
    def from_pretrained(cls, name: str, cache_dir=None, use_ema: bool = True, **overrides) -> DFINE:
        """Build a model for a released checkpoint and load its weights.

        ``name`` is a catalogue entry (``"dfine-s"``, ``"dfine-l-obj365"`` ...);
        see :func:`dfine.registry.list_checkpoints`. The architecture (size +
        ``num_classes``) is derived from the checkpoint so obj365's 366-class head
        is wired automatically; the weights are downloaded/cached and strict-loaded.
        Extra ``overrides`` pass through to the config (avoid changing ``imgsz`` —
        it's baked into the checkpoint's anchor buffer).
        """
        from ...downloads import download_weights
        from ...registry import config_for, resolve

        spec = resolve(name)
        # The checkpoint already carries backbone weights, so skip the ImageNet
        # backbone download (it would be overwritten anyway). Caller can override.
        cfg = config_for(spec, **{"backbone_pretrained": False, **overrides})
        model = cls.from_config(cfg).eval()
        path = download_weights(spec, cache_dir_override=cache_dir)
        model.load(path, use_ema=use_ema, strict=True)
        return model

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def load(self, path, use_ema: bool = True, strict: bool = True):
        """Load an upstream ``.pth`` checkpoint into this model (in place).

        Delegates to :func:`load_checkpoint`; returns the same
        ``(missing, unexpected)`` key lists (empty on a clean strict load).
        """
        return load_checkpoint(self, path, use_ema=use_ema, strict=strict)

    def deploy(self) -> DFINE:
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
