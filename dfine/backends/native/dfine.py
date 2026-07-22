"""Assembled DFINE model — native D-FINE port.

Ported from ``D-FINE/src/zoo/dfine/dfine.py`` (Apache-2.0, © 2024 The D-FINE
Authors). Wires the ported backbone (:class:`HGNetv2`), encoder
(:class:`HybridEncoder`) and decoder (:class:`DFINETransformer`) into one
``nn.Module`` whose forward is ``decoder(encoder(backbone(x)))``. Changes from
upstream:

- Dropped ``@register()`` / ``__inject__``; submodules are built from a
  :class:`DFINEConfig` via :meth:`from_config`.
- Added :meth:`load` for loading upstream ``.pth`` checkpoints.
- Segmentation wiring (``_seg_wiring``: the stride-8 low-level-feature plumbing and
  the ``SemSegDecoder`` decoder swap) follows D-FINE-seg's model assembly
  (Apache-2.0, © ArgoHA, https://github.com/ArgoHA/D-FINE-seg). No-ops for
  ``task="detect"`` — the detection path is byte-identical.

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
from .sem_seg_decoder import SemSegDecoder

if TYPE_CHECKING:
    from ...config import DFINEConfig

__all__ = ["DFINE"]


class DFINE(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    @staticmethod
    def _seg_wiring(cfg: DFINEConfig) -> tuple[list[int] | None, int | None]:
        """Segmentation wiring: (backbone return_idx override, mask_low_level_ch).

        When a mask-fuser task is on (segment or sem_seg) and the encoder has no native
        stride-8 level (nano), the backbone must emit an extra stride-8 feature (stage
        index 1) for the mask decoder's low-level input. Returns ``(None, None)`` for the
        detection path.
        """
        if not cfg.uses_mask_fuser or 8 in cfg.feat_strides:
            return None, None
        return_idx = cfg.return_idx if 1 in cfg.return_idx else [1, *cfg.return_idx]
        name = HGNetv2._normalize_name(cfg.backbone)
        mask_low_level_ch = HGNetv2.arch_configs[name]["stage_config"]["stage2"][2]
        return return_idx, mask_low_level_ch

    @classmethod
    def from_config(cls, cfg: DFINEConfig) -> DFINE:
        """Build the full model (backbone + encoder + decoder) from a config.

        For ``task="sem_seg"`` the whole detection decoder is replaced by
        :class:`SemSegDecoder`; ``detect``/``segment`` use :class:`DFINETransformer`.
        """
        backbone_return_idx, mask_low_level_ch = cls._seg_wiring(cfg)
        if cfg.task == "sem_seg":
            decoder: nn.Module = SemSegDecoder.from_config(cfg, mask_low_level_ch=mask_low_level_ch)
        else:
            decoder = DFINETransformer.from_config(
                cfg,
                enable_mask_head=cfg.enable_mask_head,
                mask_dim=cfg.mask_dim,
                mask_low_level_ch=mask_low_level_ch,
            )
        return cls(
            backbone=HGNetv2.from_config(cfg, return_idx=backbone_return_idx),
            encoder=HybridEncoder.from_config(cfg),
            decoder=decoder,
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
        cfg = config_for(spec, **{"backbone_pretrained": False, **overrides})
        model = cls.from_config(cfg).eval()
        path = download_weights(spec, cache_dir_override=cache_dir)
        model.load(path, use_ema=use_ema, strict=True)
        return model

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        low_level_feat = None
        if len(feats) > len(self.encoder.in_channels):
            low_level_feat, feats = feats[0], feats[1:]
        feats = self.encoder(feats)
        return self.decoder(feats, targets, low_level_feat=low_level_feat)

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
