"""Shape tests for the native HybridEncoder, incl. backbone -> encoder integration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import HGNetv2, HybridEncoder  # noqa: E402
from dfine.config import SIZES  # noqa: E402


def _cfg(size: str) -> DFINEConfig:
    return DFINEConfig.preset(size, backbone_pretrained=False, freeze_norm=False, freeze_at=-1)


@pytest.mark.parametrize("size", SIZES)
def test_encoder_out_channels_and_strides(size):
    cfg = _cfg(size)
    enc = HybridEncoder.from_config(cfg)
    assert enc.out_channels == [cfg.hidden_dim] * cfg.num_levels
    assert enc.out_strides == cfg.feat_strides


@pytest.mark.parametrize("size", SIZES)
def test_encoder_forward_shapes(size):
    cfg = _cfg(size)
    enc = HybridEncoder.from_config(cfg).eval()

    # Feed backbone-shaped feature maps at the right strides.
    feats = [
        torch.randn(2, ch, cfg.imgsz // s, cfg.imgsz // s)
        for ch, s in zip(cfg.in_channels, cfg.feat_strides)
    ]
    with torch.no_grad():
        outs = enc(feats)

    assert len(outs) == cfg.num_levels
    for out, stride in zip(outs, cfg.feat_strides):
        assert out.shape[1] == cfg.hidden_dim
        assert out.shape[2] == cfg.imgsz // stride
        assert out.shape[3] == cfg.imgsz // stride


@pytest.mark.parametrize("size", ["n", "l"])
def test_backbone_to_encoder_integration(size):
    cfg = _cfg(size)
    backbone = HGNetv2.from_config(cfg).eval()
    enc = HybridEncoder.from_config(cfg).eval()

    with torch.no_grad():
        outs = enc(backbone(torch.randn(1, 3, cfg.imgsz, cfg.imgsz)))

    assert len(outs) == cfg.num_levels
    for out, stride in zip(outs, cfg.feat_strides):
        assert out.shape == (1, cfg.hidden_dim, cfg.imgsz // stride, cfg.imgsz // stride)


def test_eval_uses_cached_pos_embed():
    cfg = _cfg("l")
    enc = HybridEncoder.from_config(cfg)
    # eval_spatial_size set -> a cached pos_embed buffer/attr exists per encoder idx.
    for idx in cfg.use_encoder_idx:
        assert getattr(enc, f"pos_embed{idx}", None) is not None
