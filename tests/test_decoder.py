"""Shape tests for the native DFINETransformer decoder + full pipeline.

Uses a small imgsz for speed and random init (no weights/network). Eval mode returns
just ``pred_logits``/``pred_boxes``; training-only denoising/aux paths aren't exercised.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import DFINETransformer, HGNetv2, HybridEncoder  # noqa: E402
from dfine.config import SIZES  # noqa: E402

# Large enough that every preset yields >= num_queries tokens (N is only 2-level:
# at 320 -> 20^2 + 10^2 = 500 > 300).
IMGSZ = 320


def _cfg(size: str) -> DFINEConfig:
    return DFINEConfig.preset(
        size, imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1
    )


@pytest.mark.parametrize("size", SIZES)
def test_decoder_forward_shapes(size):
    cfg = _cfg(size)
    dec = DFINETransformer.from_config(cfg).eval()

    # Encoder-shaped inputs: hidden_dim channels at each feat stride.
    feats = [torch.randn(2, cfg.hidden_dim, IMGSZ // s, IMGSZ // s) for s in cfg.feat_strides]
    with torch.no_grad():
        out = dec(feats)

    assert set(out) == {"pred_logits", "pred_boxes"}
    assert out["pred_logits"].shape == (2, cfg.num_queries, cfg.num_classes)
    assert out["pred_boxes"].shape == (2, cfg.num_queries, 4)
    assert torch.isfinite(out["pred_boxes"]).all()


def test_decoder_dim_feedforward_wiring():
    # N uses 512, others 1024 — verified against upstream configs.
    assert DFINEConfig.preset("n").decoder_dim_feedforward == 512
    assert DFINEConfig.preset("l").decoder_dim_feedforward == 1024
    dec = DFINETransformer.from_config(_cfg("n"))
    assert dec.decoder.layers[0].linear1.out_features == 512


def test_custom_num_classes():
    cfg = DFINEConfig.preset("s", imgsz=IMGSZ, backbone_pretrained=False, num_classes=3)
    dec = DFINETransformer.from_config(cfg).eval()
    feats = [torch.randn(1, cfg.hidden_dim, IMGSZ // s, IMGSZ // s) for s in cfg.feat_strides]
    with torch.no_grad():
        out = dec(feats)
    assert out["pred_logits"].shape == (1, cfg.num_queries, 3)


@pytest.mark.parametrize("size", ["n", "l"])
def test_full_pipeline(size):
    cfg = _cfg(size)
    backbone = HGNetv2.from_config(cfg).eval()
    encoder = HybridEncoder.from_config(cfg).eval()
    decoder = DFINETransformer.from_config(cfg).eval()

    with torch.no_grad():
        out = decoder(encoder(backbone(torch.randn(1, 3, IMGSZ, IMGSZ))))

    assert out["pred_logits"].shape == (1, cfg.num_queries, cfg.num_classes)
    assert out["pred_boxes"].shape == (1, cfg.num_queries, 4)
    assert torch.isfinite(out["pred_logits"]).all()
