"""Shape/consistency tests for the native HGNetv2 backbone.

Runs with random init (``pretrained=False``) so no network is needed. The key
invariant: for every preset, the backbone's emitted channels/strides must equal the
config's ``in_channels``/``feat_strides`` — that's what the encoder is wired to expect.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import HGNetv2  # noqa: E402
from dfine.config import SIZES  # noqa: E402


def _build(size: str) -> tuple[DFINEConfig, HGNetv2]:
    cfg = DFINEConfig.preset(size, backbone_pretrained=False, freeze_norm=False, freeze_at=-1)
    return cfg, HGNetv2.from_config(cfg)


@pytest.mark.parametrize("size", SIZES)
def test_out_channels_match_config(size):
    cfg, backbone = _build(size)
    assert backbone.out_channels == cfg.in_channels
    assert backbone.out_strides == cfg.feat_strides


@pytest.mark.parametrize("size", SIZES)
def test_forward_shapes(size):
    cfg, backbone = _build(size)
    backbone.eval()
    imgsz = 640
    with torch.no_grad():
        feats = backbone(torch.randn(2, 3, imgsz, imgsz))

    assert len(feats) == cfg.num_levels
    for feat, ch, stride in zip(feats, cfg.in_channels, cfg.feat_strides):
        assert feat.shape[0] == 2
        assert feat.shape[1] == ch
        assert feat.shape[2] == imgsz // stride
        assert feat.shape[3] == imgsz // stride


def test_name_normalization_and_bad_name():
    assert HGNetv2._normalize_name("hgnetv2_b4") == "B4"
    assert HGNetv2._normalize_name("B4") == "B4"
    with pytest.raises(ValueError):
        HGNetv2(name="resnet50", pretrained=False)


def test_freeze_at_freezes_stem():
    # freeze_at>=0 with freeze_stem_only should freeze the stem params only.
    backbone = HGNetv2(
        name="hgnetv2_b0",
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_stem_only=True,
        freeze_norm=False,
        pretrained=False,
    )
    assert all(not p.requires_grad for p in backbone.stem.parameters())
    assert any(p.requires_grad for p in backbone.stages.parameters())


def test_freeze_norm_replaces_batchnorm():
    from dfine.backends.native.common import FrozenBatchNorm2d

    backbone = HGNetv2(name="hgnetv2_b0", freeze_norm=True, pretrained=False)
    has_frozen = any(isinstance(m, FrozenBatchNorm2d) for m in backbone.modules())
    has_plain_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in backbone.modules())
    assert has_frozen and not has_plain_bn
