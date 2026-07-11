"""Config surface tests: preset values match upstream, validation rejects bad configs."""

from __future__ import annotations

import pytest

from dfine import DFINEConfig, list_presets
from dfine.config import SIZE_PRESETS, SIZES


def test_all_sizes_present():
    assert list_presets() == SIZES
    assert set(SIZE_PRESETS) == set(SIZES)


@pytest.mark.parametrize("size", SIZES)
def test_preset_builds_and_records_size(size):
    cfg = DFINEConfig.preset(size)
    assert cfg.size == size
    # Preset dict values must survive onto the instance.
    for key, val in SIZE_PRESETS[size].items():
        assert getattr(cfg, key) == val, f"{size}.{key}"


def test_nano_is_two_level_128():
    # The nano preset is the structurally distinct one (verified vs upstream config).
    cfg = DFINEConfig.preset("n")
    assert cfg.num_levels == 2
    assert cfg.hidden_dim == 128
    assert cfg.in_channels == [512, 1024]
    assert cfg.num_points == [6, 6]


def test_size_specific_values():
    assert DFINEConfig.preset("x").reg_scale == 8.0
    assert DFINEConfig.preset("x").hidden_dim == 384
    assert DFINEConfig.preset("m").backbone == "hgnetv2_b2"
    assert DFINEConfig.preset("m").use_lab is True
    assert DFINEConfig.preset("l").freeze_norm is True
    assert DFINEConfig.preset("s").depth_mult == 0.34


def test_overrides_win_over_preset():
    cfg = DFINEConfig.preset("l", num_classes=3, reg_max=16)
    assert cfg.num_classes == 3
    assert cfg.reg_max == 16
    assert cfg.backbone == "hgnetv2_b4"  # preset default preserved


def test_override_method_revalidates():
    cfg = DFINEConfig.preset("l")
    assert cfg.override(num_classes=10).num_classes == 10
    with pytest.raises(ValueError):
        cfg.override(num_classes=0)


def test_roundtrip_dict():
    cfg = DFINEConfig.preset("s", num_classes=5)
    assert DFINEConfig.from_dict(cfg.to_dict()) == cfg
    # Unknown keys are ignored.
    assert DFINEConfig.from_dict({**cfg.to_dict(), "bogus": 1}) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": "xl"},
        {"num_classes": 0},
        {"backbone": "resnet50"},
        {"num_levels": 3, "in_channels": [1, 2]},
        {"class_names": ["a", "b"], "num_classes": 3},
        {"conf": 1.5},
        {"reg_scale": 0},
        {"decoder_layers": 3, "eval_idx": 5},
        {"use_encoder_idx": [9]},
    ],
)
def test_validation_rejects(kwargs):
    with pytest.raises(ValueError):
        DFINEConfig(**kwargs)


def test_length_consistency_holds_for_all_presets():
    for size in SIZES:
        cfg = DFINEConfig.preset(size)
        n = cfg.num_levels
        assert len(cfg.in_channels) == n
        assert len(cfg.feat_strides) == n
        assert len(cfg.feat_channels) == n
        assert len(cfg.num_points) == n
        assert len(cfg.return_idx) == n
