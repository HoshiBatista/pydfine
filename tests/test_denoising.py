"""Contrastive-denoising query group tests (training-only path).

Focus on `box_noise_scale`: it is a documented, typed config knob, so it must (a) actually
reach the denoising group — not be silently pinned to 1.0 — and (b) accept 0.0 (box noising
off) without raising. Label/box noise use randomness, so tests seed torch where they compare.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from dfine.backends.native.denoising import (  # noqa: E402
    get_contrastive_denoising_training_group,
)

_TARGETS = [
    {
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
    }
]


def _run(box_noise_scale: float, seed: int = 0):
    torch.manual_seed(seed)
    emb = nn.Embedding(81, 8)
    return get_contrastive_denoising_training_group(
        _TARGETS,
        num_classes=80,
        num_queries=300,
        class_embed=emb,
        num_denoising=100,
        label_noise_ratio=0.0,  # isolate box noise
        box_noise_scale=box_noise_scale,
    )


def test_box_noise_scale_zero_does_not_raise_and_keeps_boxes_clean():
    # Regression: input_query_bbox_unact used to be defined only inside `if box_noise_scale > 0`,
    # so box_noise_scale=0 raised UnboundLocalError. With noise off the positive queries must be
    # the untouched GT boxes (label noise is also off here).
    _, bbox_unact, _, _ = _run(0.0)
    recovered = bbox_unact.sigmoid()[0, :2]  # first 2 positives = the 2 GTs
    assert torch.allclose(recovered, _TARGETS[0]["boxes"], atol=1e-4)


def test_box_noise_scale_positive_perturbs_boxes():
    # A positive scale must actually move the boxes off the GT (proves the value is used).
    _, bbox_unact, _, _ = _run(1.0)
    recovered = bbox_unact.sigmoid()[0, :2]
    assert not torch.allclose(recovered, _TARGETS[0]["boxes"], atol=1e-3)


def test_decoder_threads_configured_box_noise_scale():
    # The decoder must pass cfg.box_noise_scale into the denoising group, not a hardcoded 1.0.
    pytest.importorskip("scipy")
    from dfine import DFINEConfig
    from dfine.backends.native import DFINE as NativeDFINE
    from dfine.backends.native import DFINECriterion
    from dfine.train.trainer import build_optimizer, train_one_epoch

    cfg = DFINEConfig.preset(
        "n",
        imgsz=320,
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=100,
        box_noise_scale=0.0,  # would UnboundLocalError if honored but the fn weren't robust
    )
    model = NativeDFINE.from_config(cfg)
    assert model.decoder.box_noise_scale == 0.0
    criterion = DFINECriterion.from_config(cfg)
    opt = build_optimizer(model, cfg)
    samples = torch.rand(1, 3, 320, 320)
    # Must run the training denoising path (model.train()) without raising.
    stats = train_one_epoch(
        model, criterion, [(samples, _TARGETS)], opt, torch.device("cpu"), 0, print_freq=100
    )
    assert "loss" in stats and stats["loss"] == stats["loss"]  # finite
