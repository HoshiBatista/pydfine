"""Tests for HungarianMatcher + DFINECriterion (training loss).

The matcher needs scipy (train extra); skipped when absent. The criterion test
runs the real backbone→encoder→decoder in training mode and checks the full loss
dict is finite and differentiable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import (  # noqa: E402
    DFINECriterion,
    DFINETransformer,
    HGNetv2,
    HungarianMatcher,
    HybridEncoder,
)

IMGSZ = 320
BASE_LOSSES = {"loss_vfl", "loss_bbox", "loss_giou", "loss_fgl"}


def _cfg(**kw):
    return DFINEConfig.preset(
        "n", imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1, **kw
    )


def _target(n, num_classes=80):
    # cxcywh boxes safely inside the image; random class labels.
    return {
        "labels": torch.randint(0, num_classes, (n,)),
        "boxes": torch.rand(n, 4) * 0.5 + 0.25,
    }


def _synthetic_outputs(batch=2, num_queries=300, num_classes=80):
    return {
        "pred_logits": torch.randn(batch, num_queries, num_classes),
        "pred_boxes": torch.rand(batch, num_queries, 4) * 0.5 + 0.25,
    }


# --- matcher ------------------------------------------------------------------


def test_matcher_from_config():
    m = HungarianMatcher.from_config(_cfg())
    assert (m.cost_class, m.cost_bbox, m.cost_giou) == (2.0, 5.0, 2.0)
    assert m.alpha == 0.25 and m.use_focal_loss is True


def test_matcher_returns_one_to_one_indices():
    m = HungarianMatcher.from_config(_cfg())
    targets = [_target(3), _target(5)]
    indices = m(_synthetic_outputs(), targets)["indices"]
    assert len(indices) == 2
    for (src, tgt), t in zip(indices, targets):
        n = len(t["boxes"])
        assert src.shape == (n,) and tgt.shape == (n,)
        assert sorted(tgt.tolist()) == list(range(n))  # every target matched once
        assert src.unique().numel() == n  # distinct predictions


def test_matcher_return_topk():
    m = HungarianMatcher.from_config(_cfg())
    targets = [_target(3), _target(4)]
    out = m(_synthetic_outputs(), targets, return_topk=2)
    o2m = out["indices_o2m"]
    # k=2 matches per target -> 2*n indices per image.
    assert o2m[0][0].shape[0] == 2 * len(targets[0]["boxes"])


# --- criterion ----------------------------------------------------------------


def test_criterion_from_config():
    c = DFINECriterion.from_config(_cfg())
    assert c.losses == ["vfl", "boxes", "local"]
    assert c.weight_dict == {
        "loss_vfl": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }
    assert c.alpha == 0.75 and c.gamma == 2.0
    assert isinstance(c.matcher, HungarianMatcher)


def _train_outputs(cfg, targets):
    backbone = HGNetv2.from_config(cfg).train()
    encoder = HybridEncoder.from_config(cfg).train()
    decoder = DFINETransformer.from_config(cfg).train()
    feats = encoder(backbone(torch.randn(len(targets), 3, IMGSZ, IMGSZ)))
    return decoder(feats, targets), decoder


def test_criterion_end_to_end_finite_and_differentiable():
    torch.manual_seed(0)
    cfg = _cfg()
    targets = [_target(3), _target(2)]
    outputs, decoder = _train_outputs(cfg, targets)

    criterion = DFINECriterion.from_config(cfg)
    losses = criterion(outputs, targets)

    # Final-layer base losses are present and finite.
    assert BASE_LOSSES.issubset(losses.keys())
    assert all(torch.isfinite(v) for v in losses.values())

    # Auxiliary/denoising terms are emitted (aux, enc, pre, dn suffixes).
    assert any(k.endswith("_aux_0") for k in losses)
    assert any("_dn_" in k for k in losses)
    assert any(k.startswith("loss_ddf") for k in losses)  # DDF appears on aux layers

    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    grad = sum(p.grad.abs().sum() for p in decoder.parameters() if p.grad is not None)
    assert float(grad) > 0  # loss is connected to the decoder


def test_criterion_no_denoising_still_works():
    # num_denoising=0 -> no dn_* outputs, but the loss must still compute.
    torch.manual_seed(1)
    cfg = _cfg(num_denoising=0)
    targets = [_target(2), _target(4)]
    outputs, _ = _train_outputs(cfg, targets)
    assert "dn_outputs" not in outputs

    losses = DFINECriterion.from_config(cfg)(outputs, targets)
    assert BASE_LOSSES.issubset(losses.keys())
    assert not any("_dn_" in k for k in losses)
    assert torch.isfinite(sum(losses.values()))
