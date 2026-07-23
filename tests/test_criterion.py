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
from dfine.backends.native.matcher import dice_cost, sigmoid_focal_cost  # noqa: E402

IMGSZ = 320
BASE_LOSSES = {"loss_vfl", "loss_bbox", "loss_giou", "loss_fgl"}


def _cfg(**kw):
    return DFINEConfig.preset(
        "n", imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1, **kw
    )


def _target(n, num_classes=80, masks=False):
    # cxcywh boxes safely inside the image; random class labels.
    t = {
        "labels": torch.randint(0, num_classes, (n,)),
        "boxes": torch.rand(n, 4) * 0.5 + 0.25,
    }
    if masks:
        t["masks"] = (torch.rand(n, IMGSZ, IMGSZ) > 0.5).float()
    return t


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


def test_mask_cost_functions():
    # Dice cost is ~0 for a perfect match and ~1 for the complement; focal cost is lower
    # for the matching query than the opposite one.
    gt = torch.zeros(1, 4, 4)
    gt[0, :2] = 1.0  # top half foreground
    dice = dice_cost(gt.clone(), gt)  # probs == gt
    assert float(dice[0, 0]) < 1e-3
    assert float(dice_cost(1 - gt, gt)[0, 0]) > 0.99

    logits = torch.stack([(gt[0] * 20 - 10), (10 - gt[0] * 20)])  # [2, 4, 4]: match vs opposite
    focal = sigmoid_focal_cost(logits.flatten(1), gt.flatten(1))  # [2, 1]
    assert float(focal[0, 0]) < float(focal[1, 0])


def test_matcher_from_config_segment_adds_mask_costs():
    assert (HungarianMatcher.from_config(_cfg()).cost_mask,) == (0,)  # detect: off
    ms = HungarianMatcher.from_config(_cfg(task="segment"))
    assert ms.cost_mask == 1.0 and ms.cost_mask_dice == 1.0


def test_matcher_mask_cost_breaks_box_tie():
    # Two queries identical in class/box (tie); only their masks differ. The mask-aware
    # (segment) matcher assigns the query whose mask matches the target; the mask-blind
    # (detect) matcher ignores masks and falls back to the first query.
    Hm = 8
    outputs = {
        "pred_logits": torch.zeros(1, 2, 80),  # identical class cost for both queries
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.4]] * 2]),  # both == target box
        "pred_masks": torch.stack(
            [torch.full((Hm, Hm), -10.0), torch.full((Hm, Hm), 10.0)]  # q0 empty, q1 full
        ).view(1, 2, Hm, Hm),
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "masks": torch.ones(1, Hm, Hm),  # matches q1
        }
    ]
    seg_src = HungarianMatcher.from_config(_cfg(task="segment"))(outputs, targets)["indices"][0][0]
    det_src = HungarianMatcher.from_config(_cfg())(outputs, targets)["indices"][0][0]
    assert int(seg_src) == 1  # mask cost pulls the match to the full-mask query
    assert int(det_src) == 0  # no mask cost -> tie broken by query order


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


# --- segmentation mask losses (TS1) -------------------------------------------


def test_criterion_detect_has_no_mask_losses():
    # Detection is byte-identical: no "masks" loss, no loss_mask_* weights/keys.
    c = DFINECriterion.from_config(_cfg())
    assert "masks" not in c.losses
    assert not any("mask" in k for k in c.weight_dict)


def test_criterion_from_config_segment_adds_mask_losses():
    c = DFINECriterion.from_config(_cfg(task="segment"))
    assert c.losses == ["vfl", "boxes", "local", "masks"]
    assert c.weight_dict["loss_mask_bce"] == 1.0
    assert c.weight_dict["loss_mask_dice"] == 1.0


def test_criterion_segment_mask_losses_finite_and_differentiable():
    torch.manual_seed(2)
    from dfine.backends.native import DFINE as NativeDFINE

    cfg = _cfg(task="segment")
    model = NativeDFINE.from_config(cfg).train()
    targets = [_target(3, masks=True), _target(2, masks=True)]
    outputs = model(torch.randn(2, 3, IMGSZ, IMGSZ), targets)
    # The seg decoder emits per-instance mask logits on the final + denoising layers.
    assert "pred_masks" in outputs and "dn_pred_masks" in outputs

    losses = DFINECriterion.from_config(cfg)(outputs, targets)
    assert {"loss_mask_bce", "loss_mask_dice"}.issubset(losses)
    assert "loss_mask_bce_dn_final" in losses  # DN mask supervision term
    assert any(k.startswith("loss_mask_bce_aux_") for k in losses)  # aux-layer mask loss
    assert all(torch.isfinite(v) for v in losses.values())

    sum(losses.values()).backward()
    mask_grad = sum(
        p.grad.abs().sum()
        for n, p in model.named_parameters()
        if p.grad is not None and "mask" in n
    )
    assert float(mask_grad) > 0  # the mask branch actually receives gradient
