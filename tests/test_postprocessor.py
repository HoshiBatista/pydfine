"""Tests for the native DFINEPostProcessor.

Covers the decode contract (top-k, cxcywh->xyxy in original scale), the
end-to-end decoder->postproc wiring, the COCO id remap branch, and deploy mode.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import (  # noqa: E402
    DFINEPostProcessor,
    DFINETransformer,
    HGNetv2,
    HybridEncoder,
)
from dfine.backends.native.coco import mscoco_label2category  # noqa: E402

IMGSZ = 320


def _outputs(batch, num_queries, num_classes):
    # normalized cxcywh: centers in 0.3..0.7, w/h in 0..0.2, so every xyxy corner
    # stays in [0, 1] (the postprocessor doesn't clamp, matching upstream).
    cxcy = torch.rand(batch, num_queries, 2) * 0.4 + 0.3
    wh = torch.rand(batch, num_queries, 2) * 0.2
    return {
        "pred_logits": torch.randn(batch, num_queries, num_classes),
        "pred_boxes": torch.cat([cxcy, wh], dim=-1),
    }


def test_from_config_fields():
    cfg = DFINEConfig.preset("s")
    pp = DFINEPostProcessor.from_config(cfg)
    assert pp.num_classes == cfg.num_classes
    assert pp.num_top_queries == cfg.num_top_queries
    assert pp.use_focal_loss is True
    assert pp.remap_mscoco_category == cfg.remap_mscoco_category


def test_decode_shapes_and_scale():
    pp = DFINEPostProcessor(num_classes=80, num_top_queries=300)
    outputs = _outputs(2, 500, 80)
    sizes = torch.tensor([[640, 480], [1024, 768]])  # (w, h) per image

    results = pp(outputs, sizes)

    assert isinstance(results, list) and len(results) == 2
    for r, (w, h) in zip(results, sizes.tolist()):
        assert r["boxes"].shape == (300, 4)
        assert r["scores"].shape == (300,)
        assert r["labels"].shape == (300,)
        # focal-loss path -> sigmoid scores in (0, 1); labels valid class ids.
        assert (r["scores"] >= 0).all() and (r["scores"] <= 1).all()
        assert (r["labels"] >= 0).all() and (r["labels"] < 80).all()
        # boxes decoded into original pixel scale (xyxy), x<=w, y<=h.
        assert r["boxes"][:, [0, 2]].max() <= w + 1e-3
        assert r["boxes"][:, [1, 3]].max() <= h + 1e-3


def test_scores_sorted_descending():
    pp = DFINEPostProcessor(num_classes=80, num_top_queries=100)
    results = pp(_outputs(1, 300, 80), torch.tensor([[640, 640]]))
    scores = results[0]["scores"]
    assert torch.all(scores[:-1] >= scores[1:])


def test_remap_mscoco_category():
    pp = DFINEPostProcessor(num_classes=80, num_top_queries=10, remap_mscoco_category=True)
    results = pp(_outputs(1, 50, 80), torch.tensor([[640, 640]]))
    cats = results[0]["labels"].tolist()
    # every remapped id must be a real COCO category id (1..90, non-contiguous).
    assert all(c in mscoco_label2category.values() for c in cats)


def test_deploy_mode_returns_tensors():
    pp = DFINEPostProcessor(num_classes=80, num_top_queries=300).deploy()
    assert pp.deploy_mode is True
    labels, boxes, scores = pp(_outputs(2, 500, 80), torch.tensor([[640, 640], [640, 640]]))
    assert labels.shape == (2, 300)
    assert boxes.shape == (2, 300, 4)
    assert scores.shape == (2, 300)


@pytest.mark.parametrize("size", ["n", "s"])
def test_pipeline_with_postprocessor(size):
    cfg = DFINEConfig.preset(
        size, imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1
    )
    backbone = HGNetv2.from_config(cfg).eval()
    encoder = HybridEncoder.from_config(cfg).eval()
    decoder = DFINETransformer.from_config(cfg).eval()
    pp = DFINEPostProcessor.from_config(cfg)

    with torch.no_grad():
        out = decoder(encoder(backbone(torch.randn(1, 3, IMGSZ, IMGSZ))))
        results = pp(out, torch.tensor([[IMGSZ, IMGSZ]]))

    r = results[0]
    assert r["boxes"].shape == (cfg.num_top_queries, 4)
    assert torch.isfinite(r["boxes"]).all()
