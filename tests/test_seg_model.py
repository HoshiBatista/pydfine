"""S4: the assembled DFINE model wired for instance segmentation (task="segment")."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native.dfine import DFINE  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402


def test_seg_wiring_nano_prepends_low_level():
    """Nano (no stride-8 encoder level) emits an extra backbone stride-8 feature."""
    cfg = DFINEConfig.preset("n", task="segment")
    return_idx, mask_low_level_ch = DFINE._seg_wiring(cfg)
    assert return_idx == [1, 2, 3]  # 1 (stride-8) prepended to the nano [2, 3]
    assert mask_low_level_ch == 256  # B0 stage2 out channels


def test_seg_wiring_noop_when_encoder_has_stride8():
    """s/m/l/x already have a stride-8 encoder level — no low-level plumbing needed."""
    cfg = DFINEConfig.preset("s", task="segment")
    assert DFINE._seg_wiring(cfg) == (None, None)


def test_detection_model_has_no_mask_head():
    """Default task="detect" builds no mask modules and yields only detection keys."""
    cfg = DFINEConfig.preset("n", backbone_pretrained=False, imgsz=320)
    model = DFINE.from_config(cfg).eval()
    assert not hasattr(model.decoder, "mask_decoder")
    assert model.backbone.return_idx == cfg.return_idx  # unchanged
    with torch.no_grad():
        out = model(torch.rand(1, 3, cfg.imgsz, cfg.imgsz))
    assert set(out) == {"pred_logits", "pred_boxes"}


def test_segment_model_forward_produces_masks():
    """End-to-end: a segment model returns sigmoid masks [B, Q, H/4, W/4]."""
    cfg = DFINEConfig.preset("n", task="segment", backbone_pretrained=False, imgsz=320)
    model = DFINE.from_config(cfg).eval()
    assert model.backbone.return_idx == [1, 2, 3]
    assert len(model.decoder.mask_decoder.lateral) == 3
    with torch.no_grad():
        out = model(torch.rand(1, 3, cfg.imgsz, cfg.imgsz))
    m = out["pred_masks"]
    assert m.shape == (1, cfg.num_queries, cfg.imgsz // 4, cfg.imgsz // 4)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_predict_segment_returns_aligned_masks():
    """DFINE(task="segment").predict returns per-instance masks aligned with boxes."""
    import numpy as np
    from PIL import Image

    from dfine.model import DFINE as PublicDFINE

    model = PublicDFINE(size="n", task="segment", backbone_pretrained=False, imgsz=320)
    img = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8"))
    res = model.predict(img, conf=0.0)[0]  # conf=0 keeps all top-k so masks are exercised
    assert res.masks is not None
    assert len(res.masks) == len(res.boxes)  # one mask per detection
    assert res.masks.data.shape == (len(res.boxes), 240, 320)  # original scale
    assert res.masks.data.dtype == torch.bool


def test_predict_detect_has_no_masks():
    """A detection model's results carry no masks (byte-identical detect behavior)."""
    import numpy as np
    from PIL import Image

    from dfine.model import DFINE as PublicDFINE

    model = PublicDFINE(size="n", backbone_pretrained=False, imgsz=320)
    img = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8"))
    assert model.predict(img, conf=0.0)[0].masks is None


def _cached_seg_ckpt():
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg", filename="dfine_seg_n_coco.pt", local_files_only=True
        )
    except Exception:
        return None


def test_segment_model_strict_loads_seg_checkpoint():
    """The whole assembled segment model strict-loads dfine_seg_n_coco.pt (0 miss/0 extra)."""
    path = _cached_seg_ckpt()
    if path is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    cfg = DFINEConfig.preset("n", task="segment", backbone_pretrained=False)  # imgsz=640
    model = DFINE.from_config(cfg)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_from_pretrained_seg_end_to_end():
    """DFINE.from_pretrained("dfine-seg-n") builds the mask head + strict-loads HF weights."""
    if _cached_seg_ckpt() is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    model = DFINE.from_pretrained("dfine-seg-n")
    assert hasattr(model.decoder, "mask_decoder")
    with torch.no_grad():
        out = model(torch.rand(1, 3, 640, 640))
    assert out["pred_masks"].shape[0] == 1 and out["pred_masks"].dim() == 4
