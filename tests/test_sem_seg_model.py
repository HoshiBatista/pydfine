"""SS2: assembled sem_seg model (decoder-slot swap) + argmax label-map postprocess."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import SemSegPostProcessor  # noqa: E402
from dfine.backends.native.dfine import DFINE  # noqa: E402
from dfine.backends.native.dfine_decoder import DFINETransformer  # noqa: E402
from dfine.backends.native.sem_seg_decoder import SemSegDecoder  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402


def test_seg_wiring_fires_for_semseg_nano():
    """Nano sem_seg (no stride-8 encoder level) prepends the backbone stride-8 feature."""
    cfg = DFINEConfig.preset("n", task="sem_seg")
    return_idx, mask_low_level_ch = DFINE._seg_wiring(cfg)
    assert return_idx == [1, 2, 3]
    assert mask_low_level_ch == 256


def test_seg_wiring_noop_for_semseg_with_stride8():
    """s/m/l/x already have a stride-8 level — no low-level plumbing needed."""
    assert DFINE._seg_wiring(DFINEConfig.preset("s", task="sem_seg")) == (None, None)


def test_semseg_model_swaps_decoder_and_forwards_label_logits():
    """A sem_seg model uses SemSegDecoder and returns [B, C, H, W] logits at input res."""
    cfg = DFINEConfig.preset(
        "n", task="sem_seg", num_classes=19, backbone_pretrained=False, imgsz=320
    )
    model = DFINE.from_config(cfg).eval()
    assert isinstance(model.decoder, SemSegDecoder)
    assert model.backbone.return_idx == [1, 2, 3]
    with torch.no_grad():
        out = model(torch.rand(1, 3, cfg.imgsz, cfg.imgsz))
    assert set(out) == {"sem_seg_logits"}
    assert out["sem_seg_logits"].shape == (1, 19, cfg.imgsz, cfg.imgsz)


def test_detect_and_segment_still_use_transformer_decoder():
    """The decoder swap is sem_seg-only; detect/segment keep DFINETransformer."""
    for task in ("detect", "segment"):
        cfg = DFINEConfig.preset("n", task=task, backbone_pretrained=False, imgsz=320)
        assert isinstance(DFINE.from_config(cfg).decoder, DFINETransformer)


def test_semseg_postprocessor_argmax_and_resize_to_original():
    """argmax over classes → NEAREST resize to each image's (W, H) → uint8 [H0, W0]."""
    pp = SemSegPostProcessor().eval()
    logits = torch.randn(2, 5, 16, 16)
    outputs = {"sem_seg_logits": logits}
    orig = torch.tensor([[40, 30], [64, 48]])  # (W, H) per image
    maps = pp(outputs, orig)
    assert isinstance(maps, list) and len(maps) == 2
    assert maps[0].shape == (30, 40) and maps[1].shape == (48, 64)  # (H0, W0)
    assert all(m.dtype == torch.uint8 for m in maps)
    assert int(maps[0].max()) < 5  # label ids are valid class indices


def test_semseg_postprocessor_matches_manual_argmax_on_identity_size():
    """When target size equals the logits size, the map is exactly the per-pixel argmax."""
    pp = SemSegPostProcessor().eval()
    logits = torch.randn(1, 4, 8, 8)
    maps = pp({"sem_seg_logits": logits}, torch.tensor([[8, 8]]))
    expected = logits.argmax(1)[0].to(torch.uint8)
    assert torch.equal(maps[0], expected)


def _cached_seg_ckpt():
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg", filename="dfine_seg_n_coco.pt", local_files_only=True
        )
    except Exception:
        return None


def test_semseg_model_fuser_strict_loads_seg_checkpoint():
    """The assembled sem_seg model's reused fuser strict-loads the checkpoint's fuser."""
    path = _cached_seg_ckpt()
    if path is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    cfg = DFINEConfig.preset("n", task="sem_seg", backbone_pretrained=False)
    model = DFINE.from_config(cfg)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "decoder.mask_decoder."
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.decoder.mask_decoder.load_state_dict(sub, strict=True)
    assert not missing and not unexpected
