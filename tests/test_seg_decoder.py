"""S3: the native DFINETransformer instance-mask branch (D-FINE-seg port)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import DFINETransformer  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402


def _nano_cfg() -> DFINEConfig:
    return DFINEConfig.preset("n", backbone_pretrained=False, freeze_norm=False, freeze_at=-1)


def test_detection_output_unchanged_when_mask_head_off():
    """With the head off the decoder yields exactly the detection keys — no pred_masks."""
    cfg = _nano_cfg()
    dec = DFINETransformer.from_config(cfg).eval()  # enable_mask_head defaults False
    assert not dec.enable_mask_head
    feats = [
        torch.randn(1, ch, cfg.imgsz // s, cfg.imgsz // s)
        for ch, s in zip(cfg.feat_channels, cfg.feat_strides)
    ]
    with torch.no_grad():
        out = dec(feats)
    assert set(out) == {"pred_logits", "pred_boxes"}


def test_mask_branch_forward_shape():
    """Eval-mode mask branch produces sigmoid masks [B, Q, H/4, W/4]."""
    cfg = _nano_cfg()
    low_ch = 64  # arbitrary stride-8 low-level channel count for this smoke test
    dec = DFINETransformer.from_config(
        cfg, enable_mask_head=True, mask_dim=128, mask_low_level_ch=low_ch
    ).eval()  # nano mask_dim is 128
    assert len(dec.mask_decoder.lateral) == cfg.num_levels + 1  # +1 for the low-level feat

    feats = [
        torch.randn(1, ch, cfg.imgsz // s, cfg.imgsz // s)
        for ch, s in zip(cfg.feat_channels, cfg.feat_strides)
    ]
    low_level = torch.randn(1, low_ch, cfg.imgsz // 8, cfg.imgsz // 8)
    with torch.no_grad():
        out = dec(feats, low_level_feat=low_level)
    m = out["pred_masks"]
    assert m.shape == (1, cfg.num_queries, cfg.imgsz // 4, cfg.imgsz // 4)
    assert m.min() >= 0.0 and m.max() <= 1.0  # sigmoid'd at inference


def _cached_seg_ckpt():
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg", filename="dfine_seg_n_coco.pt", local_files_only=True
        )
    except Exception:
        return None


def test_decoder_strict_loads_seg_checkpoint():
    """The whole decoder (det core + mask branch) strict-loads dfine_seg_n_coco.pt."""
    path = _cached_seg_ckpt()
    if path is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    sub = {k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")}
    # Infer the mask dims the checkpoint was built with.
    mask_dim = sub["mask_decoder.lateral.0.weight"].shape[0]
    low_ch = sub["mask_decoder.lateral.0.weight"].shape[1]  # nano: level-0 lateral = low-level

    cfg = _nano_cfg()
    dec = DFINETransformer.from_config(
        cfg, enable_mask_head=True, mask_dim=mask_dim, mask_low_level_ch=low_ch
    )
    missing, unexpected = dec.load_state_dict(sub, strict=True)
    assert not missing and not unexpected
