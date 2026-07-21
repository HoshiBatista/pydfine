"""SS1: shape, aux, from_config, and fuser checkpoint-parity for SemSegDecoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import SemSegDecoder  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402


def _feats(channels, base):
    """Encoder PAN features, finest first: level i is at spatial size base // 2**i."""
    return [torch.randn(2, c, base // (2**i), base // (2**i)) for i, c in enumerate(channels)]


def test_eval_forward_upsamples_logits_to_input_res():
    """Eval returns only `sem_seg_logits` at full input resolution (fuser 1/4 → ×4)."""
    dec = SemSegDecoder(num_classes=19, feat_channels=[256, 256, 256], mask_dim=256).eval()
    base = 40  # stride-8 level → input side 320
    with torch.no_grad():
        out = dec(_feats([256, 256, 256], base))
    assert set(out) == {"sem_seg_logits"}
    assert out["sem_seg_logits"].shape == (2, 19, base * 8, base * 8)


def test_train_forward_adds_aux_logits():
    """Training adds `sem_seg_logits_aux` at the same resolution as the main logits."""
    dec = SemSegDecoder(num_classes=8, feat_channels=[256, 256, 256], mask_dim=256).train()
    out = dec(_feats([256, 256, 256], 40))
    assert set(out) == {"sem_seg_logits", "sem_seg_logits_aux"}
    assert out["sem_seg_logits_aux"].shape == out["sem_seg_logits"].shape


def test_nano_low_level_feat_prepended_to_fuser():
    """Nano has no stride-8 encoder level: the low-level feat feeds the fuser as level 0."""
    dec = SemSegDecoder(
        num_classes=19, feat_channels=[128, 128], mask_dim=128, mask_low_level_ch=256
    ).eval()
    assert len(dec.mask_decoder.lateral) == 3  # 256 (low-level) + [128, 128]
    low = torch.randn(2, 256, 80, 80)  # stride-8 → input side 640
    feats = [torch.randn(2, 128, 40, 40), torch.randn(2, 128, 20, 20)]  # stride 16/32
    with torch.no_grad():
        out = dec(feats, low_level_feat=low)
    assert out["sem_seg_logits"].shape == (2, 19, 640, 640)


def test_from_config_matches_preset_dims():
    """`from_config` wires num_classes/feat_channels/mask_dim (+ derived low-level ch)."""
    cfg = DFINEConfig.preset("n", task="sem_seg", num_classes=19)
    dec = SemSegDecoder.from_config(cfg, mask_low_level_ch=256).eval()
    assert dec.classifier.out_channels == 19
    assert len(dec.mask_decoder.lateral) == 3  # nano fuser: low-level + 2 encoder levels
    assert dec.mask_decoder.lateral[0].out_channels == cfg.mask_dim == 128


def _cached_seg_ckpt():
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg", filename="dfine_seg_n_coco.pt", local_files_only=True
        )
    except Exception:
        return None


def test_fuser_strict_loads_from_seg_checkpoint():
    """The reused `mask_decoder` fuser strict-loads dfine_seg_n_coco.pt's decoder.mask_decoder.*.

    The instance-seg checkpoint carries no sem_seg neck/classifier (those train from
    scratch); this pins that the shared fuser transfers with 0 missing / 0 unexpected.
    """
    path = _cached_seg_ckpt()
    if path is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "decoder.mask_decoder."
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    assert sub, "checkpoint has no decoder.mask_decoder.* keys"

    cfg = DFINEConfig.preset("n", task="sem_seg")
    dec = SemSegDecoder.from_config(cfg, mask_low_level_ch=256)
    missing, unexpected = dec.mask_decoder.load_state_dict(sub, strict=True)
    assert not missing and not unexpected
