"""Shape + checkpoint-key parity tests for the native MaskDecoder (D-FINE-seg port)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import MaskDecoder  # noqa: E402


@pytest.mark.parametrize(
    "in_chs, out_ch",
    [
        ([128, 128], 128),  # nano encoder levels (mask_dim 128)
        ([64, 128, 128], 128),  # nano + prepended stride-8 low-level feat
        ([256, 256, 256], 256),  # s/m/l
        ([384, 384, 384], 256),  # x (encoder feat_channels 384, mask_dim 256)
    ],
)
def test_forward_shape_is_quarter_res(in_chs, out_ch):
    md = MaskDecoder(in_chs=in_chs, out_ch=out_ch).eval()
    # feats[0] is the finest level; output is 2x its spatial size (1/8 -> 1/4).
    base = 40  # e.g. 320/8
    feats = [torch.randn(2, c, base // (2**i), base // (2**i)) for i, c in enumerate(in_chs)]
    with torch.no_grad():
        out = md(feats)
    assert out.shape == (2, out_ch, base * 2, base * 2)
    assert len(md.lateral) == len(in_chs) == len(md.bn)


def _cached_seg_ckpt():
    """Return the cached ``dfine_seg_n_coco.pt`` path, or None if not downloaded."""
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg",
            filename="dfine_seg_n_coco.pt",
            local_files_only=True,
        )
    except Exception:
        return None


def test_state_dict_keys_match_seg_checkpoint():
    """The ported module's keys/shapes must equal the checkpoint's decoder.mask_decoder.*."""
    path = _cached_seg_ckpt()
    if path is None:
        pytest.skip("dfine_seg_n_coco.pt not cached — run the S1 probe to populate it")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "decoder.mask_decoder."
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    assert sub, "checkpoint has no decoder.mask_decoder.* keys"

    # Infer in_chs from the lateral 1x1 conv weights, out_ch from lateral[0] out-channels.
    n_levels = len({k.split(".")[1] for k in sub if k.startswith("lateral.")})
    in_chs = [sub[f"lateral.{i}.weight"].shape[1] for i in range(n_levels)]
    out_ch = sub["lateral.0.weight"].shape[0]

    md = MaskDecoder(in_chs=in_chs, out_ch=out_ch)
    missing, unexpected = md.load_state_dict(sub, strict=True)
    assert not missing and not unexpected
