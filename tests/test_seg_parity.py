"""Numeric parity against genuine upstream D-FINE-seg (instance segmentation).

The fixture ``tests/data/seg_parity_<size>.pt`` was produced by running the real
D-FINE-seg model (``D-FINE-seg/src`` via ``build_model``) on a deterministic seeded
input — see ``scripts/gen_seg_parity_fixture.py``. Here we build our native port from
the *same* released seg checkpoint, feed the *same* input, and assert we reproduce
its raw detection logits/boxes **and** its instance-mask maps. That is the end-to-end
proof that our mask branch computes what D-FINE-seg computes, not merely that the seg
weights load.

Gated on the seg checkpoint being present in the local Hugging Face cache (populated
by the S1 probe / ``from_pretrained``), so CI without weights stays green. D-FINE-seg
itself is *not* imported here — only the small committed fixture is.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native.dfine import DFINE  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _cached_seg_ckpt(size: str) -> str | None:
    hf = pytest.importorskip("huggingface_hub")
    try:
        return hf.hf_hub_download(
            repo_id="ArgoSA/D-FINE-seg",
            filename=f"dfine_seg_{size}_coco.pt",
            local_files_only=True,
        )
    except Exception:
        return None


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_seg_numeric_parity(size):
    fixture_path = os.path.join(_DATA, f"seg_parity_{size}.pt")
    if not os.path.exists(fixture_path):
        pytest.skip(f"no seg parity fixture for size {size!r}")
    ckpt = _cached_seg_ckpt(size)
    if ckpt is None:
        pytest.skip(f"dfine_seg_{size}_coco.pt not cached — run the S1 probe to populate it")

    ref = torch.load(fixture_path, map_location="cpu", weights_only=True)

    cfg = DFINEConfig.preset(size, task="segment", backbone_pretrained=False)
    assert cfg.imgsz == ref["imgsz"], "fixture built at a different resolution"
    model = DFINE.from_config(cfg).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    torch.manual_seed(ref["seed"])
    x = torch.rand(1, 3, ref["imgsz"], ref["imgsz"])
    with torch.no_grad():
        out = model(x)

    # Raw detection head: logits + boxes over all queries (postprocessor-independent).
    assert torch.allclose(out["pred_logits"], ref["pred_logits"], atol=1e-4, rtol=1e-4), (
        f"pred_logits diverge (max abs {(out['pred_logits'] - ref['pred_logits']).abs().max():.2e})"
    )
    assert torch.allclose(out["pred_boxes"], ref["pred_boxes"], atol=1e-4, rtol=1e-4), (
        f"pred_boxes diverge (max abs {(out['pred_boxes'] - ref['pred_boxes']).abs().max():.2e})"
    )

    # Instance-mask maps for the stored query slice (sigmoid probs at eval). The
    # reference is stored fp16 to keep the fixture small, so the tolerance covers
    # that rounding on top of any numeric drift.
    k = ref["mask_k"]
    ours = out["pred_masks"][0, :k].float()
    assert torch.allclose(ours, ref["pred_masks"].float(), atol=3e-3, rtol=1e-3), (
        f"pred_masks diverge (max abs {(ours - ref['pred_masks'].float()).abs().max():.2e})"
    )
