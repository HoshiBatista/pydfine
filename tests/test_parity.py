"""Numeric parity against genuine upstream D-FINE.

The fixtures in ``tests/data/parity_<size>.pt`` were produced by running the real
upstream model (``D-FINE/src`` via ``YAMLConfig``) on a deterministic seeded input
— see ``scripts/gen_parity_fixture.py``. Here we build our native port from the
*same* released checkpoint, feed the *same* input, and assert we reproduce upstream's
raw boxes + final postprocessed detections. That is the end-to-end proof that the
port computes what upstream computes, not merely that the weights load.

Gated on ``DFINE_WEIGHTS_DIR`` (a dir holding the released COCO ``.pth``), exactly
like the loader parity test, so CI without weights stays green. Upstream itself is
*not* imported here — only the small committed fixture is.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import DFINE, DFINEPostProcessor, load_checkpoint  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _weights_path(size: str) -> str:
    from dfine.registry import resolve_weights

    spec = resolve_weights(size, "coco")
    return os.path.join(os.environ.get("DFINE_WEIGHTS_DIR", ""), spec.filename)


@pytest.mark.skipif(
    not os.environ.get("DFINE_WEIGHTS_DIR"),
    reason="set DFINE_WEIGHTS_DIR to a dir of released .pth for upstream parity",
)
@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_upstream_numeric_parity(size):
    fixture_path = os.path.join(_DATA, f"parity_{size}.pt")
    if not os.path.exists(fixture_path):
        pytest.skip(f"no parity fixture for size {size!r}")
    ckpt = _weights_path(size)
    if not os.path.exists(ckpt):
        pytest.skip(f"checkpoint for {size!r} not in DFINE_WEIGHTS_DIR")

    ref = torch.load(fixture_path, map_location="cpu", weights_only=False)

    # Build our native port + postprocessor for the matching checkpoint and load it.
    from dfine.registry import config_for

    cfg = config_for(f"dfine-{size}", backbone_pretrained=False)
    assert cfg.imgsz == ref["imgsz"], "fixture built at a different resolution"
    model = DFINE.from_config(cfg).eval()
    postproc = DFINEPostProcessor.from_config(cfg).eval()
    missing, unexpected = load_checkpoint(model, ckpt, strict=True)
    assert missing == [] and unexpected == [], f"missing={missing} unexpected={unexpected}"

    # Reproduce the exact upstream input, then run model + postprocessor.
    torch.manual_seed(ref["seed"])
    x = torch.rand(1, 3, ref["imgsz"], ref["imgsz"])
    with torch.no_grad():
        out = model(x)
        det = postproc(out, torch.tensor(ref["orig_size"]))[0]

    # Raw regression head output over all 300 queries (normalized cxcywh).
    assert torch.allclose(out["pred_boxes"], ref["pred_boxes"], atol=1e-4, rtol=1e-4), (
        f"raw pred_boxes diverge (max abs "
        f"{(out['pred_boxes'] - ref['pred_boxes']).abs().max():.2e})"
    )

    # Final postprocessed detections: labels must match exactly, scores/boxes closely.
    assert torch.equal(det["labels"], ref["labels"]), "predicted labels diverge"
    assert torch.allclose(det["scores"], ref["scores"], atol=1e-4, rtol=1e-4), (
        f"scores diverge (max abs {(det['scores'] - ref['scores']).abs().max():.2e})"
    )
    assert torch.allclose(det["boxes"], ref["boxes"], atol=1e-3, rtol=1e-4), (
        f"boxes diverge (max abs {(det['boxes'] - ref['boxes']).abs().max():.2e})"
    )
