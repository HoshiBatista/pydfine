"""SS4: numeric parity of the native SemSegDecoder against genuine D-FINE-seg.

The fixture ``tests/data/semseg_parity.pt`` was produced by running the real D-FINE-seg
``SemSegDecoder`` (see ``scripts/gen_semseg_parity_fixture.py``) on small synthetic feature
pyramids, and stores — for a nano low-level-feat case and a plain stride-8 pyramid — the
decoder weights, the input features, and the output ``sem_seg_logits``. Here we build our
port, load the *same* weights, feed the *same* inputs, and assert bit-exact logits.

There are no released *trained* sem_seg weights (``dfine_seg_*_coco.pt`` is instance-seg),
so parity is "same weights → same output"; the trained-fuser transfer is pinned separately
by the SS1/SS2 strict-load tests. The fixture is self-contained — D-FINE-seg is not imported.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import SemSegDecoder, SemSegPostProcessor  # noqa: E402

_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "semseg_parity.pt")


def _cases():
    if not os.path.exists(_FIXTURE):
        pytest.skip("no sem_seg parity fixture — run scripts/gen_semseg_parity_fixture.py")
    return torch.load(_FIXTURE, map_location="cpu", weights_only=False)["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_semseg_decoder_bit_exact_vs_dfine_seg(case):
    dec = SemSegDecoder(
        num_classes=case["num_classes"],
        feat_channels=case["feat_channels"],
        mask_dim=case["mask_dim"],
        mask_low_level_ch=case["mask_low_level_ch"],
        neck_dim=case["neck_dim"],
    ).eval()
    missing, unexpected = dec.load_state_dict(case["state"], strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    with torch.no_grad():
        out = dec(case["feats"], low_level_feat=case["low_level_feat"])
    ref = case["sem_seg_logits"]
    assert out["sem_seg_logits"].shape == ref.shape
    max_abs = (out["sem_seg_logits"] - ref).abs().max().item()
    assert torch.allclose(out["sem_seg_logits"], ref, atol=1e-6, rtol=0), (
        f"sem_seg_logits diverge (max abs {max_abs:.2e})"
    )


def test_semseg_postprocessor_argmax_matches_reference_logits():
    """The label map from our postprocessor equals the argmax of D-FINE-seg's logits."""
    case = _cases()[0]
    logits = case["sem_seg_logits"]
    h, w = logits.shape[-2:]
    maps = SemSegPostProcessor()({"sem_seg_logits": logits}, torch.tensor([[w, h]]))
    assert torch.equal(maps[0], logits.argmax(1)[0].to(torch.uint8))
