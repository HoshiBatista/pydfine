"""Tests for the native weight loader + assembled DFINE model.

Two layers:
- Offline (always run): round-trip an upstream-style checkpoint through the
  loader and assert a clean ``strict=True`` load with identical outputs, plus
  the ``extract_state_dict`` unwrap rules.
- Parity (opt-in): set ``DFINE_TEST_CKPT`` to a real ``.pth`` (and optionally
  ``DFINE_TEST_SIZE``) to strict-load actual released weights. Skipped when
  unset so CI without weights stays green.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import DFINE, extract_state_dict, load_checkpoint  # noqa: E402

IMGSZ = 320


def _model(num_classes=80, imgsz=IMGSZ):
    cfg = DFINEConfig.preset(
        "n",
        imgsz=imgsz,
        num_classes=num_classes,
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
    )
    return DFINE.from_config(cfg).eval()


# --- extract_state_dict unwrap rules -----------------------------------------


def test_extract_prefers_ema():
    ckpt = {"model": {"a": torch.tensor(1.0)}, "ema": {"module": {"a": torch.tensor(2.0)}}}
    assert extract_state_dict(ckpt, use_ema=True)["a"].item() == 2.0
    assert extract_state_dict(ckpt, use_ema=False)["a"].item() == 1.0


def test_extract_falls_back_to_model_without_ema():
    ckpt = {"model": {"a": torch.tensor(1.0)}}
    assert extract_state_dict(ckpt, use_ema=True)["a"].item() == 1.0


def test_extract_bare_state_dict():
    sd = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    assert extract_state_dict(sd) == sd


def test_extract_strips_module_prefix():
    ckpt = {"model": {"module.a": torch.tensor(1.0), "module.b": torch.tensor(2.0)}}
    out = extract_state_dict(ckpt)
    assert set(out) == {"a", "b"}


def test_extract_rejects_unknown_shape():
    with pytest.raises(KeyError):
        extract_state_dict({"foo": {"nested": "not-a-tensor"}})


# --- round-trip through the assembled model ----------------------------------


def test_roundtrip_strict_load_and_identical_output(tmp_path):
    src = _model()
    # Save as an upstream-style checkpoint (weights under both "model" and EMA).
    sd = src.state_dict()
    ckpt_path = tmp_path / "fake.pth"
    torch.save({"model": sd, "ema": {"module": sd, "updates": 1}}, ckpt_path)

    dst = _model()  # freshly (differently) initialized
    missing, unexpected = load_checkpoint(dst, ckpt_path, use_ema=True, strict=True)
    assert missing == [] and unexpected == []

    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        a, b = src(x), dst(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"])
    assert torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_model_load_method(tmp_path):
    src = _model()
    ckpt_path = tmp_path / "fake.pth"
    torch.save({"model": src.state_dict()}, ckpt_path)
    dst = _model()
    missing, unexpected = dst.load(ckpt_path)
    assert missing == [] and unexpected == []


# --- opt-in real-weight parity -----------------------------------------------
#
# Two ways to supply weights (both skipped when absent, so CI stays green):
#  * DFINE_TEST_CKPT[+DFINE_TEST_SIZE]  — one explicit .pth (any dataset).
#  * DFINE_WEIGHTS_DIR                  — a dir of released .pth; the per-size
#    parametrized test loads whichever files are present.


def _strict_parity(path, size, expect_num_classes=None):
    """Build the matching model, strict-load, run a forward pass. Returns outputs."""
    from dfine.registry import config_for

    sd = extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    num_classes = sd["decoder.enc_score_head.weight"].shape[0]
    if expect_num_classes is not None:
        assert num_classes == expect_num_classes, f"{path}: head has {num_classes} classes"

    # config_for wires size + num_classes exactly like the release (imgsz=640).
    cfg = config_for(f"dfine-{size}", num_classes=num_classes, backbone_pretrained=False)
    model = DFINE.from_config(cfg).eval()
    missing, unexpected = load_checkpoint(model, path, strict=True)
    assert missing == [] and unexpected == [], f"{path}: missing={missing} unexpected={unexpected}"

    with torch.no_grad():
        out = model(torch.randn(1, 3, cfg.imgsz, cfg.imgsz))
    assert out["pred_logits"].shape[-1] == num_classes
    assert torch.isfinite(out["pred_boxes"]).all()
    return out


@pytest.mark.skipif(
    not os.environ.get("DFINE_TEST_CKPT"),
    reason="set DFINE_TEST_CKPT to a real .pth to run weight parity",
)
def test_real_checkpoint_strict_parity():
    _strict_parity(os.environ["DFINE_TEST_CKPT"], os.environ.get("DFINE_TEST_SIZE", "n"))


@pytest.mark.skipif(
    not os.environ.get("DFINE_WEIGHTS_DIR"),
    reason="set DFINE_WEIGHTS_DIR to a dir of released .pth for per-size parity",
)
@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_per_size_coco_parity(size):
    from dfine.registry import resolve_weights

    spec = resolve_weights(size, "coco")
    path = os.path.join(os.environ["DFINE_WEIGHTS_DIR"], spec.filename)
    if not os.path.exists(path):
        pytest.skip(f"{spec.filename} not in DFINE_WEIGHTS_DIR")
    _strict_parity(path, size, expect_num_classes=80)
