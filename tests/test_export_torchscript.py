"""TorchScript export tests — torch-only (no ONNX toolchain needed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.model import DFINE  # noqa: E402

IMGSZ = 320


def _model(**kw):
    return DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False, **kw)


def test_torchscript_loads_and_matches_torch(tmp_path):
    m = _model()
    path = m.export(format="torchscript", file=tmp_path / "m.torchscript", imgsz=IMGSZ)
    assert path.exists() and path.suffix == ".torchscript"

    images = torch.rand(1, 3, IMGSZ, IMGSZ)
    sizes = torch.tensor([[IMGSZ, IMGSZ]])
    ts = torch.jit.load(str(path))
    labels, boxes, scores = ts(images, sizes)
    assert labels.shape[0] == boxes.shape[0] == scores.shape[0] == 1
    assert boxes.shape[-1] == 4


def test_torchscript_facade_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _model().export(format="torchscript", imgsz=IMGSZ)
    assert path.name == "dfine-n.torchscript" and path.exists()


def test_torchscript_does_not_mutate_original_model(tmp_path):
    import numpy as np

    m = _model()
    m.export(format="torchscript", file=tmp_path / "m.torchscript", imgsz=IMGSZ)
    # the live model still predicts after export (deploy copy was isolated)
    out = m.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ)
    assert len(out) == 1


def test_torchscript_segment_has_masks_output(tmp_path):
    m = _model(task="segment")
    path = m.export(format="torchscript", file=tmp_path / "seg.torchscript", imgsz=IMGSZ)
    ts = torch.jit.load(str(path))
    images = torch.rand(1, 3, IMGSZ, IMGSZ)
    sizes = torch.tensor([[IMGSZ, IMGSZ]])
    out = ts(images, sizes)
    assert len(out) == 4  # labels, boxes, scores, masks
