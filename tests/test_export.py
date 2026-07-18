"""ONNX export tests (Phase 3).

Exports a small D-FINE model and checks the graph is valid, runs on onnxruntime, and
matches the torch deploy path numerically; also covers the dynamic batch dim and the
`DFINE.export` facade. Needs onnx + onnxruntime (dev/export extra).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from dfine import DFINE  # noqa: E402
from dfine.export.onnx import DeployModel, export_onnx, tensorrt_command  # noqa: E402

# Small but >= the decoder's 300-query top-k needs (320px -> 500 encoder tokens).
IMGSZ = 320


def _model(**kw):
    return DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False, **kw)


def _ort_run(path, images, sizes):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(
        ["labels", "boxes", "scores"],
        {"images": images.numpy(), "orig_target_sizes": sizes.numpy()},
    )
    return dict(zip(("labels", "boxes", "scores"), out))


def test_export_produces_valid_graph_with_named_io(tmp_path):
    m = _model()
    path = export_onnx(m.model, m.postprocessor, tmp_path / "m.onnx", imgsz=IMGSZ)
    assert path.exists()

    import onnx

    graph = onnx.load(str(path)).graph
    assert [i.name for i in graph.input] == ["images", "orig_target_sizes"]
    assert [o.name for o in graph.output] == ["labels", "boxes", "scores"]


def test_onnxruntime_matches_torch(tmp_path):
    m = _model()
    ref = DeployModel(m.model, m.postprocessor).eval()  # same deploy graph, torch side
    path = export_onnx(m.model, m.postprocessor, tmp_path / "m.onnx", imgsz=IMGSZ)

    images = torch.rand(1, 3, IMGSZ, IMGSZ)
    sizes = torch.tensor([[IMGSZ, IMGSZ]])
    with torch.no_grad():
        t_labels, t_boxes, t_scores = ref(images, sizes)
    got = _ort_run(path, images, sizes)

    assert got["labels"].shape == (1, 300)
    assert got["boxes"].shape == (1, 300, 4)
    assert got["scores"].shape == (1, 300)
    assert np.isfinite(got["boxes"]).all()
    # Sorted scores compare cleanly regardless of any topk index ties.
    np.testing.assert_allclose(
        np.sort(got["scores"], axis=1), np.sort(t_scores.numpy(), axis=1), atol=1e-4
    )
    assert abs(float(got["scores"].max()) - float(t_scores.max())) < 1e-4


def test_dynamic_batch_dim(tmp_path):
    m = _model()
    path = export_onnx(m.model, m.postprocessor, tmp_path / "m.onnx", imgsz=IMGSZ, batch=1)
    # Exported with batch=1 but the dynamic axis lets a batch of 3 run.
    images = torch.rand(3, 3, IMGSZ, IMGSZ)
    sizes = torch.tensor([[IMGSZ, IMGSZ]] * 3)
    got = _ort_run(path, images, sizes)
    assert got["boxes"].shape == (3, 300, 4)
    assert got["scores"].shape == (3, 300)


def test_static_batch_when_dynamic_false(tmp_path):
    m = _model()
    path = export_onnx(m.model, m.postprocessor, tmp_path / "m.onnx", imgsz=IMGSZ, dynamic=False)
    import onnx

    dim = onnx.load(str(path)).graph.input[0].type.tensor_type.shape.dim[0]
    assert dim.dim_value == 1  # fixed batch, not a symbolic "N"


def test_export_does_not_mutate_original_model(tmp_path):
    m = _model()
    export_onnx(m.model, m.postprocessor, tmp_path / "m.onnx", imgsz=IMGSZ)
    # deepcopy inside DeployModel -> the live model still predicts.
    out = m.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))
    assert len(out) == 1


def test_facade_default_filename_and_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _model()
    path = m.export(imgsz=IMGSZ)
    assert path.name == "dfine-n.onnx"
    assert path.exists()


def test_facade_rejects_unknown_format():
    m = _model()
    with pytest.raises(ValueError, match="only 'onnx'"):
        m.export(format="tensorrt")


def test_tensorrt_command_builder():
    cmd = tensorrt_command("model.onnx", fp16=True)
    assert cmd.startswith("trtexec ")
    assert "--onnx=model.onnx" in cmd
    assert "--fp16" in cmd
    assert "--minShapes=images:" in cmd
    assert "3x640x640" in cmd  # default imgsz
    assert "--fp16" not in tensorrt_command("m.onnx", fp16=False)


def test_tensorrt_command_uses_imgsz_and_max_batch():
    cmd = tensorrt_command("m.onnx", imgsz=320, max_batch=8)
    assert "--minShapes=images:1x3x320x320" in cmd
    assert "--optShapes=images:1x3x320x320" in cmd
    assert "--maxShapes=images:8x3x320x320" in cmd
    assert "640" not in cmd  # no hardcoded resolution leaking through


def test_cli_export(tmp_path):
    from dfine.cli import main

    out = tmp_path / "m.onnx"
    rc = main(["export", "n", "--imgsz", str(IMGSZ), "--file", str(out)])
    assert rc == 0
    assert out.exists()
