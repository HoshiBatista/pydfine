"""Tests for DFINE.predict_video.

The real decode/encode round-trip needs OpenCV (skipped when absent); the
missing-OpenCV error path is exercised deterministically via monkeypatch.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from dfine import DFINE  # noqa: E402

IMGSZ = 320
N_FRAMES = 5


def _model():
    return DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False, num_classes=80)


def _write_sample_video(cv2, path, n=N_FRAMES, w=160, h=120, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n):
        frame = (np.random.rand(h, w, 3) * 255).astype("uint8")
        writer.write(frame)
    writer.release()
    return path


def test_predict_video_without_opencv(monkeypatch):
    # Block the cv2 import regardless of whether it's installed.
    monkeypatch.setitem(sys.modules, "cv2", None)
    m = _model()
    with pytest.raises(ImportError, match="OpenCV"):
        m.predict_video("whatever.mp4")


def test_predict_video_writes_annotated_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    m = _model()
    src = _write_sample_video(cv2, tmp_path / "in.mp4")
    out = m.predict_video(src, output=tmp_path / "out.mp4", conf=0.0, imgsz=IMGSZ)
    assert out.exists() and out.stat().st_size > 0


def test_predict_video_stream_yields_results(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from dfine.results import Results

    m = _model()
    src = _write_sample_video(cv2, tmp_path / "in.mp4")
    results = list(m.predict_video(src, conf=0.0, imgsz=IMGSZ, stream=True))
    assert len(results) == N_FRAMES
    assert all(isinstance(r, Results) for r in results)
    assert results[0].orig_shape == (120, 160)  # (h, w)


def test_predict_video_bad_source_raises(tmp_path):
    pytest.importorskip("cv2")
    m = _model()
    with pytest.raises(FileNotFoundError):
        m.predict_video(tmp_path / "does_not_exist.mp4", output=tmp_path / "o.mp4")
