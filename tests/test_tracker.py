"""Tests for the vendored ByteTrack tracker and predict_video(track=True).

The core tracker is fed synthetic per-frame Results (no model needed) so the
association behaviour is deterministic; a light integration test drives the real
model over a tiny video. Needs scipy (assignment) — skipped without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from PIL import Image  # noqa: E402

from dfine.results import Boxes, Results  # noqa: E402
from dfine.track import ByteTrack  # noqa: E402

IMGSZ = 320
N_FRAMES = 5


def _result(boxes_xyxy, scores=None, cls=None, wh=(200, 200)):
    img = Image.fromarray(np.zeros((wh[1], wh[0], 3), dtype="uint8"))
    n = len(boxes_xyxy)
    return Results(
        img,
        Boxes(
            xyxy=torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
            conf=torch.tensor(scores if scores is not None else [0.9] * n, dtype=torch.float32),
            cls=torch.tensor(cls if cls is not None else [0] * n),
        ),
        names={0: "obj"},
    )


# --- Kalman filter ------------------------------------------------------------


def test_kalman_zero_velocity_keeps_center_grows_uncertainty():
    from dfine.track.kalman_filter import KalmanFilterXYAH

    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([50.0, 60.0, 1.0, 20.0]))
    mean2, cov2 = kf.predict(mean, cov)
    assert np.allclose(mean2[:4], mean[:4])  # no velocity yet -> position unchanged
    assert np.trace(cov2) > np.trace(cov)  # prediction inflates uncertainty


# --- association behaviour ----------------------------------------------------


def test_track_id_stable_for_moving_box():
    tracker = ByteTrack(frame_rate=30)
    ids = []
    for i in range(6):
        x = 10.0 + i * 5
        r = tracker.update(_result([[x, 20, x + 30, 60]]))
        if len(r.boxes):
            ids.append(int(r.boxes.id[0]))
    assert len(ids) >= 5
    assert len(set(ids)) == 1  # one object -> one stable id across frames


def test_two_objects_get_distinct_ids():
    tracker = ByteTrack(frame_rate=30)
    seen: set[int] = set()
    for i in range(5):
        x = 10.0 + i * 4
        r = tracker.update(_result([[x, 20, x + 30, 60], [x + 100, 120, x + 130, 160]]))
        if len(r.boxes) == 2:
            seen = {int(v) for v in r.boxes.id}
    assert len(seen) == 2


def test_track_empty_frame_returns_empty_ids():
    tracker = ByteTrack()
    r = tracker.update(_result([]))
    assert len(r.boxes) == 0
    assert r.boxes.id is not None and len(r.boxes.id) == 0


def test_ids_reset_per_tracker_instance():
    # Two independent trackers both start numbering from 1 (deterministic).
    a = ByteTrack().update(_result([[10, 10, 40, 40]]))
    b = ByteTrack().update(_result([[10, 10, 40, 40]]))
    assert int(a.boxes.id[0]) == int(b.boxes.id[0]) == 1


# --- rendering ----------------------------------------------------------------


def test_label_includes_track_id_prefix():
    r = _result([[1, 1, 20, 20]])
    assert r._label(0, 0.9, track_id=7).startswith("#7 ")
    assert r._label(0, 0.9).startswith("obj ")  # no prefix without an id


def test_plot_with_ids_runs():
    img = Image.fromarray(np.zeros((60, 80, 3), dtype="uint8"))
    boxes = Boxes(
        xyxy=torch.tensor([[1.0, 1.0, 20.0, 20.0]]),
        conf=torch.tensor([0.9]),
        cls=torch.tensor([0]),
        id=torch.tensor([3]),
    )
    arr = Results(img, boxes, {0: "obj"}).plot()
    assert arr.shape == (60, 80, 3) and arr.dtype == np.uint8


# --- integration with predict_video ------------------------------------------


def _write_sample_video(cv2, path, n=N_FRAMES, w=160, h=120, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n):
        writer.write((np.random.rand(h, w, 3) * 255).astype("uint8"))
    writer.release()
    return path


def test_predict_video_track_stream_sets_ids(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from dfine import DFINE

    m = DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False, num_classes=80)
    src = _write_sample_video(cv2, tmp_path / "in.mp4")
    results = list(m.predict_video(src, conf=0.0, imgsz=IMGSZ, stream=True, track=True))
    assert len(results) == N_FRAMES
    # Tracking is on: every frame's boxes carry an id array aligned with the boxes.
    for r in results:
        assert r.boxes.id is not None
        assert len(r.boxes.id) == len(r.boxes)
