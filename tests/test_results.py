"""Tests for Results/Boxes containers and plot/save."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from dfine.results import Boxes, Results  # noqa: E402


def _results(n=2):
    img = Image.fromarray((np.zeros((64, 96, 3))).astype("uint8"))
    boxes = Boxes(
        xyxy=torch.tensor([[1.0, 1.0, 20.0, 20.0], [5.0, 5.0, 40.0, 30.0]][:n]),
        conf=torch.tensor([0.9, 0.5][:n]),
        cls=torch.tensor([0, 2][:n]),
    )
    return Results(img, boxes, names={0: "person", 2: "car"})


def test_boxes_len_and_iter():
    r = _results(2)
    assert len(r) == 2 and len(r.boxes) == 2
    rows = list(r.boxes)
    assert len(rows) == 2
    xyxy, conf, cls = rows[0]
    assert xyxy.shape == (4,) and float(conf) == pytest.approx(0.9) and int(cls) == 0


def test_results_orig_shape_and_repr():
    r = _results()
    assert r.orig_shape == (64, 96)  # (h, w)
    assert "boxes=2" in repr(r)


def test_plot_returns_rgb_array():
    arr = _results().plot()
    assert arr.shape == (64, 96, 3) and arr.dtype == np.uint8


def test_save_writes_file(tmp_path):
    out = _results().save(tmp_path / "out.jpg")
    assert out.exists() and out.stat().st_size > 0


def test_empty_results_plot():
    r = _results(0)
    assert len(r) == 0
    assert r.plot().shape == (64, 96, 3)  # no boxes -> unchanged image
