"""SS3: the public sem_seg surface — Results.sem_seg, palette plot, DFINE.predict."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from dfine.results import Boxes, Results, SemSeg  # noqa: E402


def _empty_boxes() -> Boxes:
    return Boxes(
        xyxy=torch.zeros((0, 4)), conf=torch.zeros((0,)), cls=torch.zeros((0,), dtype=torch.long)
    )


def _semseg_results(h=32, w=48) -> Results:
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    data = torch.zeros((h, w), dtype=torch.uint8)
    data[:, : w // 2] = 1  # left half class 1
    data[: h // 2, :] = 255  # top half void/ignore
    return Results(img, _empty_boxes(), names={0: "bg", 1: "road"}, sem_seg=SemSeg(data))


def test_semseg_container_shape_and_repr():
    data = torch.zeros((30, 40), dtype=torch.uint8)
    data[0, 0] = 1
    data[1, 1] = 255  # void — excluded from the class count
    ss = SemSeg(data)
    assert ss.shape == (30, 40)
    assert "40x30" in repr(ss) and "2 classes" in repr(ss)  # {0, 1}, 255 ignored


def test_plot_overlays_palette_and_skips_void():
    r = _semseg_results()
    arr = r.plot()
    assert arr.shape == (32, 48, 3) and arr.dtype == np.uint8
    # class-1 pixels in the bottom-left get tinted; top (void) stays black.
    assert arr[24, 8].sum() > 0  # bottom-left, class 1 → tinted
    assert arr[4, 40].sum() == 0  # top-right, void (255) → untouched
    assert "sem_seg=48x32" in repr(r)


def test_semseg_result_has_no_boxes_or_masks():
    r = _semseg_results()
    assert len(r.boxes) == 0 and r.masks is None
    assert r.sem_seg is not None


def test_predict_semseg_returns_label_map_at_original_scale():
    from dfine.model import DFINE as PublicDFINE

    model = PublicDFINE(
        size="n", task="sem_seg", num_classes=19, backbone_pretrained=False, imgsz=320
    )
    img = Image.fromarray((np.random.rand(240, 360, 3) * 255).astype("uint8"))
    res = model.predict(img)[0]
    assert res.sem_seg is not None
    assert res.sem_seg.data.shape == (240, 360)  # original (H, W)
    assert res.sem_seg.data.dtype == torch.uint8
    assert int(res.sem_seg.data.max()) < 19  # valid class ids
    assert len(res.boxes) == 0 and res.masks is None


def test_predict_detect_has_no_sem_seg():
    from dfine.model import DFINE as PublicDFINE

    model = PublicDFINE(size="n", backbone_pretrained=False, imgsz=320)
    img = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8"))
    assert model.predict(img, conf=0.0)[0].sem_seg is None
