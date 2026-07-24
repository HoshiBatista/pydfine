"""Tests for Results/Boxes containers and plot/save."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from dfine.results import Boxes, Masks, Results  # noqa: E402


def _results(n=2, masks=False):
    img = Image.fromarray((np.zeros((64, 96, 3))).astype("uint8"))
    boxes = Boxes(
        xyxy=torch.tensor([[1.0, 1.0, 20.0, 20.0], [5.0, 5.0, 40.0, 30.0]][:n]),
        conf=torch.tensor([0.9, 0.5][:n]),
        cls=torch.tensor([0, 2][:n]),
    )
    masks_obj = None
    if masks:
        data = torch.zeros((n, 64, 96), dtype=torch.bool)
        for i in range(n):
            data[i, 2 : 20 + i, 2 : 20 + i] = True  # a filled block per instance
        masks_obj = Masks(data)
    return Results(img, boxes, names={0: "person", 2: "car"}, masks=masks_obj)


def test_masks_container_len_iter_repr():
    r = _results(2, masks=True)
    assert len(r.masks) == 2
    assert r.masks.data.shape == (2, 64, 96) and r.masks.data.dtype == torch.bool
    assert list(r.masks)[0].shape == (64, 96)
    assert "96x64" in repr(r.masks)


def test_plot_with_masks_overlays_and_keeps_shape():
    arr = _results(2, masks=True).plot()
    assert arr.shape == (64, 96, 3) and arr.dtype == np.uint8
    assert arr.sum() > 0  # mask overlay tinted some pixels on the black image


def test_to_supervision_attaches_masks():
    sv = pytest.importorskip("supervision")  # noqa: F841
    det = _results(2, masks=True).to_supervision()
    assert det.mask is not None and det.mask.shape == (2, 64, 96) and det.mask.dtype == bool


def test_detection_results_have_no_masks():
    r = _results(2)
    assert r.masks is None
    assert _results(2).to_supervision().mask is None


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


def test_to_coco():
    r = _results(2)
    dets = r.to_coco(image_id=7)
    assert [d["image_id"] for d in dets] == [7, 7]
    assert [d["category_id"] for d in dets] == [0, 2]
    # xyxy [1,1,20,20] -> xywh [1,1,19,19]
    assert dets[0]["bbox"] == [1.0, 1.0, 19.0, 19.0]
    assert dets[0]["score"] == pytest.approx(0.9)
    assert dets[1]["bbox"] == [5.0, 5.0, 35.0, 25.0]


def test_to_coco_empty():
    assert _results(0).to_coco() == []


def test_to_coco_detection_has_no_segmentation():
    for d in _results(2).to_coco():
        assert "segmentation" not in d  # boxes-only stays exactly as before


def test_to_coco_masks_emit_rle_segmentation():
    r = _results(2, masks=True)
    dets = r.to_coco(image_id=3)
    assert len(dets) == 2
    for i, d in enumerate(dets):
        seg = d["segmentation"]
        assert seg["size"] == [64, 96]  # [H, W], original scale
        assert isinstance(seg["counts"], list) and sum(seg["counts"]) == 64 * 96
        # RLE decodes back to the exact foreground pixel count (block is (18+i)^2).
        fg = sum(seg["counts"][1::2])  # 1-runs are the odd-indexed counts
        assert fg == int(r.masks.data[i].sum()) == (18 + i) ** 2


def test_to_coco_rle_roundtrips_through_coco():
    """The uncompressed RLE is accepted by faster-coco-eval and decodes to our mask.

    Uncompressed (list-``counts``) RLE is normalized to compressed RLE via the standard
    COCO ``frPyObjects`` call before ``decode`` — the same path COCO uses for list-form
    segmentations in ground-truth JSON.
    """
    mask_utils = pytest.importorskip("faster_coco_eval.core.mask")
    r = _results(2, masks=True)
    for i, d in enumerate(r.to_coco()):
        seg = d["segmentation"]
        compressed = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
        decoded = mask_utils.decode(compressed)  # [H, W] uint8
        assert np.array_equal(decoded.astype(bool), r.masks.data[i].cpu().numpy())


def test_to_pandas():
    pytest.importorskip("pandas")
    df = _results(2).to_pandas()
    assert list(df.columns) == ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    assert len(df) == 2
    assert df.iloc[0]["name"] == "person" and int(df.iloc[0]["class"]) == 0
    assert df.iloc[1]["xmax"] == pytest.approx(40.0)


def test_to_pandas_empty_keeps_columns():
    pytest.importorskip("pandas")
    df = _results(0).to_pandas()
    assert len(df) == 0
    assert list(df.columns) == ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]


def test_to_supervision():
    sv = pytest.importorskip("supervision")
    det = _results(2).to_supervision()
    assert isinstance(det, sv.Detections)
    assert det.xyxy.shape == (2, 4)
    assert list(det.class_id) == [0, 2]
    assert det.confidence[0] == pytest.approx(0.9)


def test_to_supervision_empty():
    pytest.importorskip("supervision")
    det = _results(0).to_supervision()
    assert det.xyxy.shape == (0, 4)
    assert len(det) == 0
