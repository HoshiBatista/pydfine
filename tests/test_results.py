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


def test_save_crop_writes_per_class_crops(tmp_path):
    paths = _results(2).save_crop(tmp_path, file_name="frame.png")
    assert len(paths) == 2 and all(p.exists() for p in paths)
    # one crop per class subfolder, named by the class
    assert (tmp_path / "person" / "frame.png").exists()
    assert (tmp_path / "car" / "frame.png").exists()
    # crop size matches the box: xyxy [1,1,20,20] → 19x19
    crop = Image.open(tmp_path / "person" / "frame.png")
    assert crop.size == (19, 19)


def test_save_crop_dedups_same_class(tmp_path):
    img = Image.fromarray(np.zeros((64, 96, 3), "uint8"))
    boxes = Boxes(
        xyxy=torch.tensor([[1.0, 1.0, 20.0, 20.0], [5.0, 5.0, 30.0, 30.0]]),
        conf=torch.tensor([0.9, 0.8]),
        cls=torch.tensor([0, 0]),  # both "person"
    )
    paths = Results(img, boxes, names={0: "person"}).save_crop(tmp_path)
    assert len(paths) == 2
    assert {p.name for p in paths} == {"im.jpg", "im_2.jpg"}  # second avoided a clobber


def test_save_crop_empty_returns_empty(tmp_path):
    assert _results(0).save_crop(tmp_path) == []
    assert not any(tmp_path.iterdir())  # nothing written


def test_save_crop_clips_box_to_image(tmp_path):
    img = Image.fromarray(np.zeros((64, 96, 3), "uint8"))
    # box spills past the right/bottom edges → clipped to the 96x64 frame
    boxes = Boxes(
        torch.tensor([[90.0, 60.0, 200.0, 200.0]]), torch.tensor([0.9]), torch.tensor([0])
    )
    (path,) = Results(img, boxes, names={0: "person"}).save_crop(tmp_path)
    assert Image.open(path).size == (6, 4)  # (96-90, 64-60)


def test_summary_detection_layout():
    rows = _results(2).summary()
    assert [r["class"] for r in rows] == [0, 2]
    assert [r["name"] for r in rows] == ["person", "car"]
    assert rows[0]["confidence"] == pytest.approx(0.9)
    # pixel-space box corners by default; xyxy [1,1,20,20]
    assert rows[0]["box"] == {"x1": 1.0, "y1": 1.0, "x2": 20.0, "y2": 20.0}
    assert "track_id" not in rows[0] and "segments" not in rows[0]


def test_summary_normalize_puts_box_in_unit_range():
    rows = _results(2).summary(normalize=True)
    for r in rows:
        assert all(0.0 <= v <= 1.0 for v in r["box"].values())
    # x1=1/96, y1=1/64 on a 96x64 image
    assert rows[0]["box"]["x1"] == pytest.approx(1 / 96, abs=1e-5)
    assert rows[0]["box"]["y1"] == pytest.approx(1 / 64, abs=1e-5)


def test_summary_includes_track_id_when_present():
    r = _results(2)
    r.boxes.id = torch.tensor([7, 9])
    rows = r.summary()
    assert [row["track_id"] for row in rows] == [7, 9]


def test_summary_includes_segments_for_masks():
    pytest.importorskip("cv2")
    rows = _results(2, masks=True).summary(normalize=True)
    for row in rows:
        seg = row["segments"]
        assert len(seg["x"]) == len(seg["y"]) >= 3
        assert all(0.0 <= v <= 1.0 for v in seg["x"] + seg["y"])


def test_tojson_is_valid_json():
    import json

    parsed = json.loads(_results(2).tojson())
    assert isinstance(parsed, list) and len(parsed) == 2
    assert parsed[0]["name"] == "person" and parsed[0]["class"] == 0


def test_verbose_counts_per_class():
    # _results(2): one class-0 (person) + one class-2 (car), one each.
    assert _results(2).verbose() == "1 person, 1 car"


def test_verbose_pluralizes_and_orders_by_class_id():
    img = Image.fromarray(np.zeros((64, 96, 3), "uint8"))
    boxes = Boxes(
        xyxy=torch.tensor([[1.0, 1.0, 5.0, 5.0]] * 3),
        conf=torch.tensor([0.9, 0.8, 0.7]),
        cls=torch.tensor([2, 0, 0]),  # two class-0, one class-2
    )
    r = Results(img, boxes, names={0: "person", 2: "car"})
    assert r.verbose() == "2 persons, 1 car"  # class 0 before class 2, plural 's'


def test_verbose_empty():
    assert _results(0).verbose() == "(no detections)"


def test_verbose_sem_seg_lists_classes():
    from dfine.results import SemSeg

    img = Image.fromarray(np.zeros((8, 8, 3), "uint8"))
    data = torch.full((8, 8), 255, dtype=torch.uint8)  # start all-void
    data[:4] = 0
    data[4:] = 1
    r = Results(
        img,
        Boxes(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)),
        names={0: "road", 1: "sky"},
        sem_seg=SemSeg(data),
    )
    assert r.verbose() == "2 classes: road, sky"  # 255 void excluded


def test_save_txt_detection_yolo_format(tmp_path):
    out = _results(2).save_txt(tmp_path / "labels" / "img.txt")
    assert out is not None and out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    # `class cx cy w h`, all normalized into [0, 1]; class ids preserved (0, 2).
    for line, cls in zip(lines, (0, 2)):
        parts = line.split()
        assert len(parts) == 5 and int(parts[0]) == cls
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
    # xyxy [1,1,20,20] on a 96x64 image → cx=10.5/96, cy=10.5/64, w=19/96, h=19/64.
    # Coords are written to 6 decimals, so compare with an absolute tolerance.
    cx, cy, bw, bh = (float(v) for v in lines[0].split()[1:])
    assert cx == pytest.approx(10.5 / 96, abs=1e-6) and cy == pytest.approx(10.5 / 64, abs=1e-6)
    assert bw == pytest.approx(19 / 96, abs=1e-6) and bh == pytest.approx(19 / 64, abs=1e-6)


def test_save_txt_save_conf_appends_confidence(tmp_path):
    out = _results(2).save_txt(tmp_path / "img.txt", save_conf=True)
    lines = out.read_text().strip().splitlines()
    assert len(lines[0].split()) == 6  # class cx cy w h conf
    assert float(lines[0].split()[-1]) == pytest.approx(0.9)


def test_save_txt_empty_writes_nothing(tmp_path):
    out = tmp_path / "img.txt"
    assert _results(0).save_txt(out) is None
    assert not out.exists()


def test_save_txt_appends(tmp_path):
    out = tmp_path / "img.txt"
    _results(2).save_txt(out)
    _results(1).save_txt(out)
    assert len(out.read_text().strip().splitlines()) == 3  # 2 + 1 appended


def test_save_txt_segmentation_polygons(tmp_path):
    pytest.importorskip("cv2")
    out = _results(2, masks=True).save_txt(tmp_path / "seg.txt")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line, cls in zip(lines, (0, 2)):
        parts = line.split()
        assert int(parts[0]) == cls
        coords = parts[1:]
        assert len(coords) >= 6 and len(coords) % 2 == 0  # ≥3 polygon points, x/y pairs
        assert all(0.0 <= float(v) <= 1.0 for v in coords)


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
