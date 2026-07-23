"""TS4: YOLO-style segmentation datasets (instance polygons + sem_seg PNG masks).

Builds tiny on-disk datasets in tmp_path and checks the parsing, polygon rasterization,
and the target dicts the seg criteria consume. Needs torch + PIL (train extra); the
polygon path needs opencv.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from dfine.config import DFINEConfig  # noqa: E402
from dfine.train.seg_dataset import (  # noqa: E402
    SemSegDataset,
    YoloInstanceSegDataset,
    _label_path,
    _resolve_split,
    build_seg_dataloader,
    build_seg_dataloaders,
    parse_yolo_seg_label,
    polygons_to_masks,
)

IMGSZ = 32


def _write_image(path, size=(50, 40)):
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _seg_root(tmp_path, label_lines):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    _write_image(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text("\n".join(label_lines))
    return root


# --- parsing ------------------------------------------------------------------


def test_parse_detection_and_polygon_lines(tmp_path):
    p = tmp_path / "l.txt"
    # a detection row (5 cols) and a unit-square polygon row for class 2.
    p.write_text("0 0.5 0.5 0.2 0.2\n2 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n")
    boxes, polys = parse_yolo_seg_label(p)
    assert boxes.shape == (2, 5)
    assert polys[0].shape == (0, 2)  # detection-only row -> empty polygon
    assert polys[1].shape == (4, 2)
    # polygon bbox -> full-image cxcywh
    np.testing.assert_allclose(boxes[1, 1:], [0.5, 0.5, 1.0, 1.0], atol=1e-6)


def test_parse_missing_file_is_empty(tmp_path):
    boxes, polys = parse_yolo_seg_label(tmp_path / "nope.txt")
    assert boxes.shape == (0, 5) and polys == []


# --- rasterization ------------------------------------------------------------


def test_polygons_to_masks_area_and_alignment():
    pytest.importorskip("cv2")
    full = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)  # whole image
    empty = np.empty((0, 2), dtype=np.float32)  # detection-only row
    masks = polygons_to_masks([full, empty], 16, 16)
    assert masks.shape == (2, 16, 16) and masks.dtype == torch.uint8
    assert masks[0].float().mean() > 0.95  # full polygon fills the frame
    assert masks[1].sum() == 0  # empty polygon -> zero mask, still row-aligned


# --- instance dataset ---------------------------------------------------------


def test_instance_dataset_yields_aligned_masks(tmp_path):
    pytest.importorskip("cv2")
    root = _seg_root(tmp_path, ["0 0.25 0.25 0.5 0.0 0.5 0.5 0.25 0.5", "1 0.5 0.5 0.2 0.2"])
    ds = YoloInstanceSegDataset(root, imgsz=IMGSZ)
    image, t = ds[0]
    assert image.shape == (3, IMGSZ, IMGSZ)
    assert t["boxes"].shape == (2, 4) and t["labels"].tolist() == [0, 1]
    assert t["masks"].shape == (2, IMGSZ, IMGSZ) and t["masks"].dtype == torch.uint8
    assert t["orig_size"].tolist() == [50, 40]  # (W, H) at original resolution


def test_build_segment_dataloader_batches(tmp_path):
    pytest.importorskip("cv2")
    root = _seg_root(tmp_path, ["0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0"])
    cfg = DFINEConfig.preset("n", task="segment", imgsz=IMGSZ)
    loader = build_seg_dataloader(root, cfg=cfg, batch_size=1, num_workers=0, train=False)
    images, targets = next(iter(loader))
    assert images.shape == (1, 3, IMGSZ, IMGSZ)
    assert set(targets[0]) >= {"boxes", "labels", "masks"}


# --- sem_seg dataset ----------------------------------------------------------


def _sem_root(tmp_path, mask_arr):
    root = tmp_path / "ss"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    _write_image(root / "images" / "a.jpg")
    Image.fromarray(mask_arr.astype(np.uint8), mode="L").save(root / "labels" / "a.png")
    return root


def test_sem_seg_dataset_label_map(tmp_path):
    arr = np.zeros((40, 50), dtype=np.uint8)
    arr[:20] = 3  # top half class 3
    arr[20:, :10] = 255  # a strip of ignore_index
    ds = SemSegDataset(_sem_root(tmp_path, arr), num_classes=19, imgsz=IMGSZ)
    _, t = ds[0]
    assert t["sem_mask"].shape == (IMGSZ, IMGSZ) and t["sem_mask"].dtype == torch.int64
    vals = set(t["sem_mask"].unique().tolist())
    assert vals <= {0, 3, 255}  # only the classes we drew + ignore_index survive NEAREST resize
    assert 255 in vals  # ignore_index preserved


def test_sem_seg_rejects_out_of_range_class(tmp_path):
    arr = np.full((40, 50), 30, dtype=np.uint8)  # class 30 >= num_classes=19, not ignore_index
    ds = SemSegDataset(_sem_root(tmp_path, arr), num_classes=19, imgsz=IMGSZ)
    with pytest.raises(ValueError, match="num_classes"):
        _ = ds[0]


def test_build_sem_seg_dataloader_from_cfg(tmp_path):
    arr = np.zeros((40, 50), dtype=np.uint8)
    root = _sem_root(tmp_path, arr)
    cfg = DFINEConfig.preset("n", task="sem_seg", num_classes=19, imgsz=IMGSZ)
    loader = build_seg_dataloader(root, cfg=cfg, batch_size=1, num_workers=0, train=False)
    images, targets = next(iter(loader))
    assert images.shape == (1, 3, IMGSZ, IMGSZ)
    assert targets[0]["sem_mask"].shape == (IMGSZ, IMGSZ)


# --- train/val split ----------------------------------------------------------


def _flat_root(tmp_path, n=10):
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n):
        _write_image(root / "images" / f"img{i:02d}.jpg")
        (root / "labels" / f"img{i:02d}.txt").write_text("0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0")
    return root


def test_label_path_maps_images_to_labels_both_layouts(tmp_path):
    flat = tmp_path / "images" / "a.jpg"
    assert _label_path(flat, ".txt") == tmp_path / "labels" / "a.txt"
    sub = tmp_path / "images" / "val" / "b.png"
    assert _label_path(sub, ".png") == tmp_path / "labels" / "val" / "b.png"


def test_resolve_split_ratio_is_deterministic_and_disjoint(tmp_path):
    root = _flat_root(tmp_path, n=10)
    train, val = _resolve_split(root, val_split=0.3, seed=0)
    assert len(val) == 3 and len(train) == 7  # 30% -> 3 val
    assert set(train).isdisjoint(val)  # no leakage
    assert (train, val) == _resolve_split(root, val_split=0.3, seed=0)  # deterministic
    # val_split=0 -> everything trains, no val split.
    train_all, val_none = _resolve_split(root, val_split=0.0, seed=0)
    assert len(train_all) == 10 and val_none == []


def test_resolve_split_uses_train_val_subdirs(tmp_path):
    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    _write_image(root / "images" / "train" / "a.jpg")
    _write_image(root / "images" / "train" / "b.jpg")
    _write_image(root / "images" / "val" / "c.jpg")
    train, val = _resolve_split(root, val_split=0.5, seed=0)  # val_split ignored when subdirs exist
    assert [p.stem for p in train] == ["a", "b"] and [p.stem for p in val] == ["c"]


def test_build_seg_dataloaders_returns_train_and_val(tmp_path):
    pytest.importorskip("cv2")
    root = _flat_root(tmp_path, n=10)
    cfg = DFINEConfig.preset("n", task="segment", imgsz=IMGSZ)
    train_loader, val_loader = build_seg_dataloaders(
        root, cfg=cfg, batch_size=2, num_workers=0, val_split=0.2
    )
    assert len(train_loader.dataset) == 8 and len(val_loader.dataset) == 2
    assert val_loader.drop_last is False  # val runs unshuffled, keeps the tail batch
    # val_split=0 -> no val loader.
    _, none_val = build_seg_dataloaders(root, cfg=cfg, num_workers=0, val_split=0.0)
    assert none_val is None
