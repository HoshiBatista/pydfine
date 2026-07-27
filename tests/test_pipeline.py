"""End-to-end pipeline templates (CI smoke of the whole user journey).

Unlike the focused unit tests, these run the *entire* public workflow on a tiny
synthetic dataset so a regression anywhere along the chain fails CI:

    detection : YOLO dataset -> convert -> DFINE.train -> DFINE.val -> predict -> export
    segment   : YOLO-Seg dataset -> DFINE.train (mask AP val) -> predict (masks)

They double as copy-paste **code templates**: each test body is the minimal, real
sequence of calls a user makes to fine-tune and ship a model. Everything is sized to
run on CPU in seconds (preset "n", imgsz 320, 1 epoch, no pretrained backbone), so they
belong in the standard `pytest` run across the whole Python matrix.

Needs the train extra (faster-coco-eval / torchvision v2 / scipy / opencv); the module
skips cleanly without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_coco_eval")  # COCO train/val
pytest.importorskip("scipy")  # Hungarian matcher
from PIL import Image  # noqa: E402

IMGSZ = 320  # >= the nano decoder's 300-query top-k needs


def _write_yolo_det(root, split, sizes):
    """Write a tiny YOLO detection split: images/<split>/*.jpg + labels/<split>/*.txt."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, (w, h) in enumerate(sizes, start=1):
        Image.new("RGB", (w, h), color=(30 * i, 60, 90)).save(img_dir / f"img{i}.jpg")
        # two normalized cxcywh boxes (classes 0 and 1), safely inside the frame
        (lbl_dir / f"img{i}.txt").write_text("0 0.35 0.35 0.20 0.20\n1 0.65 0.65 0.20 0.20\n")


def test_detection_pipeline_end_to_end(tmp_path):
    """YOLO dataset -> convert -> build -> train(1 epoch) -> val -> predict -> export."""
    from dfine import DFINE
    from dfine.convert import yolo_to_coco

    # 1) A tiny YOLO dataset on disk, then convert it to the COCO layout DFINE trains on.
    yolo = tmp_path / "yolo"
    _write_yolo_det(yolo, "train", ((200, 150), (160, 128)))
    _write_yolo_det(yolo, "val", ((176, 144),))
    coco = tmp_path / "coco"
    written = yolo_to_coco(yolo, coco, class_names=["person", "car"])
    assert set(written) == {"train", "val"}

    # 2) Build a small model (2 classes; ImageNet backbone off so CI needs no download).
    model = DFINE(
        size="n",
        imgsz=IMGSZ,
        num_classes=2,
        class_names=["person", "car"],
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=0,
        device="cpu",
    )

    # 3) Fine-tune one epoch; val (COCO AP) runs automatically off the converted val split.
    out = model.train(
        data=str(coco),
        epochs=1,
        batch_size=2,
        num_workers=0,
        augment=False,
        output_dir=str(tmp_path / "runs"),
        visualize=False,
    )
    assert out is model
    assert (tmp_path / "runs" / "last.pth").exists()

    # 4) Explicit evaluation returns the 12 COCO metrics (finite, AP present).
    metrics = model.val(data=str(coco), batch_size=2, num_workers=0)
    assert "AP" in metrics and metrics["AP"] == metrics["AP"]  # finite

    # 5) Inference on an image -> Results with original-scale boxes.
    img_path = yolo / "images" / "val" / "img1.jpg"
    results = model.predict(str(img_path), conf=0.0)
    assert len(results) == 1
    res = results[0]
    assert res.orig_shape == (144, 176)  # (H, W) of the source image

    # 6) Export a deployable graph (TorchScript needs only torch — no ONNX toolchain).
    ts = model.export("torchscript", file=str(tmp_path / "model.torchscript"))
    assert ts.exists() and ts.stat().st_size > 0


def _write_yolo_seg(root, split, sizes):
    """Write a tiny YOLO-Seg split: images/<split>/*.jpg + labels/<split>/*.txt polygons."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, (w, h) in enumerate(sizes, start=1):
        Image.new("RGB", (w, h), color=(30 * i, 60, 90)).save(img_dir / f"img{i}.jpg")
        # one square polygon (class 0), normalized `cls x1 y1 x2 y2 x3 y3 x4 y4`
        (lbl_dir / f"img{i}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")


def test_segment_pipeline_end_to_end(tmp_path):
    """YOLO-Seg dataset -> build -> train(1 epoch, mask-AP val) -> predict (instance masks)."""
    pytest.importorskip("cv2")  # polygon rasterization
    pytest.importorskip("torchmetrics")  # mask AP val
    from dfine import DFINE

    seg = tmp_path / "seg"
    _write_yolo_seg(seg, "train", ((200, 150), (160, 128)))
    _write_yolo_seg(seg, "val", ((176, 144),))

    model = DFINE(
        size="n",
        imgsz=IMGSZ,
        num_classes=1,
        class_names=["thing"],
        task="segment",
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=0,
        device="cpu",
    )

    out = model.train(
        data=str(seg),
        epochs=1,
        batch_size=2,
        num_workers=0,
        output_dir=str(tmp_path / "runs"),
        visualize=False,
    )
    assert out is model
    assert (tmp_path / "runs" / "last.pth").exists()

    # Predict carries per-instance masks at the original image resolution.
    img_path = seg / "images" / "val" / "img1.jpg"
    res = model.predict(str(img_path), conf=0.0)[0]
    assert res.masks is not None
    if len(res.masks):
        assert res.masks.data.shape[-2:] == (144, 176)  # (H, W) of the source image
