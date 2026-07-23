"""YOLO -> COCO dataset converter tests.

Builds a tiny YOLO dataset on disk (images + label txts + data.yaml) and checks
`dfine.convert.yolo_to_coco` writes the COCO layout D-FINE trains on: correct
box conversion (normalized cxcywh -> absolute xywh), 0-indexed category ids, split
folders, and background images. A round-trip through `build_coco_dataloader` /
`evaluate` proves the output is consumable (including category_id 0).
"""

from __future__ import annotations

import json

import pytest

Image = pytest.importorskip("PIL.Image")

from dfine.convert import yolo_to_coco  # noqa: E402


def _make_yolo(root, *, with_yaml=True, val_label=True):
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / sub).mkdir(parents=True)
    Image.new("RGB", (100, 100), (10, 20, 30)).save(root / "images/train/img1.jpg")
    Image.new("RGB", (100, 100), (40, 50, 60)).save(root / "images/train/img2.jpg")
    Image.new("RGB", (100, 100), (70, 80, 90)).save(root / "images/val/img3.jpg")
    # img1: two boxes; img2: empty label -> background (no objects, still an image).
    (root / "labels/train/img1.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.5 0.5\n")
    (root / "labels/train/img2.txt").write_text("")
    if val_label:
        (root / "labels/val/img3.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    if with_yaml:
        (root / "data.yaml").write_text(
            "names: [person, car]\ntrain: images/train\nval: images/val\n"
        )
    return root


def _load(path):
    return json.loads(open(path).read())


def test_layout_and_return(tmp_path):
    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    written = yolo_to_coco(tmp_path / "yolo", out)

    assert set(written) == {"train", "val"}
    assert (out / "train/img1.jpg").exists()
    assert (out / "val/img3.jpg").exists()
    assert (out / "annotations/instances_train.json").exists()
    assert (out / "annotations/instances_val.json").exists()


def test_box_conversion_and_categories(tmp_path):
    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out)
    coco = _load(out / "annotations/instances_train.json")

    assert coco["categories"] == [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "car"},
    ]
    assert len(coco["images"]) == 2  # img1 + background img2

    by_img = {i["id"]: i["file_name"] for i in coco["images"]}
    img1_id = next(k for k, v in by_img.items() if v == "img1.jpg")
    anns = [a for a in coco["annotations"] if a["image_id"] == img1_id]
    assert len(anns) == 2

    # `0 0.5 0.5 0.2 0.4` on a 100x100 image -> xywh [40, 30, 20, 40], cat 0.
    first = min(anns, key=lambda a: a["id"])
    assert first["category_id"] == 0
    assert first["bbox"] == [40.0, 30.0, 20.0, 40.0]
    assert first["area"] == pytest.approx(800.0)
    assert first["iscrowd"] == 0


def test_background_image_has_no_annotations(tmp_path):
    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out)
    coco = _load(out / "annotations/instances_train.json")

    img2_id = next(i["id"] for i in coco["images"] if i["file_name"] == "img2.jpg")
    assert [a for a in coco["annotations"] if a["image_id"] == img2_id] == []


def test_explicit_names_override_yaml(tmp_path):
    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out, class_names=["a", "b", "c"])
    coco = _load(out / "annotations/instances_train.json")
    assert [c["name"] for c in coco["categories"]] == ["a", "b", "c"]


def test_infer_names_without_yaml(tmp_path):
    _make_yolo(tmp_path / "yolo", with_yaml=False)
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out)
    coco = _load(out / "annotations/instances_train.json")
    # highest class id seen is 1 -> class_0, class_1.
    assert [c["name"] for c in coco["categories"]] == ["class_0", "class_1"]


def test_too_few_class_names_raises(tmp_path):
    # Labels reference class id 1, but only one name is given -> guard fires.
    _make_yolo(tmp_path / "yolo", with_yaml=False)
    with pytest.raises(ValueError, match="class id 1 but only 1 class"):
        yolo_to_coco(tmp_path / "yolo", tmp_path / "coco", class_names=["only_one"])


def test_polygon_row_becomes_bbox(tmp_path):
    root = tmp_path / "yolo"
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    Image.new("RGB", (100, 100), (0, 0, 0)).save(root / "images/train/p.jpg")
    # class 0 polygon spanning x in [0.2,0.6], y in [0.1,0.5] -> bbox [20,10,40,40].
    (root / "labels/train/p.txt").write_text("0 0.2 0.1 0.6 0.1 0.6 0.5 0.2 0.5\n")
    out = tmp_path / "coco"
    yolo_to_coco(root, out, class_names=["x"])
    coco = _load(out / "annotations/instances_train.json")
    assert coco["annotations"][0]["bbox"] == pytest.approx([20.0, 10.0, 40.0, 40.0])


def test_symlink_mode(tmp_path):
    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out, copy_images=False)
    assert (out / "train/img1.jpg").is_symlink()


def test_missing_splits_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No YOLO splits"):
        yolo_to_coco(tmp_path / "empty", tmp_path / "coco")


def test_roundtrip_through_dataloader(tmp_path):
    pytest.importorskip("faster_coco_eval")
    from dfine.train.dataset import build_coco_dataloader

    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out)

    loader = build_coco_dataloader(
        str(out / "train"),
        str(out / "annotations/instances_train.json"),
        imgsz=64,
        batch_size=2,
        train=False,
        num_workers=0,
        remap_mscoco_category=False,  # ids are already contiguous 0..N-1
    )
    _, targets = next(iter(loader))
    labels = torch_cat_labels(targets)
    assert labels <= {0, 1}


def torch_cat_labels(targets):
    vals = set()
    for t in targets:
        vals |= set(t["labels"].tolist())
    return vals


def test_roundtrip_evaluate_accepts_category_zero(tmp_path):
    """category_id 0 must survive COCO eval (faster-coco-eval)."""
    pytest.importorskip("faster_coco_eval")
    import torch

    from dfine.train.dataset import build_coco_val_dataloader
    from dfine.train.evaluator import COCO_STAT_NAMES, evaluate
    from tests.test_evaluator import _IdentityPost, _perfect_predictions, _ReplayModel

    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    yolo_to_coco(tmp_path / "yolo", out)

    loader = build_coco_val_dataloader(str(out), imgsz=64, batch_size=1, num_workers=0)
    metrics = evaluate(
        _ReplayModel(_perfect_predictions(loader)),
        _IdentityPost(),
        loader,
        torch.device("cpu"),
    )
    assert set(metrics) == set(COCO_STAT_NAMES)
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-6)


def test_cli_convert(tmp_path):
    from dfine.cli import main

    _make_yolo(tmp_path / "yolo")
    out = tmp_path / "coco"
    rc = main(["convert", str(tmp_path / "yolo"), str(out)])
    assert rc == 0
    assert (out / "annotations/instances_train.json").exists()
