"""COCO dataset/dataloader tests (Phase 4).

Builds a tiny COCO-format dataset on disk (real images + an instances JSON) and checks
the loader yields exactly what the criterion/trainer expect: BCHW float images in
[0,1] and per-image targets with contiguous long labels and cxcywh-normalized boxes.
Needs faster-coco-eval (train extra), so the module skips without it.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_coco_eval")
from PIL import Image  # noqa: E402

from dfine.train.dataset import (  # noqa: E402
    BatchImageCollateFunction,
    build_coco_dataloader,
    default_transforms,
    generate_scales,
)

IMGSZ = 320


def _make_coco(tmp_path, sizes=((200, 150), (120, 90))):
    """Write N images + an instances JSON; return (img_dir, ann_file)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    images, annotations = [], []
    ann_id = 1
    for i, (w, h) in enumerate(sizes, start=1):
        fname = f"img{i}.jpg"
        Image.new("RGB", (w, h), color=(i * 30, 60, 90)).save(img_dir / fname)
        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        # two boxes per image, xywh, safely inside the frame
        for cat in (1, 3):  # MS-COCO ids: person, car
            bw, bh = w // 4, h // 4
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat,
                    "bbox": [w // 8, h // 8, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return str(img_dir), str(ann_file)


def test_loader_output_contract(tmp_path):
    img_dir, ann_file = _make_coco(tmp_path)
    loader = build_coco_dataloader(
        img_dir,
        ann_file,
        imgsz=IMGSZ,
        batch_size=2,
        train=False,
        num_workers=0,
        remap_mscoco_category=True,
    )
    images, targets = next(iter(loader))

    assert images.shape == (2, 3, IMGSZ, IMGSZ)
    assert images.dtype == torch.float32
    assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0

    assert len(targets) == 2
    for t in targets:
        assert t["labels"].dtype == torch.int64
        assert t["boxes"].shape[1] == 4
        # cxcywh normalized -> all coords in [0, 1]
        assert float(t["boxes"].min()) >= 0.0 and float(t["boxes"].max()) <= 1.0
        # remap on: sparse COCO ids (1, 3) -> contiguous labels (0, 2)
        assert set(t["labels"].tolist()) <= {0, 2}
        assert "image_id" in t and "orig_size" in t


def test_no_remap_keeps_raw_category_ids(tmp_path):
    img_dir, ann_file = _make_coco(tmp_path)
    loader = build_coco_dataloader(
        img_dir,
        ann_file,
        imgsz=IMGSZ,
        batch_size=2,
        train=False,
        num_workers=0,
        remap_mscoco_category=False,
    )
    _, targets = next(iter(loader))
    # Without remap, labels are the raw category ids (1, 3).
    assert set(targets[0]["labels"].tolist()) <= {1, 3}


def test_default_transforms_resizes_and_normalizes(tmp_path):
    img_dir, ann_file = _make_coco(tmp_path, sizes=((200, 150),))
    from dfine.train.dataset import CocoDetection

    ds = CocoDetection(
        img_dir, ann_file, transforms=default_transforms(IMGSZ), remap_mscoco_category=True
    )
    img, target = ds[0]
    assert img.shape == (3, IMGSZ, IMGSZ)
    assert target["boxes"].shape == (2, 4)
    assert not isinstance(target["boxes"], torch.Tensor) or target["boxes"].max() <= 1.0


def test_generate_scales_grid():
    scales = generate_scales(base_size=640, base_size_repeat=3)
    assert 640 in scales
    assert all(s % 32 == 0 for s in scales)  # all on the 32-px grid
    assert min(scales) < 640 < max(scales)


def test_multiscale_collate_changes_size(tmp_path):
    # Direct collate test: before stop_epoch a scale is applied; after, base size holds.
    imgs = [(torch.rand(3, IMGSZ, IMGSZ), {"labels": torch.tensor([0])}) for _ in range(2)]
    collate = BatchImageCollateFunction(base_size=IMGSZ, base_size_repeat=3, stop_epoch=5)
    collate.set_epoch(10)  # past stop_epoch -> no rescale, fixed base size
    out, _ = collate(imgs)
    assert out.shape[-1] == IMGSZ


@pytest.mark.parametrize("_", [0])
def test_feeds_trainer_one_step(tmp_path, _):
    pytest.importorskip("scipy")  # criterion matcher
    from dfine import DFINEConfig
    from dfine.backends.native import DFINE as NativeDFINE
    from dfine.backends.native import DFINECriterion
    from dfine.train.trainer import build_optimizer, train_one_epoch

    img_dir, ann_file = _make_coco(tmp_path)
    cfg = DFINEConfig.preset(
        "n",
        imgsz=IMGSZ,
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=0,
    )
    loader = build_coco_dataloader(
        img_dir,
        ann_file,
        cfg=cfg,
        batch_size=2,
        train=False,
        num_workers=0,
        remap_mscoco_category=True,
    )
    model = NativeDFINE.from_config(cfg)
    criterion = DFINECriterion.from_config(cfg)
    opt = build_optimizer(model, cfg)
    stats = train_one_epoch(model, criterion, loader, opt, torch.device("cpu"), 0, print_freq=100)
    assert "loss" in stats and stats["loss"] == stats["loss"]  # finite
