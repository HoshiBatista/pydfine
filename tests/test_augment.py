"""Training augmentation tests (Phase 4).

Checks the ported D-FINE augment pipeline: output contract after the full transform,
the two-phase ``stop_epoch`` policy (advanced augs switch off in the no-aug tail),
epoch forwarding through the dataloader, and one real train step with augmentation on.
Needs faster-coco-eval (train extra), so the module skips without it.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_coco_eval")
from PIL import Image  # noqa: E402

from dfine.train.augment import ADVANCED_OPS, TrainCompose, train_transforms  # noqa: E402
from dfine.train.dataset import CocoDetection, build_coco_dataloader  # noqa: E402

IMGSZ = 320


def _make_coco(tmp_path, sizes=((200, 150), (160, 120))):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    images, annotations = [], []
    ann_id = 1
    for i, (w, h) in enumerate(sizes, start=1):
        fname = f"img{i}.jpg"
        Image.new("RGB", (w, h), color=(i * 30, 60, 90)).save(img_dir / fname)
        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        for cat in (1, 3):
            bw, bh = w // 3, h // 3
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat,
                    "bbox": [w // 6, h // 6, bw, bh],
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


def test_train_transforms_output_contract(tmp_path):
    torch.manual_seed(0)
    img_dir, ann_file = _make_coco(tmp_path)
    ds = CocoDetection(img_dir, ann_file, transforms=None, remap_mscoco_category=True)
    img, target = ds[0]  # raw PIL + xyxy tv_tensor boxes

    tf = train_transforms(IMGSZ, iou_crop_p=0.0)  # keep boxes for a stable check
    out_img, out_t = tf(img, target)
    assert out_img.shape == (3, IMGSZ, IMGSZ)
    assert out_img.dtype == torch.float32
    assert 0.0 <= float(out_img.min()) and float(out_img.max()) <= 1.0
    # boxes: cxcywh normalized plain tensor
    assert out_t["boxes"].shape[1] == 4
    assert out_t["boxes"].numel() == 0 or float(out_t["boxes"].max()) <= 1.0
    assert out_t["labels"].dtype == torch.int64


def test_stop_epoch_policy_skips_only_advanced():
    class _Op:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample):
            self.calls += 1
            return sample

    class Adv(_Op):
        pass

    class Keep(_Op):
        pass

    adv, keep = Adv(), Keep()
    comp = TrainCompose([adv, keep], stop_ops={"Adv"}, stop_epoch=5)

    comp.set_epoch(0)
    comp(("img", {}))
    assert (adv.calls, keep.calls) == (1, 1)  # before stop: both run

    comp.set_epoch(5)  # at stop_epoch onwards
    comp(("img", {}))
    assert (adv.calls, keep.calls) == (1, 2)  # advanced skipped, tail keeps running


def test_advanced_ops_are_the_upstream_three():
    assert set(ADVANCED_OPS) == {"RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"}


def test_dataset_forwards_epoch_to_transform(tmp_path):
    img_dir, ann_file = _make_coco(tmp_path)
    tf = train_transforms(IMGSZ, stop_epoch=5)
    ds = CocoDetection(img_dir, ann_file, transforms=tf, remap_mscoco_category=True)
    ds.set_epoch(7)
    assert tf.epoch == 7  # dataset.set_epoch propagated to the compose


def test_dataloader_forwards_epoch_and_yields_batch(tmp_path):
    img_dir, ann_file = _make_coco(tmp_path)
    tf = train_transforms(IMGSZ, iou_crop_p=0.0, stop_epoch=3)
    loader = build_coco_dataloader(
        img_dir,
        ann_file,
        imgsz=IMGSZ,
        batch_size=2,
        train=True,
        num_workers=0,
        remap_mscoco_category=True,
        transforms=tf,
        multiscale=False,
    )
    loader.set_epoch(1)
    assert tf.epoch == 1  # loader -> dataset -> transform
    images, targets = next(iter(loader))
    assert images.shape[0] == 2 and images.shape[-1] == IMGSZ
    assert all(t["boxes"].shape[1] == 4 for t in targets)


def test_augmented_training_step(tmp_path):
    pytest.importorskip("scipy")
    torch.manual_seed(0)
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
    # iou_crop_p=0 keeps every box so targets are never empty (stable for the matcher).
    tf = train_transforms(IMGSZ, iou_crop_p=0.0)
    loader = build_coco_dataloader(
        img_dir,
        ann_file,
        cfg=cfg,
        batch_size=2,
        train=True,
        num_workers=0,
        remap_mscoco_category=True,
        transforms=tf,
        multiscale=False,
    )
    model = NativeDFINE.from_config(cfg)
    criterion = DFINECriterion.from_config(cfg)
    opt = build_optimizer(model, cfg)
    stats = train_one_epoch(model, criterion, loader, opt, torch.device("cpu"), 0, print_freq=100)
    assert stats["loss"] == stats["loss"]  # finite
