"""COCO-format dataset + dataloader for training (Phase 4).

Ports D-FINE's ``src/data`` detection dataset/collate (``coco_dataset.py`` +
``dataloader.py``) with the registry/YAML layer removed and a config-first
``build_coco_dataloader`` entry point added. It yields exactly what the trainer +
criterion expect: an image batch ``BCHW`` float in ``[0,1]`` and per-image targets
with ``labels`` (``LongTensor``) and ``boxes`` (``cxcywh``, normalized to ``[0,1]``),
plus ``image_id``/``orig_size`` for later COCO eval.

The *default* transform here is deliberately minimal — resize-to-square + tensor +
box conversion, matching upstream's **val** pipeline and our ``predict`` preprocess.
The advanced training augmentations (photometric distort, zoom-out, IoU-crop, H-flip,
multi-scale) are Phase-4's ``augment.py`` task; pass your own ``transforms=`` to plug
them in. Multi-scale jitter is available now via the collate function.

Needs ``pip install dfine[train]`` (``faster-coco-eval`` provides the COCO parser).
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image as PILImage

from ..backends.native.coco import mscoco_category2label, mscoco_category2name

torchvision.disable_beta_transforms_warning()

try:
    from faster_coco_eval.utils.pytorch import FasterCocoDetection
except ImportError as exc:  # pragma: no cover - exercised only without the train extra
    raise ImportError(
        "COCO training data needs faster-coco-eval — install with `pip install dfine[train]`."
    ) from exc

# torchvision >= 0.16 exposes tv_tensors; 0.15.x still calls them datapoints.
try:
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
    from torchvision.tv_tensors import Image as TVImage

    _CANVAS_KEY = "canvas_size"
except ImportError:  # pragma: no cover - old torchvision
    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat
    from torchvision.datapoints import Image as TVImage

    _CANVAS_KEY = "spatial_size"

__all__ = [
    "CocoDetection",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
    "default_transforms",
    "build_coco_dataloader",
    "build_coco_dataloaders",
    "build_coco_val_dataloader",
]


def _boxes_tv_tensor(boxes: torch.Tensor, canvas_hw: tuple[int, int]) -> BoundingBoxes:
    """Wrap an ``xyxy`` box tensor as a torchvision ``BoundingBoxes`` (canvas = H,W)."""
    return BoundingBoxes(boxes, format=BoundingBoxFormat.XYXY, **{_CANVAS_KEY: canvas_hw})


class _PrepareCocoTarget:
    """Turn raw COCO annotations into a clean detection target (boxes/labels/meta).

    Mirrors upstream ``ConvertCocoPolysToMask`` for the detection path: drop crowds,
    convert ``xywh`` → ``xyxy``, clamp to the image, keep only non-degenerate boxes,
    and remap category ids to contiguous labels when asked. Masks/keypoints are not
    handled (detection-only port).
    """

    def __call__(self, image: PILImage.Image, target: dict, category2label=None) -> dict:
        w, h = image.size
        anno = [obj for obj in target["annotations"] if obj.get("iscrowd", 0) == 0]

        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
        labels = torch.tensor(labels, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, labels = boxes[keep], labels[keep]
        area = torch.as_tensor([obj["area"] for obj in anno], dtype=torch.float32)[keep]
        iscrowd = torch.as_tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)[keep]

        return {
            "boxes": _boxes_tv_tensor(boxes, (h, w)),
            "labels": labels,
            "image_id": torch.tensor([target["image_id"]]),
            "image_path": target["image_path"],
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(w), int(h)]),
        }


class CocoDetection(FasterCocoDetection):
    """COCO-format detection dataset (port of upstream ``CocoDetection``).

    Args:
        img_folder: directory with the images.
        ann_file: COCO-format instances JSON.
        transforms: callable ``(image, target) -> (image, target)`` applied per item
            (defaults to :func:`default_transforms`). torchvision v2 transforms work
            directly on the ``(PIL image, target-dict)`` pair.
        remap_mscoco_category: remap the 80 sparse MS-COCO ids to contiguous ``0..79``
            labels (use for stock COCO; leave ``False`` for already-contiguous data).
    """

    def __init__(self, img_folder, ann_file, transforms=None, remap_mscoco_category=False):
        super().__init__(img_folder, ann_file)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.remap_mscoco_category = remap_mscoco_category
        self._prepare = _PrepareCocoTarget()

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def load_item(self, idx):
        image, anno = super().__getitem__(idx)
        image_id = self.ids[idx]
        image_path = os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"])
        raw = {"image_id": image_id, "image_path": image_path, "annotations": anno}
        cat2label = mscoco_category2label if self.remap_mscoco_category else None
        target = self._prepare(image, raw, category2label=cat2label)
        target["idx"] = torch.tensor([idx])
        return image, target

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        # Forward to epoch-aware transforms (e.g. augment.TrainCompose's no-aug policy).
        if hasattr(self._transforms, "set_epoch"):
            self._transforms.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)


# --- transforms ---------------------------------------------------------------


class _ConvertPILImage(T.Transform):
    """PIL image -> float ``tv_tensors.Image`` in ``[0,1]`` (port of ConvertPILImage)."""

    _transformed_types = (PILImage.Image,)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        from torchvision.transforms.v2 import functional as VF

        return TVImage(VF.pil_to_tensor(inpt).float() / 255.0)


class _ConvertBoxes(T.Transform):
    """Boxes -> ``cxcywh`` normalized to ``[0,1]`` plain tensor (port of ConvertBoxes)."""

    _transformed_types = (BoundingBoxes,)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        canvas = getattr(inpt, _CANVAS_KEY)  # (H, W)
        boxes = torchvision.ops.box_convert(inpt, in_fmt="xyxy", out_fmt="cxcywh")
        norm = torch.tensor([canvas[1], canvas[0]]).tile(2)[None]  # [W,H,W,H]
        return boxes / norm


def default_transforms(imgsz: int = 640, train: bool = True) -> T.Compose:
    """Minimal resize-to-square + tensor + ``cxcywh``-normalize pipeline (no aug).

    Matches upstream's val transform and our ``predict`` preprocess. ``train`` is
    accepted for symmetry / future augmentation hooks but does not change the ops yet
    (advanced augs live in ``augment.py``).
    """
    return T.Compose(
        [
            T.Resize((imgsz, imgsz)),
            _ConvertPILImage(),
            _ConvertBoxes(),
        ]
    )


# --- collate / dataloader -----------------------------------------------------


def batch_image_collate_fn(items):
    """Stack same-size images to ``BCHW`` and gather targets into a list."""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


def generate_scales(base_size: int, base_size_repeat: int) -> list[int]:
    """Multi-scale set around ``base_size`` on a 32-px grid (port of upstream)."""
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class BatchImageCollateFunction:
    """Collate that optionally resizes each batch to a random scale (multi-scale).

    Until ``stop_epoch`` (and only when ``base_size_repeat`` is set) every batch is
    interpolated to a random size from :func:`generate_scales`, matching upstream's
    ``BatchImageCollateFunction``. Call :meth:`set_epoch` each epoch (the dataloader
    below forwards it) so the tail epochs run at the fixed ``base_size``.
    """

    def __init__(self, base_size=640, base_size_repeat=None, stop_epoch=None):
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 10**9
        self._epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
        return images, targets


class _CocoDataLoader(data.DataLoader):
    """``DataLoader`` that forwards ``set_epoch`` to the dataset + collate (multi-scale)."""

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)


def build_coco_dataloader(
    img_folder: str,
    ann_file: str,
    *,
    cfg=None,
    imgsz: int = 640,
    batch_size: int = 4,
    train: bool = True,
    shuffle: bool | None = None,
    num_workers: int = 4,
    remap_mscoco_category: bool = False,
    transforms: Callable | None = None,
    multiscale: bool = True,
    drop_last: bool | None = None,
) -> _CocoDataLoader:
    """Build a ready-to-train COCO dataloader.

    Pass ``cfg`` (a :class:`~dfine.config.DFINEConfig`) to inherit ``imgsz`` and the
    multi-scale ``no_aug_epoch`` cutoff; explicit kwargs override it. Yields
    ``(images, targets)`` batches consumable directly by ``DFINE.train`` /
    ``dfine.train.Trainer``.

    Example::

        from dfine import DFINE
        from dfine.train.dataset import build_coco_dataloader

        m = DFINE(size="n", num_classes=80)
        loader = build_coco_dataloader(
            "coco/train2017", "coco/annotations/instances_train2017.json",
            cfg=m.config, batch_size=4, remap_mscoco_category=True,
        )
        m.train(loader, epochs=10)
    """
    if cfg is not None:
        imgsz = cfg.imgsz
    if shuffle is None:
        shuffle = train
    if drop_last is None:
        drop_last = train

    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=transforms or default_transforms(imgsz, train=train),
        remap_mscoco_category=remap_mscoco_category,
    )

    # Multi-scale only for training, and only up to the no-aug tail (like upstream).
    repeat = 3 if (train and multiscale) else None
    stop_epoch = (cfg.epochs - cfg.no_aug_epoch) if cfg is not None else None
    collate = BatchImageCollateFunction(
        base_size=imgsz, base_size_repeat=repeat, stop_epoch=stop_epoch
    )
    return _CocoDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=drop_last,
    )


def build_coco_dataloaders(
    data_root: str,
    *,
    cfg=None,
    imgsz: int = 640,
    batch_size: int = 4,
    num_workers: int = 4,
    remap_mscoco_category: bool = False,
    augment: bool = True,
    train_images: str = "train2017",
    train_ann: str = "annotations/instances_train2017.json",
    val_images: str = "val2017",
    val_ann: str = "annotations/instances_val2017.json",
) -> tuple[_CocoDataLoader, _CocoDataLoader | None]:
    """Build ``(train_loader, val_loader)`` from a standard COCO dataset root.

    Expects the MS-COCO on-disk layout under ``data_root`` (override the split names
    for custom datasets)::

        data_root/
          train2017/                                  # train images
          val2017/                                    # val images (optional)
          annotations/
            instances_train2017.json
            instances_val2017.json                    # optional

    The train loader uses the full two-phase augmentation pipeline
    (:func:`~dfine.train.augment.train_transforms`, ``stop_epoch`` derived from
    ``cfg``) when ``augment`` is set, plus multi-scale collate. The val loader (built
    only if its images/annotations exist) uses the plain resize preprocess and returns
    ``None`` when absent. This is what powers ``DFINE.train(data=...)``.
    """
    if cfg is not None:
        imgsz = cfg.imgsz

    data_root = os.fspath(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root not found: {data_root!r}")

    train_img_dir = os.path.join(data_root, train_images)
    train_ann_path = os.path.join(data_root, train_ann)
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(f"Train image folder not found: {train_img_dir!r}")
    if not os.path.isfile(train_ann_path):
        raise FileNotFoundError(f"Train annotations not found: {train_ann_path!r}")

    transforms = None
    if augment:
        from .augment import train_transforms

        stop_epoch = (cfg.epochs - cfg.no_aug_epoch) if cfg is not None else None
        transforms = train_transforms(imgsz, stop_epoch=stop_epoch)

    train_loader = build_coco_dataloader(
        train_img_dir,
        train_ann_path,
        cfg=cfg,
        imgsz=imgsz,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        remap_mscoco_category=remap_mscoco_category,
        transforms=transforms,
    )

    val_loader = None
    if os.path.isdir(os.path.join(data_root, val_images)) and os.path.isfile(
        os.path.join(data_root, val_ann)
    ):
        val_loader = build_coco_val_dataloader(
            data_root,
            cfg=cfg,
            imgsz=imgsz,
            batch_size=batch_size,
            num_workers=num_workers,
            remap_mscoco_category=remap_mscoco_category,
            val_images=val_images,
            val_ann=val_ann,
        )
    return train_loader, val_loader


def build_coco_val_dataloader(
    data_root: str,
    *,
    cfg=None,
    imgsz: int = 640,
    batch_size: int = 4,
    num_workers: int = 4,
    remap_mscoco_category: bool = False,
    val_images: str = "val2017",
    val_ann: str = "annotations/instances_val2017.json",
) -> _CocoDataLoader:
    """Build a single COCO **val** loader (plain resize, no multi-scale) from a root.

    Resolves ``data_root/val2017`` + ``data_root/annotations/instances_val2017.json``
    (split names overridable) and raises :class:`FileNotFoundError` if either is
    missing. This is what powers ``DFINE.val(data=…)``; for eval only the ground-truth
    ``.coco`` and the image tensors are used, so ``remap_mscoco_category`` (which only
    affects target labels) does not change the reported metrics.
    """
    if cfg is not None:
        imgsz = cfg.imgsz

    data_root = os.fspath(data_root)
    img_dir = os.path.join(data_root, val_images)
    ann_path = os.path.join(data_root, val_ann)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Val image folder not found: {img_dir!r}")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Val annotations not found: {ann_path!r}")

    return build_coco_dataloader(
        img_dir,
        ann_path,
        cfg=cfg,
        imgsz=imgsz,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        remap_mscoco_category=remap_mscoco_category,
        multiscale=False,
    )


# Re-exported for callers building custom label maps / class-name lists.
COCO_CATEGORY2LABEL = mscoco_category2label
COCO_CATEGORY2NAME = mscoco_category2name
