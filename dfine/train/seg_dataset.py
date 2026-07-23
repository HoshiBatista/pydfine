"""YOLO-style segmentation datasets (Phase TS4).

Two lightweight datasets that produce exactly the target keys the native seg criteria
consume (see ``dfine.backends.native``):

- :class:`YoloInstanceSegDataset` (``task="segment"``) — reads YOLO-Seg polygon labels
  (``labels/<stem>.txt``: ``cls x1 y1 x2 y2 … xN yN`` normalized, or a 5-col ``cls xc yc w
  h`` detection line) and rasterizes each polygon to a binary instance mask. Yields
  ``boxes`` (cxcywh, normalized), ``labels``, and ``masks`` ``[N, imgsz, imgsz]`` — the
  keys ``DFINECriterion.loss_masks`` / ``HungarianMatcher`` expect.
- :class:`SemSegDataset` (``task="sem_seg"``) — reads a single-channel PNG label map
  (``labels/<stem>.png``, pixel value = class id, ``ignore_index`` excluded) alongside each
  image and yields ``sem_mask`` ``[imgsz, imgsz]`` long — the key ``SemSegCriterion`` expects.

Both resize every image to a square ``imgsz`` (so a plain collate stacks them); boxes are
normalized and thus resize-invariant, instance masks are rasterized at ``imgsz``, and the
dense label map is resized with NEAREST (``ignore_index``-preserving). Mosaic / heavy
augmentation are intentionally out of scope (see ``docs/SEG_ROADMAP.md``). The polygon
rasterization follows D-FINE-seg (``src/dl``; Apache-2.0, © ArgoHA).

Needs ``pip install pydfine[train]`` (``opencv-python`` for ``fillPoly``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .dataset import batch_image_collate_fn

if TYPE_CHECKING:
    from ..config import DFINEConfig

__all__ = [
    "SemSegDataset",
    "YoloInstanceSegDataset",
    "build_seg_dataloader",
    "parse_yolo_seg_label",
    "polygons_to_masks",
]

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_yolo_seg_label(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse a YOLO-Seg label file into normalized boxes and polygons.

    Supports detection lines (``cls xc yc w h``) and segmentation lines
    (``cls x1 y1 … xN yN``, ≥3 points). Returns ``boxes`` ``[N, 5]`` = ``[cls, xc, yc, w, h]``
    (normalized) and a length-``N`` list of ``(K, 2)`` normalized polygons (empty ``(0, 2)``
    for detection-only rows). Mirrors D-FINE-seg's ``parse_yolo_label_file``.
    """
    boxes: list[list[float]] = []
    polys: list[np.ndarray] = []
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32), []

    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        cls = float(parts[0])
        nums = [float(x) for x in parts[1:]]
        if len(nums) == 4:  # detection line
            boxes.append([cls, *nums[:4]])
            polys.append(np.empty((0, 2), dtype=np.float32))
        elif len(nums) >= 6:  # polygon line
            if len(nums) % 2 == 1:  # drop a stray trailing coordinate
                nums = nums[:-1]
            poly = np.asarray(nums, dtype=np.float32).reshape(-1, 2)
            polys.append(poly)
            (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
            boxes.append([cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
        else:
            raise ValueError(f"Invalid label line {path}:{ln}: {s!r}")

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), []
    return np.asarray(boxes, dtype=np.float32), polys


def polygons_to_masks(polys_norm: list[np.ndarray], h: int, w: int) -> torch.Tensor:
    """Rasterize normalized polygons to a ``[N, h, w]`` uint8 mask stack (via ``cv2.fillPoly``).

    A detection-only row (empty polygon) yields an all-zero mask so masks stay aligned 1:1
    with the boxes. Follows D-FINE-seg's ``poly_abs_to_mask``.
    """
    if len(polys_norm) == 0:
        return torch.zeros((0, h, w), dtype=torch.uint8)
    import cv2

    scale = np.asarray([w, h], dtype=np.float32)
    masks = np.zeros((len(polys_norm), h, w), dtype=np.uint8)
    for i, poly in enumerate(polys_norm):
        if poly.shape[0] >= 3:
            pts = np.round(poly * scale).astype(np.int32)
            cv2.fillPoly(masks[i], [pts], 1)
    return torch.from_numpy(masks)


def _list_images(root: Path) -> list[Path]:
    imgs = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no images under {root / 'images'} (expected an images/ folder)")
    return imgs


def _load_image(path: Path, imgsz: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Open an image, return ``(CHW float[0,1] resized to imgsz, (orig_w, orig_h))``."""
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((imgsz, imgsz), Image.BILINEAR)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return arr, (orig_w, orig_h)


class YoloInstanceSegDataset(data.Dataset):
    """YOLO-Seg instance dataset: ``images/<stem>.*`` + ``labels/<stem>.txt`` polygons."""

    def __init__(self, root: str | Path, imgsz: int = 640):
        self.root = Path(root)
        self.imgsz = imgsz
        self.images = _list_images(self.root)
        self.labels_dir = self.root / "labels"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image, (orig_w, orig_h) = _load_image(img_path, self.imgsz)
        boxes, polys = parse_yolo_seg_label(self.labels_dir / f"{img_path.stem}.txt")

        labels = torch.from_numpy(boxes[:, 0].astype(np.int64))
        boxes_t = torch.from_numpy(boxes[:, 1:5].astype(np.float32))  # cxcywh, normalized
        masks = polygons_to_masks(polys, self.imgsz, self.imgsz)
        target = {
            "boxes": boxes_t,
            "labels": labels,
            "masks": masks,
            "orig_size": torch.tensor([orig_w, orig_h]),
            "image_path": str(img_path),
            "idx": torch.tensor([idx]),
        }
        return image, target


class SemSegDataset(data.Dataset):
    """Semantic-seg dataset: ``images/<stem>.*`` + ``labels/<stem>.png`` dense label maps."""

    def __init__(
        self, root: str | Path, num_classes: int, imgsz: int = 640, ignore_index: int = 255
    ):
        self.root = Path(root)
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.images = _list_images(self.root)
        self.labels_dir = self.root / "labels"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image, (orig_w, orig_h) = _load_image(img_path, self.imgsz)

        mask_path = self.labels_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"sem_seg mask not found: {mask_path}")
        mask_img = Image.open(mask_path).resize((self.imgsz, self.imgsz), Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.int64)
        invalid = (mask >= self.num_classes) & (mask != self.ignore_index)
        if invalid.any():
            raise ValueError(
                f"{mask_path}: class id {int(mask[invalid][0])} >= num_classes={self.num_classes} "
                f"(ignore_index={self.ignore_index}); masks must use contiguous 0..num_classes-1"
            )
        target = {
            "sem_mask": torch.from_numpy(mask),
            "orig_size": torch.tensor([orig_w, orig_h]),
            "image_path": str(img_path),
            "idx": torch.tensor([idx]),
        }
        return image, target


def build_seg_dataloader(
    root: str | Path,
    *,
    cfg: DFINEConfig | None = None,
    task: str | None = None,
    num_classes: int | None = None,
    imgsz: int = 640,
    ignore_index: int = 255,
    batch_size: int = 4,
    train: bool = True,
    shuffle: bool | None = None,
    num_workers: int = 4,
    drop_last: bool | None = None,
) -> data.DataLoader:
    """Build a segmentation dataloader for a YOLO-style ``root`` (``images/`` + ``labels/``).

    Pass ``cfg`` (a :class:`~dfine.config.DFINEConfig`) to inherit ``task`` / ``num_classes`` /
    ``imgsz`` / ``sem_seg_ignore_index``; explicit kwargs override it. ``task="segment"`` builds
    :class:`YoloInstanceSegDataset`, ``task="sem_seg"`` builds :class:`SemSegDataset`. Yields
    ``(images, targets)`` batches consumable directly by ``DFINE.train`` / ``dfine.train.Trainer``.
    """
    if cfg is not None:
        task = task or cfg.task
        num_classes = num_classes if num_classes is not None else cfg.num_classes
        imgsz = cfg.imgsz
        ignore_index = cfg.sem_seg_ignore_index
    if shuffle is None:
        shuffle = train
    if drop_last is None:
        drop_last = train

    if task == "sem_seg":
        if num_classes is None:
            raise ValueError("sem_seg needs num_classes (pass it or a cfg).")
        dataset: data.Dataset = SemSegDataset(root, num_classes, imgsz, ignore_index)
    elif task == "segment":
        dataset = YoloInstanceSegDataset(root, imgsz)
    else:
        raise ValueError(f"build_seg_dataloader supports task in (segment, sem_seg), got {task!r}.")

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=batch_image_collate_fn,
        drop_last=drop_last,
    )
