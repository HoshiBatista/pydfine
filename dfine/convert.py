"""Convert a YOLO-format detection dataset into the COCO layout ``dfine`` trains on.

YOLO stores one ``.txt`` per image with ``class cx cy w h`` rows (normalized
``cxcywh``, class 0-indexed), images under ``images/<split>/`` and labels under the
mirror ``labels/<split>/`` path; an optional ``data.yaml`` names the classes and points
at the splits. D-FINE (via :func:`dfine.train.dataset.build_coco_dataloaders`) wants the
COCO layout instead::

    output_dir/
      train2017/                          # images
      val2017/
      annotations/
        instances_train2017.json
        instances_val2017.json

:func:`yolo_to_coco` writes exactly that, so the result feeds ``DFINE.train(data=…)`` /
``DFINE.val(data=…)`` directly. **Category ids are kept 0-indexed (= the YOLO class
id)** so they line up with the model's contiguous labels under the default
``remap_mscoco_category=False`` — for both training and COCO eval.

Torch-free: only needs Pillow (image sizes) and, to read a ``data.yaml``, PyYAML —
both imported lazily. Segmentation-style rows (a class id followed by polygon points)
are accepted too; their bounding box is derived.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

__all__ = ["yolo_to_coco"]

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# YOLO split -> COCO split-folder name (matches build_coco_dataloaders defaults).
_DEFAULT_SPLIT_NAMES = {"train": "train2017", "val": "val2017", "test": "test2017"}


def _require_pil():
    try:
        from PIL import Image

        return Image
    except ImportError as exc:  # pragma: no cover - exercised only without pillow
        raise ImportError(
            "yolo_to_coco needs Pillow to read image sizes — install with "
            "`pip install dfine[torch]`."
        ) from exc


def _label_path_for_image(image: Path) -> Path:
    """Mirror an ``.../images/<...>/x.jpg`` path to ``.../labels/<...>/x.txt`` (YOLO rule)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):  # replace the *last* "images" segment
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _iter_images(image_dir: Path):
    """Yield image files under ``image_dir`` (recursive), in a stable sorted order."""
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            yield path


def _parse_label_line(line: str) -> tuple[int, list[float]] | None:
    """Parse one YOLO row into ``(class_id, [cx, cy, w, h])`` (bbox or polygon->bbox)."""
    tok = line.split()
    if len(tok) < 5:
        return None
    cls = int(float(tok[0]))
    coords = [float(v) for v in tok[1:]]
    if len(coords) == 4:
        cx, cy, w, h = coords
    else:  # polygon (x1,y1,x2,y2,...) -> tight bbox
        xs, ys = coords[0::2], coords[1::2]
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return cls, [cx, cy, w, h]


def _resolve_class_names(yolo_root: Path, class_names) -> list[str] | None:
    """Explicit names win; else read ``data.yaml`` ``names``; else ``None`` (infer later)."""
    if class_names is not None:
        return list(class_names)
    for cand in ("data.yaml", "data.yml"):
        yaml_path = yolo_root / cand
        if yaml_path.is_file():
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    f"Reading {cand} needs PyYAML — install `pip install dfine[train]` "
                    "or pass class_names=[...] explicitly."
                ) from exc
            names = yaml.safe_load(yaml_path.read_text()).get("names")
            if isinstance(names, dict):  # {0: 'cat', 1: 'dog'}
                return [names[k] for k in sorted(names)]
            if names is not None:
                return list(names)
    return None


def _detect_splits(yolo_root: Path, splits) -> dict[str, Path]:
    """Map ``split -> image dir``. Explicit ``splits`` win; else auto-detect conventions."""
    if splits is not None:
        return {s: (yolo_root / p if not os.path.isabs(p) else Path(p)) for s, p in splits.items()}
    found: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for cand in (yolo_root / "images" / split, yolo_root / split / "images"):
            if cand.is_dir():
                found[split] = cand
                break
    if not found:
        raise FileNotFoundError(
            f"No YOLO splits found under {yolo_root!r} (looked for images/<split> and "
            "<split>/images). Pass splits={'train': 'path/to/images', ...} explicitly."
        )
    return found


def _convert_split(
    image_dir: Path,
    out_image_dir: Path,
    num_classes_seen: list[int],
    copy_images: bool,
) -> dict:
    """Build a COCO dict for one split and materialize its images into ``out_image_dir``."""
    Image = _require_pil()
    out_image_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id, used_names = 1, {}
    for img_id, img_path in enumerate(_iter_images(image_dir), start=1):
        with Image.open(img_path) as im:
            w, h = im.size

        # Flatten into the output dir, disambiguating any duplicate basenames.
        file_name = img_path.name
        if file_name in used_names:
            used_names[file_name] += 1
            file_name = f"{img_path.stem}_{used_names[file_name]}{img_path.suffix}"
        else:
            used_names[file_name] = 0
        dst = out_image_dir / file_name
        if copy_images:
            shutil.copy2(img_path, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(img_path.resolve())

        images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        label_path = _label_path_for_image(img_path)
        if not label_path.is_file():
            continue  # image with no objects (a valid background/negative sample)
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_label_line(line)
            if parsed is None:
                continue
            cls, (cx, cy, bw, bh) = parsed
            x = max(0.0, (cx - bw / 2) * w)
            y = max(0.0, (cy - bh / 2) * h)
            bw_px = min(bw * w, w - x)
            bh_px = min(bh * h, h - y)
            if bw_px <= 0 or bh_px <= 0:
                continue
            num_classes_seen.append(cls)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,  # 0-indexed, == YOLO class id
                    "bbox": [x, y, bw_px, bh_px],
                    "area": bw_px * bh_px,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations}


def yolo_to_coco(
    yolo_root: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    class_names: list[str] | None = None,
    splits: dict[str, str] | None = None,
    copy_images: bool = True,
    split_names: dict[str, str] | None = None,
) -> dict[str, str]:
    """Convert a YOLO detection dataset to the COCO layout under ``output_dir``.

    Args:
        yolo_root: dataset root (with ``images/<split>`` + ``labels/<split>``, and an
            optional ``data.yaml``).
        output_dir: where the COCO ``train2017/``/``val2017/`` + ``annotations/`` are
            written (consumable directly by ``DFINE.train(data=output_dir)``).
        class_names: class names (index = class id). Falls back to ``data.yaml``'s
            ``names``, then to inferred ``class_<i>`` if neither is available.
        splits: explicit ``{split: image_dir}`` (relative to ``yolo_root`` or absolute).
            Defaults to auto-detecting ``images/<split>`` and ``<split>/images``.
        copy_images: copy images (default) or symlink them into the output.
        split_names: override the split→folder map (default ``train→train2017``,
            ``val→val2017``, ``test→test2017``).

    Returns:
        ``{output_split_name: annotation_json_path}`` for each converted split.
    """
    yolo_root = Path(yolo_root)
    output_dir = Path(output_dir)
    split_names = split_names or _DEFAULT_SPLIT_NAMES

    names = _resolve_class_names(yolo_root, class_names)
    split_dirs = _detect_splits(yolo_root, splits)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    num_classes_seen: list[int] = []
    coco_splits: dict[str, dict] = {}
    for split, image_dir in split_dirs.items():
        out_name = split_names.get(split, split)
        coco_splits[out_name] = _convert_split(
            image_dir, output_dir / out_name, num_classes_seen, copy_images
        )
        logger.info(
            "converted %s: %d images, %d annotations",
            split,
            len(coco_splits[out_name]["images"]),
            len(coco_splits[out_name]["annotations"]),
        )

    # Resolve the category table: explicit/yaml names, else infer from the labels.
    if names is None:
        n = (max(num_classes_seen) + 1) if num_classes_seen else 0
        names = [f"class_{i}" for i in range(n)]
    elif num_classes_seen and max(num_classes_seen) >= len(names):
        # Guard against a names list shorter than the labels: otherwise annotations would
        # carry a category_id with no `categories` entry and COCO eval fails cryptically.
        raise ValueError(
            f"labels reference class id {max(num_classes_seen)} but only {len(names)} class "
            f"name(s) were provided — pass class_names with at least "
            f"{max(num_classes_seen) + 1} entries, or omit it to infer names from the labels."
        )
    categories = [{"id": i, "name": name} for i, name in enumerate(names)]

    written: dict[str, str] = {}
    for out_name, coco in coco_splits.items():
        coco["categories"] = categories
        ann_path = output_dir / "annotations" / f"instances_{out_name}.json"
        ann_path.write_text(json.dumps(coco))
        written[out_name] = str(ann_path)
    return written
