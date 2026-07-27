"""Prediction results — ``Results`` and ``Boxes`` (ultralytics-style).

``DFINE.predict`` returns a ``list[Results]`` (one per input image). Each holds the
original image, the detected :class:`Boxes` (already in original-image pixel scale,
``xyxy``), and the class-name lookup, plus ``.plot()``/``.save()`` for a quick look.

Boxes carry ``.xyxy`` / ``.conf`` / ``.cls`` as CPU tensors; iterate a ``Results``
to zip over them, or index into it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

__all__ = ["Boxes", "Masks", "Results", "SemSeg"]

logger = logging.getLogger(__name__)
_warned_missing_cv2 = False

_PALETTE = [
    (255, 56, 56),
    (255, 159, 56),
    (255, 214, 56),
    (144, 214, 56),
    (56, 214, 126),
    (56, 214, 214),
    (56, 126, 214),
    (90, 56, 214),
    (176, 56, 214),
    (214, 56, 144),
]


def _mask_to_coco_rle(mask: np.ndarray) -> dict:
    """COCO **uncompressed** RLE for a binary ``[H, W]`` mask (dependency-free).

    Run-length encodes in **column-major** (Fortran) order starting from background,
    matching COCO's uncompressed-RLE convention: ``{"size": [h, w], "counts": [...]}``
    where ``counts`` alternates 0-run, 1-run, 0-run, … A mask that starts with a
    foreground pixel gets a leading ``0`` count. Accepted by pycocotools/faster-coco-eval
    ``frPyObjects``/``loadRes``, so it round-trips through COCO mask evaluation.
    """
    h, w = int(mask.shape[0]), int(mask.shape[1])
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    bounds = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    counts = [] if flat[0] == 0 else [0]  # leading 0-run when the mask opens on foreground
    counts.extend(int(x) for x in lengths)
    return {"size": [h, w], "counts": counts}


def _mask_largest_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour of a binary ``[H, W]`` mask as a pixel-space ``[K, 2]``.

    Returns the ``(x, y)`` vertices of the biggest contour, or ``None`` when the mask has
    no usable contour (< 3 points) **or** OpenCV isn't installed — callers then fall back
    to the box corners, so polygon export degrades gracefully rather than crashing a
    ``save_txt``/``summary`` call on a segmentation result. OpenCV is imported lazily; a
    one-time warning is logged when it's missing.
    """
    global _warned_missing_cv2
    try:
        import cv2
    except ImportError:  # OpenCV is only in the [video] extra; degrade to box corners.
        if not _warned_missing_cv2:
            logger.warning(
                "OpenCV not installed — exporting boxes instead of mask polygons. "
                "Install it with `pip install pydfine[video]` for polygon output."
            )
            _warned_missing_cv2 = True
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    poly = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return poly.astype(np.float32) if len(poly) >= 3 else None


class Boxes:
    """Detected boxes for one image: ``xyxy`` (pixels), ``conf``, ``cls``.

    ``id`` holds per-box track ids when the boxes came from a tracker (e.g.
    :meth:`DFINE.predict_video` with ``track=True``); it is ``None`` otherwise.
    """

    def __init__(
        self,
        xyxy: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
        id: torch.Tensor | None = None,
    ):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = id

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self.xyxy[i], self.conf[i], self.cls[i]

    def __repr__(self) -> str:
        return f"Boxes(n={len(self)})"


class Masks:
    """Per-instance binary masks for one image: ``data`` is ``[N, H, W]`` (original scale).

    Rows align 1:1 with the image's :class:`Boxes`. ``data`` is a bool CPU tensor at the
    original image resolution (post-threshold, cleaned to each box); cast as needed.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i]

    def __repr__(self) -> str:
        h, w = (self.data.shape[-2], self.data.shape[-1]) if len(self) else (0, 0)
        return f"Masks(n={len(self)}, {w}x{h})"


class SemSeg:
    """Dense semantic-segmentation label map for one image: ``data`` is uint8 ``[H, W]``.

    Each pixel holds a class id at the original image resolution; ``255`` is treated as
    void/ignore (left un-tinted by :meth:`Results.plot`). Populated for ``task="sem_seg"``
    models; ``None`` on detection/instance-segmentation results.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[-2]), int(self.data.shape[-1]))

    def __repr__(self) -> str:
        h, w = self.shape
        n = int((self.data.unique() != 255).sum()) if self.data.numel() else 0
        return f"SemSeg({w}x{h}, {n} classes)"


class Results:
    """Detections for one image + helpers to visualize them."""

    def __init__(
        self,
        orig_img: Image.Image,
        boxes: Boxes,
        names: dict[int, str],
        masks: Masks | None = None,
        sem_seg: SemSeg | None = None,
    ):
        self.orig_img = orig_img
        self.boxes = boxes
        self.masks = masks
        self.sem_seg = sem_seg
        self.names = names
        self.orig_shape = (orig_img.height, orig_img.width)

    def __len__(self) -> int:
        return len(self.boxes)

    def __repr__(self) -> str:
        extra = (
            ""
            if self.sem_seg is None
            else f", sem_seg={self.sem_seg.shape[1]}x{self.sem_seg.shape[0]}"
        )
        return f"Results(image={self.orig_shape[1]}x{self.orig_shape[0]}, boxes={len(self)}{extra})"

    def _name(self, cls_id: int) -> str:
        """Class name for ``cls_id`` from ``names`` (falls back to the numeric id)."""
        return self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)

    def _label(self, cls_id: int, conf: float, track_id: int | None = None) -> str:
        prefix = f"#{track_id} " if track_id is not None else ""
        return f"{prefix}{self._name(cls_id)} {conf:.2f}"

    def plot(self, line_width: int | None = None) -> np.ndarray:
        """Draw boxes+labels on a copy of the image; return an RGB HWC uint8 array.

        When the boxes carry track ids (``boxes.id``), each label is prefixed with
        ``#<id>`` and boxes are colored by track id so an object keeps its color.
        Instance masks (``self.masks``), when present, are overlaid semi-transparently
        in each detection's color before boxes/labels are drawn on top. A semantic-
        segmentation label map (``self.sem_seg``) is overlaid per class with a palette
        color (``255`` void pixels left untouched).
        """
        img = self.orig_img.convert("RGB").copy()
        ids = self.boxes.id

        def _color(i: int, cls_id: int) -> tuple[int, int, int]:
            track_id = int(ids[i]) if ids is not None else None
            return _PALETTE[(track_id if track_id is not None else cls_id) % len(_PALETTE)]

        if self.sem_seg is not None:
            arr = np.asarray(img).astype(np.float32)
            labels = np.asarray(self.sem_seg.data)
            alpha = 0.5
            for cls_id in np.unique(labels):
                if int(cls_id) == 255:
                    continue
                color = np.array(_PALETTE[int(cls_id) % len(_PALETTE)], dtype=np.float32)
                sel = labels == cls_id
                arr[sel] = arr[sel] * (1 - alpha) + color * alpha
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

        if self.masks is not None and len(self.masks):
            arr = np.asarray(img).astype(np.float32)
            alpha = 0.5
            for i, m in enumerate(self.masks):
                mask = np.asarray(m).astype(bool)
                color = np.array(_color(i, int(self.boxes.cls[i])), dtype=np.float32)
                arr[mask] = arr[mask] * (1 - alpha) + color * alpha
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

        draw = ImageDraw.Draw(img)
        lw = line_width or max(2, round(sum(self.orig_shape) / 600))

        for i, (xyxy, conf, cls) in enumerate(self.boxes):
            cls_id = int(cls)
            track_id = int(ids[i]) if ids is not None else None
            color = _color(i, cls_id)
            box = [float(v) for v in xyxy]
            draw.rectangle(box, outline=color, width=lw)

            text = self._label(cls_id, float(conf), track_id)
            tl = draw.textbbox((box[0], box[1]), text)
            draw.rectangle([tl[0], tl[1], tl[2], tl[3]], fill=color)
            draw.text((box[0], box[1]), text, fill=(255, 255, 255))

        return np.asarray(img)

    def save(self, filename: str | Path) -> Path:
        """Render via :meth:`plot` and write to ``filename``; return the path."""
        path = Path(filename)
        Image.fromarray(self.plot()).save(path)
        return path

    def to_pandas(self):
        """Return detections as a ``pandas.DataFrame`` (one row per box).

        Columns ``xmin, ymin, xmax, ymax, confidence, class, name`` — the
        ultralytics ``.pandas().xyxy[0]`` layout. An empty ``Results`` yields an
        empty frame that still carries those columns. Requires ``pandas``.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - trivial guard
            raise ImportError(
                "Results.to_pandas() needs pandas — install it with `pip install pandas` "
                "or `pip install pydfine[interop]`."
            ) from e

        columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
        rows = []
        for xyxy, conf, cls in self.boxes:
            cls_id = int(cls)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            rows.append(
                {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": float(conf),
                    "class": cls_id,
                    "name": self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def to_coco(self, image_id: int = 0) -> list[dict]:
        """Detections as COCO-format result dicts (the ``loadRes`` layout).

        Each box becomes ``{"image_id", "category_id", "bbox": [x, y, w, h],
        "score"}`` with the bbox in COCO ``xywh`` (top-left + size, original-image
        pixels). ``category_id`` is the contiguous class id this library predicts;
        pass ``image_id`` to tag the detections with a dataset image id. When instance
        masks are present (``task="segment"``) each dict also carries a ``segmentation``
        in COCO **uncompressed** RLE (``{"size": [h, w], "counts": [...]}``, original
        scale, aligned 1:1 with the box) — normalize to compressed RLE with the standard
        ``pycocotools``/``faster_coco_eval`` ``frPyObjects`` when a tool needs it. Pure
        Python — no extra dependency.
        """
        masks = None
        if self.masks is not None and len(self.masks):
            masks = self.masks.data.cpu().numpy().astype(np.uint8)
        out = []
        for i, (xyxy, conf, cls) in enumerate(self.boxes):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            d = {
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(conf),
            }
            if masks is not None:
                d["segmentation"] = _mask_to_coco_rle(masks[i])
            out.append(d)
        return out

    def to_supervision(self):
        """Convert to a ``supervision.Detections`` (``xyxy``/``confidence``/``class_id``).

        Boxes are the original-scale ``xyxy`` corners (float32); class ids are the
        contiguous labels. Instance masks (when present) are attached as a bool
        ``[N, H, W]`` ``mask`` array. Requires the ``supervision`` package.
        """
        try:
            import supervision as sv
        except ImportError as e:  # pragma: no cover - trivial guard
            raise ImportError(
                "Results.to_supervision() needs supervision — install it with "
                "`pip install supervision` or `pip install pydfine[interop]`."
            ) from e

        xyxy = self.boxes.xyxy.cpu().numpy().reshape(-1, 4).astype(np.float32)
        conf = self.boxes.conf.cpu().numpy().reshape(-1).astype(np.float32)
        cls = self.boxes.cls.cpu().numpy().reshape(-1).astype(int)
        mask = None
        if self.masks is not None and len(self.masks):
            mask = self.masks.data.cpu().numpy().astype(bool)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, mask=mask)

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> Path | None:
        """Write detections to a YOLO-format ``.txt`` label file (ultralytics-style).

        One line per detection, coordinates **normalized** to the original image size:

        * detection — ``class cx cy w h`` (box center + size, ``cxcywh``);
        * segmentation — ``class x1 y1 x2 y2 … xn yn`` (the mask's largest polygon;
          falls back to the box corners when a mask has no contour).

        ``save_conf=True`` appends the confidence as the final field. Lines are
        **appended** to ``txt_file`` (matching ultralytics — pass a per-image path), the
        parent directory is created, and the format round-trips through
        :func:`~dfine.convert.yolo_to_coco`. Returns the path, or ``None`` when there are
        no detections (no file is written). Polygon extraction needs OpenCV
        (``pip install pydfine[video]``); without it — or when a mask has no contour — the
        row falls back to the box corners, so the detection path is dependency-free.
        """
        if len(self) == 0:
            return None
        h, w = self.orig_shape
        masks = self.masks.data.cpu().numpy().astype(np.uint8) if self.masks is not None else None

        lines: list[str] = []
        for i, (xyxy, conf, cls) in enumerate(self.boxes):
            coords: list[float] | None = None
            if masks is not None:
                poly = _mask_largest_polygon(masks[i])
                if poly is not None:
                    coords = [float(v) for xy in poly for v in (xy[0] / w, xy[1] / h)]
            if coords is None:
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                coords = [cx, cy, (x2 - x1) / w, (y2 - y1) / h]
            row = [int(cls), *coords]
            fmt = " ".join(["%d"] + ["%.6f"] * len(coords)) % tuple(row)
            if save_conf:
                fmt += f" {float(conf):.6f}"
            lines.append(fmt)

        path = Path(txt_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Detections as a list of plain dicts (ultralytics ``Results.summary()`` layout).

        One dict per detection: ``{"name", "class", "confidence", "box": {x1,y1,x2,y2}}``.
        A track id (from :meth:`DFINE.predict_video` with ``track=True``) adds
        ``"track_id"``; an instance mask adds ``"segments": {"x": [...], "y": [...]}`` (the
        largest polygon). Coordinates are original-image pixels, or fractions of the image
        when ``normalize=True``; floats are rounded to ``decimals``. Pure Python — the
        result is JSON-serializable (see :meth:`tojson`).
        """
        h, w = self.orig_shape
        ids = self.boxes.id
        masks = self.masks.data.cpu().numpy().astype(np.uint8) if self.masks is not None else None
        sx, sy = (w, h) if normalize else (1.0, 1.0)

        out = []
        for i, (xyxy, conf, cls) in enumerate(self.boxes):
            cls = int(cls)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            row = {
                "name": self._name(cls),
                "class": cls,
                "confidence": round(float(conf), decimals),
                "box": {
                    "x1": round(x1 / sx, decimals),
                    "y1": round(y1 / sy, decimals),
                    "x2": round(x2 / sx, decimals),
                    "y2": round(y2 / sy, decimals),
                },
            }
            if ids is not None:
                row["track_id"] = int(ids[i])
            if masks is not None:
                poly = _mask_largest_polygon(masks[i])
                if poly is not None:
                    row["segments"] = {
                        "x": [round(float(x) / sx, decimals) for x in poly[:, 0]],
                        "y": [round(float(y) / sy, decimals) for y in poly[:, 1]],
                    }
            out.append(row)
        return out

    def tojson(self, normalize: bool = False, decimals: int = 5) -> str:
        """:meth:`summary` serialized to a JSON string (ultralytics ``Results.tojson``)."""
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)

    def save_crop(self, save_dir: str | Path, file_name: str = "im.jpg") -> list[Path]:
        """Save each detection's cropped image to ``save_dir/<class_name>/`` (ultralytics-style).

        The original image is cropped to every box (clipped to the frame) and written under
        a per-class subfolder as ``<file_name>`` — a numeric suffix (``_2``, ``_3``, …) is
        appended when several detections of the same class would collide, so nothing is
        overwritten. Returns the list of written paths (empty when there are no detections).
        Uses PIL only (dependency-free).
        """
        if len(self) == 0:
            return []
        save_dir = Path(save_dir)
        stem, suffix = Path(file_name).stem, (Path(file_name).suffix or ".jpg")
        img = self.orig_img.convert("RGB")
        img_w, img_h = img.width, img.height

        saved: list[Path] = []
        for xyxy, _conf, cls in self.boxes:
            cls = int(cls)
            name = self._name(cls)
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:  # box fully outside / degenerate after clipping
                continue
            cls_dir = save_dir / name
            cls_dir.mkdir(parents=True, exist_ok=True)
            path = cls_dir / f"{stem}{suffix}"
            k = 2
            while path.exists():
                path = cls_dir / f"{stem}_{k}{suffix}"
                k += 1
            img.crop((x1, y1, x2, y2)).save(path)
            saved.append(path)
        return saved

    def verbose(self) -> str:
        """A human-readable per-class summary, e.g. ``"2 persons, 1 car"`` (ultralytics-style).

        Counts detections per class (naive plural ``s``), ordered by class id. For a
        ``sem_seg`` result it instead lists the classes present in the label map (``255``
        void excluded). Returns ``"(no detections)"`` / ``"(empty)"`` when there is nothing.
        """
        if self.sem_seg is not None:
            ids = [int(c) for c in np.unique(np.asarray(self.sem_seg.data)) if int(c) != 255]
            if not ids:
                return "(empty)"
            names = ", ".join(self._name(i) for i in ids)
            return f"{len(ids)} classes: {names}"
        if len(self) == 0:
            return "(no detections)"
        counts: dict[int, int] = {}
        for c in self.boxes.cls:
            counts[int(c)] = counts.get(int(c), 0) + 1
        return ", ".join(
            f"{n} {self._name(cls_id)}{'s' if n > 1 else ''}"
            for cls_id, n in sorted(counts.items())
        )
