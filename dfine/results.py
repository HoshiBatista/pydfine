"""Prediction results — ``Results`` and ``Boxes`` (ultralytics-style).

``DFINE.predict`` returns a ``list[Results]`` (one per input image). Each holds the
original image, the detected :class:`Boxes` (already in original-image pixel scale,
``xyxy``), and the class-name lookup, plus ``.plot()``/``.save()`` for a quick look.

Boxes carry ``.xyxy`` / ``.conf`` / ``.cls`` as CPU tensors; iterate a ``Results``
to zip over them, or index into it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

__all__ = ["Boxes", "Results"]

# Distinct-ish palette; indexed by class id (wraps around).
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


class Results:
    """Detections for one image + helpers to visualize them."""

    def __init__(self, orig_img: Image.Image, boxes: Boxes, names: dict[int, str]):
        self.orig_img = orig_img
        self.boxes = boxes
        self.names = names
        self.orig_shape = (orig_img.height, orig_img.width)

    def __len__(self) -> int:
        return len(self.boxes)

    def __repr__(self) -> str:
        return f"Results(image={self.orig_shape[1]}x{self.orig_shape[0]}, boxes={len(self)})"

    def _label(self, cls_id: int, conf: float, track_id: int | None = None) -> str:
        name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        prefix = f"#{track_id} " if track_id is not None else ""
        return f"{prefix}{name} {conf:.2f}"

    def plot(self, line_width: int | None = None) -> np.ndarray:
        """Draw boxes+labels on a copy of the image; return an RGB HWC uint8 array.

        When the boxes carry track ids (``boxes.id``), each label is prefixed with
        ``#<id>`` and boxes are colored by track id so an object keeps its color.
        """
        img = self.orig_img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        lw = line_width or max(2, round(sum(self.orig_shape) / 600))
        ids = self.boxes.id

        for i, (xyxy, conf, cls) in enumerate(self.boxes):
            cls_id = int(cls)
            track_id = int(ids[i]) if ids is not None else None
            # Color by track id when tracking (stable per object), else by class.
            color = _PALETTE[(track_id if track_id is not None else cls_id) % len(_PALETTE)]
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

    # -- interop ---------------------------------------------------------------

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
                "or `pip install dfine[interop]`."
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
        pass ``image_id`` to tag the detections with a dataset image id. Pure
        Python — no extra dependency.
        """
        out = []
        for xyxy, conf, cls in self.boxes:
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            out.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(conf),
                }
            )
        return out

    def to_supervision(self):
        """Convert to a ``supervision.Detections`` (``xyxy``/``confidence``/``class_id``).

        Boxes are the original-scale ``xyxy`` corners (float32); class ids are the
        contiguous labels. Requires the ``supervision`` package.
        """
        try:
            import supervision as sv
        except ImportError as e:  # pragma: no cover - trivial guard
            raise ImportError(
                "Results.to_supervision() needs supervision — install it with "
                "`pip install supervision` or `pip install dfine[interop]`."
            ) from e

        xyxy = self.boxes.xyxy.cpu().numpy().reshape(-1, 4).astype(np.float32)
        conf = self.boxes.conf.cpu().numpy().reshape(-1).astype(np.float32)
        cls = self.boxes.cls.cpu().numpy().reshape(-1).astype(int)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
