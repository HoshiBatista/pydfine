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
    """Detected boxes for one image: ``xyxy`` (pixels), ``conf``, ``cls``."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

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

    def _label(self, cls_id: int, conf: float) -> str:
        name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        return f"{name} {conf:.2f}"

    def plot(self, line_width: int | None = None) -> np.ndarray:
        """Draw boxes+labels on a copy of the image; return an RGB HWC uint8 array."""
        img = self.orig_img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        lw = line_width or max(2, round(sum(self.orig_shape) / 600))

        for xyxy, conf, cls in self.boxes:
            cls_id = int(cls)
            color = _PALETTE[cls_id % len(_PALETTE)]
            box = [float(v) for v in xyxy]
            draw.rectangle(box, outline=color, width=lw)

            text = self._label(cls_id, float(conf))
            tl = draw.textbbox((box[0], box[1]), text)
            draw.rectangle([tl[0], tl[1], tl[2], tl[3]], fill=color)
            draw.text((box[0], box[1]), text, fill=(255, 255, 255))

        return np.asarray(img)

    def save(self, filename: str | Path) -> Path:
        """Render via :meth:`plot` and write to ``filename``; return the path."""
        path = Path(filename)
        Image.fromarray(self.plot()).save(path)
        return path
