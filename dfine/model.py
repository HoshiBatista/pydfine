"""The public, backend-agnostic ``DFINE`` class.

This is the headline API: build a fully-typed model from Python params (a preset
plus any overrides — no YAML), optionally load released weights, and ``predict``.
It wraps the native backend (assembled model + postprocessor) but never leaks
backend details through its kwargs.

    from dfine import DFINE

    model = DFINE(size="l", num_classes=80)     # architecture (ImageNet backbone)
    model.load("dfine-l")                        # released COCO weights
    results = model.predict("street.jpg", conf=0.4)
    results[0].save("out.jpg")

    model = DFINE.from_pretrained("dfine-s")     # one-liner: build + download + load
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from .config import DFINEConfig
from .results import Boxes, Results

__all__ = ["DFINE"]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coco_names() -> dict[int, str]:
    # Contiguous 0..79 label -> COCO display name (postprocessor labels are contiguous).
    from .backends.native.coco import mscoco_category2name

    return dict(enumerate(mscoco_category2name.values()))


def _build_names(cfg: DFINEConfig) -> dict[int, str]:
    if cfg.class_names:
        return dict(enumerate(cfg.class_names))
    if cfg.num_classes == 80:
        return _coco_names()
    return {i: f"class_{i}" for i in range(cfg.num_classes)}


def _to_pil(item) -> Image.Image:
    """Coerce a path / PIL image / RGB HWC array into an RGB ``PIL.Image``."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, os.PathLike)):
        return Image.open(item).convert("RGB")
    try:
        import numpy as np

        if isinstance(item, np.ndarray):
            return Image.fromarray(item).convert("RGB")
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"Unsupported image source: {type(item).__name__}")


def _load_images(source) -> list[Image.Image]:
    if isinstance(source, (list, tuple)):
        return [_to_pil(s) for s in source]
    return [_to_pil(source)]


class DFINE:
    """Config-first D-FINE detector with an ultralytics-style ``predict``."""

    def __init__(
        self,
        size: str | None = None,
        *,
        weights: str | os.PathLike | None = None,
        device: str | torch.device | None = None,
        **params,
    ):
        self.config = DFINEConfig.preset(size, **params) if size else DFINEConfig(**params)
        self.device = _resolve_device(device)
        self.names = _build_names(self.config)

        # Backend is imported here (not at module top) so it's only pulled in when a
        # model is actually built.
        from .backends.native import DFINE as _NativeDFINE
        from .backends.native import DFINEPostProcessor

        self.model = _NativeDFINE.from_config(self.config).to(self.device).eval()
        self.postprocessor = DFINEPostProcessor.from_config(self.config).to(self.device).eval()

        if weights is not None:
            self.load(weights)

    @classmethod
    def from_pretrained(
        cls, name: str, device: str | torch.device | None = None, **overrides
    ) -> DFINE:
        """Build a model matching a released checkpoint and load its weights.

        ``name`` is a catalogue entry (``"dfine-s"``, ``"dfine-l-obj365"`` ...); the
        size and ``num_classes`` are taken from it. See ``dfine models``.
        """
        from .registry import resolve

        spec = resolve(name)
        # Skip the ImageNet backbone fetch — the checkpoint overwrites it anyway.
        model = cls(
            size=spec.size,
            device=device,
            num_classes=spec.num_classes,
            backbone_pretrained=False,
            **overrides,
        )
        model.load(name)
        return model

    def load(self, weights: str | os.PathLike, use_ema: bool = True) -> DFINE:
        """Load weights into the model, in place.

        ``weights`` is either a catalogue name (downloaded + cached) or a local
        ``.pth`` path. Returns ``self`` for chaining: ``DFINE(size="s").load("dfine-s")``.
        """
        from .backends.native.loader import load_checkpoint
        from .downloads import download_weights
        from .registry import CHECKPOINTS

        if isinstance(weights, str) and weights.lower() in CHECKPOINTS:
            path = download_weights(weights.lower())
        else:
            path = Path(weights)
            if not path.exists():
                raise FileNotFoundError(
                    f"{weights!r} is neither a known checkpoint name nor an existing file."
                )
        load_checkpoint(self.model, path, use_ema=use_ema, strict=True)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def predict(self, source, conf: float = 0.25, imgsz: int | None = None) -> list[Results]:
        """Detect objects in ``source`` (path / PIL / array, or a list of them).

        Returns one :class:`~dfine.results.Results` per image; boxes are in the
        original pixel scale. ``conf`` drops low-scoring detections.
        """
        images = _load_images(source)
        size = imgsz or self.config.imgsz
        transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

        batch = torch.stack([transform(im) for im in images]).to(self.device)
        orig_sizes = torch.tensor([[im.width, im.height] for im in images], device=self.device)

        outputs = self.model(batch)
        detections = self.postprocessor(outputs, orig_sizes)
        return [self._to_results(im, det, conf) for im, det in zip(images, detections)]

    __call__ = predict

    def _to_results(self, image: Image.Image, det: dict, conf: float) -> Results:
        scores, labels, boxes = det["scores"], det["labels"], det["boxes"]
        keep = scores > conf
        boxes = Boxes(
            xyxy=boxes[keep].cpu(),
            conf=scores[keep].cpu(),
            cls=labels[keep].cpu(),
        )
        return Results(image, boxes, self.names)

    def _not_ready(self, name: str, phase: str):
        raise NotImplementedError(f"DFINE.{name}() is not implemented yet — arriving in {phase}.")

    def train(self, *args, **kwargs):
        self._not_ready("train", "Phase 4 (training)")

    def val(self, *args, **kwargs):
        self._not_ready("val", "Phase 4 (training)")

    def export(self, *args, **kwargs):
        self._not_ready("export", "Phase 3 (export)")

    def predict_video(self, *args, **kwargs):
        self._not_ready("predict_video", "Phase 2 (video)")

    def __repr__(self) -> str:
        size = self.config.size or "custom"
        return (
            f"DFINE(size={size!r}, num_classes={self.config.num_classes}, device='{self.device}')"
        )
