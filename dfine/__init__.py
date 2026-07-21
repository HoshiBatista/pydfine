"""dfine — a config-first, ultralytics-style wrapper around D-FINE.

The whole model is configured by typed params on one class. Phase 0/1 ships the
config surface (``DFINEConfig``, presets); ``DFINE`` and ``Results`` land with the
inference backend (Phase 2) and are exposed lazily so importing this package never
requires torch until you actually build a model.
"""

from __future__ import annotations

from typing import Any

from .config import SIZE_PRESETS, SIZES, DFINEConfig, list_presets
from .convert import yolo_to_coco
from .registry import list_checkpoints

__version__ = "0.0.1"

__all__ = [
    "DFINEConfig",
    "SIZE_PRESETS",
    "SIZES",
    "list_presets",
    "list_checkpoints",
    "yolo_to_coco",
    "__version__",
]

_LAZY = {
    "DFINE": "model",
    "Results": "results",
    "Boxes": "results",
    "Masks": "results",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        try:
            mod = __import__(f"dfine.{_LAZY[name]}", fromlist=[name])
        except ImportError as exc:
            raise AttributeError(
                f"dfine.{name} needs the inference deps — install with "
                f"`pip install pydfine[torch]` (missing: {exc.name})."
            ) from exc
        return getattr(mod, name)
    raise AttributeError(f"module 'dfine' has no attribute {name!r}")
