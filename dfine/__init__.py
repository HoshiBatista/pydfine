"""dfine — a config-first, ultralytics-style wrapper around D-FINE.

The whole model is configured by typed params on one class. Phase 0/1 ships the
config surface (``DFINEConfig``, presets); ``DFINE`` and ``Results`` land with the
inference backend (Phase 2) and are exposed lazily so importing this package never
requires torch until you actually build a model.
"""

from __future__ import annotations

from typing import Any

from .config import SIZE_PRESETS, SIZES, DFINEConfig, list_presets
from .registry import list_checkpoints

__version__ = "0.0.1"

__all__ = [
    "DFINEConfig",
    "SIZE_PRESETS",
    "SIZES",
    "list_presets",
    "list_checkpoints",
    "__version__",
]

# Names that will resolve once their modules exist. Kept here so `from dfine import
# DFINE` fails with a clear "not implemented yet" instead of a cryptic ImportError.
_PENDING = {
    "DFINE": ("model", "Phase 2 (backend + inference)"),
    "Results": ("results", "Phase 2 (backend + inference)"),
    "Boxes": ("results", "Phase 2 (backend + inference)"),
}


def __getattr__(name: str) -> Any:
    if name in _PENDING:
        module, phase = _PENDING[name]
        try:
            mod = __import__(f"dfine.{module}", fromlist=[name])
        except ImportError as exc:  # module not written yet
            raise AttributeError(
                f"dfine.{name} is not implemented yet — arriving in {phase}."
            ) from exc
        return getattr(mod, name)
    raise AttributeError(f"module 'dfine' has no attribute {name!r}")
