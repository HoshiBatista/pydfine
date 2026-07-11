"""Checkpoint name -> upstream release URL.

Maps friendly names (``dfine-l``, ``dfine-l-obj365`` ...) to the Apache-2.0 D-FINE
release assets. ``DFINE.load(name)`` resolves through here; the actual download/cache
lives in ``dfine/downloads.py`` (added with the inference backend).
"""

from __future__ import annotations

_BASE = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

# name -> (checkpoint url, preset size the weights were trained for)
CHECKPOINTS: dict[str, tuple[str, str]] = {
    # COCO
    "dfine-n": (f"{_BASE}/dfine_n_coco.pth", "n"),
    "dfine-s": (f"{_BASE}/dfine_s_coco.pth", "s"),
    "dfine-m": (f"{_BASE}/dfine_m_coco.pth", "m"),
    "dfine-l": (f"{_BASE}/dfine_l_coco.pth", "l"),
    "dfine-x": (f"{_BASE}/dfine_x_coco.pth", "x"),
    # Objects365 -> COCO fine-tuned
    "dfine-s-obj2coco": (f"{_BASE}/dfine_s_obj2coco.pth", "s"),
    "dfine-m-obj2coco": (f"{_BASE}/dfine_m_obj2coco.pth", "m"),
    "dfine-l-obj2coco": (f"{_BASE}/dfine_l_obj2coco_e25.pth", "l"),
    "dfine-x-obj2coco": (f"{_BASE}/dfine_x_obj2coco.pth", "x"),
    # Objects365 pretrained
    "dfine-s-obj365": (f"{_BASE}/dfine_s_obj365.pth", "s"),
    "dfine-m-obj365": (f"{_BASE}/dfine_m_obj365.pth", "m"),
    "dfine-l-obj365": (f"{_BASE}/dfine_l_obj365.pth", "l"),
    "dfine-x-obj365": (f"{_BASE}/dfine_x_obj365.pth", "x"),
}


def resolve(name: str) -> tuple[str, str]:
    """Return ``(url, size)`` for a checkpoint name, or raise ``KeyError``."""
    key = name.lower()
    if key not in CHECKPOINTS:
        raise KeyError(f"Unknown checkpoint {name!r}; see `dfine models` for the list.")
    return CHECKPOINTS[key]


def list_checkpoints() -> list[str]:
    """Sorted list of known checkpoint names."""
    return sorted(CHECKPOINTS)
