"""Checkpoint catalogue — which released weights exist and how to build for them.

Maps friendly names (``dfine-l``, ``dfine-l-obj365`` ...) to the Apache-2.0 D-FINE
release assets, and — crucially — to the ``num_classes`` each was trained with so
the matching model can be assembled and strict-loaded:

- **coco** / **obj2coco** → 80 classes (obj2coco = Objects365-pretrained, then
  COCO fine-tuned; evaluated on COCO).
- **obj365** → 366 classes (Objects365).

Availability is not uniform: **N only has COCO weights** — upstream never released
Objects365 (``obj2coco``/``obj365``) checkpoints for N. Requesting a combination
that doesn't exist raises a clear error listing what *is* available.

Instance-segmentation checkpoints (``dfine-seg-n`` .. ``dfine-seg-x``) come from the
D-FINE-seg project (© ArgoHA, Apache-2.0) and are hosted on Hugging Face rather than
the GitHub releases (``source="hf"``); their ``task`` is ``"segment"`` so
``config_for`` wires the mask head.

``DFINE.from_pretrained(name)`` resolves through here; the download/cache lives in
``dfine/downloads.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DFINEConfig

_BASE = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

# dataset variant -> the num_classes its checkpoints were trained with.
DATASET_NUM_CLASSES: dict[str, int] = {"coco": 80, "obj2coco": 80, "obj365": 366}

# (size, dataset) -> release filename. Only the pairs listed here exist upstream;
# N is COCO-only. L's obj2coco asset carries an "_e25" suffix (no plain variant).
_FILES: dict[tuple[str, str], str] = {
    ("n", "coco"): "dfine_n_coco.pth",
    ("s", "coco"): "dfine_s_coco.pth",
    ("s", "obj2coco"): "dfine_s_obj2coco.pth",
    ("s", "obj365"): "dfine_s_obj365.pth",
    ("m", "coco"): "dfine_m_coco.pth",
    ("m", "obj2coco"): "dfine_m_obj2coco.pth",
    ("m", "obj365"): "dfine_m_obj365.pth",
    ("l", "coco"): "dfine_l_coco.pth",
    ("l", "obj2coco"): "dfine_l_obj2coco_e25.pth",
    ("l", "obj365"): "dfine_l_obj365.pth",
    ("x", "coco"): "dfine_x_coco.pth",
    ("x", "obj2coco"): "dfine_x_obj2coco.pth",
    ("x", "obj365"): "dfine_x_obj365.pth",
}


# Instance-segmentation weights (D-FINE-seg, © ArgoHA, Apache-2.0) are hosted on
# Hugging Face, not the GitHub releases. Same architecture as detection + a mask head.
_SEG_REPO = "ArgoSA/D-FINE-seg"
_SEG_SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class CheckpointSpec:
    """Everything needed to fetch a checkpoint and build a model that loads it."""

    name: str  # friendly name, e.g. "dfine-l-obj365"
    size: str  # preset size the weights were trained for ("n".."x")
    dataset: str  # "coco" | "obj2coco" | "obj365"
    num_classes: int  # 80 (coco/obj2coco) or 366 (obj365)
    filename: str  # release asset filename
    url: str  # full download URL (GitHub source); "" for Hugging Face specs
    task: str = "detect"  # "detect" | "segment"
    source: str = "github"  # "github" (release URL) | "hf" (Hugging Face repo)
    repo_id: str | None = None  # Hugging Face repo id when source == "hf"


def _name_for(size: str, dataset: str) -> str:
    return f"dfine-{size}" if dataset == "coco" else f"dfine-{size}-{dataset}"


def _make_spec(size: str, dataset: str, filename: str) -> CheckpointSpec:
    return CheckpointSpec(
        name=_name_for(size, dataset),
        size=size,
        dataset=dataset,
        num_classes=DATASET_NUM_CLASSES[dataset],
        filename=filename,
        url=f"{_BASE}/{filename}",
    )


def _make_seg_spec(size: str) -> CheckpointSpec:
    return CheckpointSpec(
        name=f"dfine-seg-{size}",
        size=size,
        dataset="coco",
        num_classes=DATASET_NUM_CLASSES["coco"],
        filename=f"dfine_seg_{size}_coco.pt",
        url="",
        task="segment",
        source="hf",
        repo_id=_SEG_REPO,
    )


CHECKPOINTS: dict[str, CheckpointSpec] = {
    **{
        (spec := _make_spec(size, dataset, filename)).name: spec
        for (size, dataset), filename in _FILES.items()
    },
    **{(spec := _make_seg_spec(size)).name: spec for size in _SEG_SIZES},
}


def resolve(name: str) -> CheckpointSpec:
    """Return the :class:`CheckpointSpec` for a friendly name, or raise ``KeyError``."""
    key = name.lower()
    if key not in CHECKPOINTS:
        raise KeyError(f"Unknown checkpoint {name!r}; see `dfine models` for the list.")
    return CHECKPOINTS[key]


def available_datasets(size: str) -> list[str]:
    """Dataset variants released for a size, e.g. ``["coco"]`` for N."""
    return [ds for (sz, ds) in _FILES if sz == size]


def resolve_weights(size: str, dataset: str = "coco") -> CheckpointSpec:
    """Pick the checkpoint for a ``size`` + ``dataset`` — the "which model" logic.

    Raises ``ValueError`` for a combination upstream never released (e.g. N with
    ``obj365``), naming what *is* available for that size.
    """
    key = (size, dataset)
    if key not in _FILES:
        have = available_datasets(size)
        if not have:
            raise ValueError(f"Unknown size {size!r}; expected one of n/s/m/l/x.")
        raise ValueError(
            f"No {dataset!r} checkpoint for size {size!r}; "
            f"upstream released only {have} for {size!r}."
        )
    return _make_spec(size, dataset, _FILES[key])


def config_for(spec: CheckpointSpec | str, **overrides) -> DFINEConfig:
    """Build a :class:`DFINEConfig` whose architecture matches a checkpoint.

    Applies the preset for the checkpoint's size and its ``num_classes`` (so
    obj365's 366-class head is wired), then any user overrides. Keep ``imgsz`` at
    its 640 default: the decoder's persistent ``anchors`` buffer is sized to it,
    so a mismatch breaks the strict load.
    """
    from .config import DFINEConfig

    if isinstance(spec, str):
        spec = resolve(spec)
    # spec supplies num_classes + task; explicit overrides (e.g. a custom head) win.
    defaults = {"num_classes": spec.num_classes, "task": spec.task}
    return DFINEConfig.preset(spec.size, **{**defaults, **overrides})


def list_checkpoints() -> list[str]:
    """Sorted list of known checkpoint names."""
    return sorted(CHECKPOINTS)
