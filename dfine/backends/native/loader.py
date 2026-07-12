"""Weight loading for native D-FINE models.

Upstream training checkpoints are dicts holding several state_dicts::

    {"model": <state_dict>, "ema": {"module": <state_dict>, "updates": ...},
     "optimizer": ..., "last_epoch": ..., ...}

Inference uses the EMA weights when present (upstream's ``tools/inference`` does
the same). Because the native port keeps upstream layer/parameter names, the
extracted ``state_dict`` loads into the assembled :class:`DFINE` with
``strict=True`` and no key remapping — that clean load *is* the parity check.

A checkpoint may also be a bare ``state_dict`` (already unwrapped), optionally
with a ``module.`` prefix from ``DataParallel``/``DDP``; both are handled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

__all__ = ["extract_state_dict", "load_checkpoint"]


def _strip_module_prefix(state_dict: dict) -> dict:
    """Drop a leading ``module.`` from every key (from DataParallel/DDP saves)."""
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint: dict, use_ema: bool = True) -> dict:
    """Pull the model ``state_dict`` out of a loaded checkpoint object.

    Prefers EMA weights (``checkpoint["ema"]["module"]``) when ``use_ema`` and
    they exist, else ``checkpoint["model"]``. If ``checkpoint`` is already a bare
    state_dict (tensor values, no wrapper keys), it's returned as-is.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}.")

    if use_ema and isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"]:
        state_dict = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint  # already an unwrapped state_dict
    else:
        raise KeyError(
            "Checkpoint has no 'model'/'ema' weights and is not a bare state_dict; "
            f"top-level keys: {list(checkpoint)}"
        )

    return _strip_module_prefix(state_dict)


def load_checkpoint(model: nn.Module, path, use_ema: bool = True, strict: bool = True):
    """Load an upstream ``.pth`` into ``model`` in place.

    Returns the ``(missing_keys, unexpected_keys)`` lists from
    :meth:`torch.nn.Module.load_state_dict`. With a matching preset and
    ``strict=True`` both are empty.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint, use_ema=use_ema)
    return model.load_state_dict(state_dict, strict=strict)
