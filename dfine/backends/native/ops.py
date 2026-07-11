"""Small shared building blocks for the native port.

Ported from ``D-FINE/src/zoo/dfine/utils.py`` (Apache-2.0, © 2024 The D-FINE
Authors). Only the pieces the ported modules need live here; grows as more modules
land.
"""

from __future__ import annotations

from torch import nn


def get_activation(act: str | nn.Module | None, inplace: bool = True) -> nn.Module:
    """Resolve an activation name (or module) to an ``nn.Module``."""
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ("silu", "swish"):
        m: nn.Module = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unsupported activation: {act!r}")

    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m
