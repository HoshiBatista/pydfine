"""Small shared building blocks for the native port.

Ported from ``D-FINE/src/zoo/dfine/utils.py`` (Apache-2.0, © 2024 The D-FINE
Authors). Only the pieces the ported modules need live here; grows as more modules
land.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Logit of ``x`` (inverse of sigmoid), clamped for numerical stability."""
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Init value for a conv/fc bias given a prior foreground probability."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method: str = "default",
) -> torch.Tensor:
    """Multi-scale deformable attention core (grid-sample based).

    Args:
        value: list of per-level tensors ``[bs, n_head, c, h*w]``.
        value_spatial_shapes: ``[n_levels, 2]`` list of ``(h, w)``.
        sampling_locations: ``[bs, query_length, n_head, sum(points), 2]``.
        attention_weights: ``[bs, query_length, n_head, sum(points)]``.
        num_points_list: points per level.
        method: ``"default"`` (bilinear) or ``"discrete"`` (nearest index).
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # n l c
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


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
