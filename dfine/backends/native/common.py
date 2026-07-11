"""Shared low-level layers for the native D-FINE port.

Ported from ``D-FINE/src/nn/backbone/common.py`` (Apache-2.0, © 2024 The D-FINE
Authors). ``FrozenBatchNorm2d`` itself originates from DETR
(facebookresearch/detr). Kept verbatim in behaviour and parameter/buffer names so
upstream checkpoints load unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed batch statistics and affine params.

    Adds ``eps`` before ``rsqrt`` (as in torchvision's variant) to avoid NaNs on
    non-resnet backbones.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshapes up front to stay fuser-friendly.
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}"


def freeze_norm(module: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.BatchNorm2d`` with a ``FrozenBatchNorm2d``.

    Returns the (possibly replaced) module so callers can reassign the root.
    """
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features)
        return frozen
    for name, child in module.named_children():
        new_child = freeze_norm(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module
