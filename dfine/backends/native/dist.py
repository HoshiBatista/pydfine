"""Distributed helpers used by the criterion for loss normalization.

Ported (trimmed) from ``D-FINE/src/misc/dist_utils.py`` (Apache-2.0, © 2024 The
D-FINE Authors). The criterion divides its box counts by the world size so the
loss scale is independent of how many GPUs are training; on a single process
these degrade to ``world_size == 1`` and ``rank == 0``.
"""

from __future__ import annotations

import torch.distributed

__all__ = ["get_rank", "get_world_size", "is_dist_available_and_initialized"]


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()
