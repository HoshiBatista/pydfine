"""Weight EMA — ported from upstream ``D-FINE/src/optim/ema.py`` (``ModelEMA``).

Keeps a shadow copy of every float parameter/buffer, updated after each optimizer
step as ``ema = decay * ema + (1 - decay) * model``. The decay ramps up early
(``decay * (1 - exp(-updates / warmups))``) so the average is not dominated by the
noisy first steps. The single-process ``de_parallel`` unwrap is dropped (we never
wrap in DDP here).
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA"]


class ModelEMA:
    """Exponential moving average of a model's state_dict (params + buffers)."""

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, warmups: int = 1000, start: int = 0
    ):
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.warmups = warmups
        self.start = start
        self.before_start = 0
        self.updates = 0
        if warmups == 0:
            self.decay_fn = lambda _x: decay
        else:
            self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.before_start < self.start:
            self.before_start += 1
            return
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs) -> ModelEMA:
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.module.load_state_dict(state["module"], strict=strict)
        self.updates = state.get("updates", self.updates)

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"
