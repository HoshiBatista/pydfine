"""LR schedules — per-iteration warmup + per-epoch flat-cosine / multistep.

D-FINE warms the LR up linearly over the first ``warmup_iters`` optimizer steps,
then follows an epoch-stepped schedule. Two epoch schedules are supported (selected
by ``DFINEConfig.scheduler``):

* ``"flatcosine"`` (default) — hold the base LR flat for the augmented epochs, then
  cosine-anneal down to ``lr_min_ratio`` across the trailing ``no_aug_epoch`` epochs.
* ``"multistep"`` — plain ``MultiStepLR`` (``milestones``/``gamma``).

INTENTIONAL DEVIATION FROM UPSTREAM (kept on purpose): upstream D-FINE configures
``MultiStepLR(milestones=[500], gamma=0.1)``. Because every released recipe trains for
72–160 epochs — all well below 500 — that milestone never fires, so upstream's LR is
effectively **flat/constant** after warmup, with no annealing even during the no-aug
tail. Our ``"flatcosine"`` keeps that flat body but adds a cosine decay over the final
``no_aug_epoch`` epochs, which usually helps the no-aug fine-tuning stage converge.
It is therefore *not* byte-for-byte parity with upstream's schedule. For an exact
upstream match, use ``scheduler="multistep"`` with a milestone beyond ``epochs`` (e.g.
``lr_milestones=[500]``), which reproduces the flat/never-stepped behaviour. See the
2026-07-15 note in ``docs/ROADMAP.md``.

The warmup wraps the epoch scheduler: while warming up, the epoch scheduler is held
back (``LinearWarmup.finished()`` gates ``lr_scheduler.step()`` in the trainer), exactly
as in upstream ``det_solver``.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, MultiStepLR

__all__ = ["LinearWarmup", "build_lr_scheduler"]


class LinearWarmup:
    """Linearly scale every param group's LR from 0 to its base over ``warmup_duration``.

    Ported from upstream ``src/optim/warmup.py``. ``step()`` is called once per
    optimizer step; once ``finished()`` the base LRs are restored and the epoch
    scheduler takes over.
    """

    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1):
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg["lr"] for pg in lr_scheduler.optimizer.param_groups]
        self.warmup_duration = warmup_duration
        self.last_step = last_step
        self.step()

    def get_warmup_factor(self, step: int) -> float:
        if self.warmup_duration <= 0:
            return 1.0
        return min(1.0, (step + 1) / self.warmup_duration)

    def step(self) -> None:
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
        factor = self.get_warmup_factor(self.last_step)
        for pg, base in zip(self.lr_scheduler.optimizer.param_groups, self.warmup_end_values):
            pg["lr"] = factor * base

    def finished(self) -> bool:
        return self.last_step >= self.warmup_duration

    def state_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "lr_scheduler"}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


def _flat_cosine(
    optimizer: Optimizer, total_epochs: int, flat_epochs: int, lr_min_ratio: float
) -> LambdaLR:
    """Multiplier = 1.0 for ``flat_epochs`` epochs, then cosine to ``lr_min_ratio``."""
    decay_epochs = max(total_epochs - flat_epochs, 1)

    def fn(epoch: int) -> float:
        if epoch < flat_epochs:
            return 1.0
        progress = min((epoch - flat_epochs) / decay_epochs, 1.0)
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=fn)


def build_lr_scheduler(optimizer: Optimizer, cfg, lr_min_ratio: float = 0.01) -> LRScheduler:
    """Build the epoch-stepped LR scheduler named by ``cfg.scheduler``."""
    kind = getattr(cfg, "scheduler", "flatcosine")
    if kind == "flatcosine":
        flat = max(cfg.epochs - cfg.no_aug_epoch, 1)
        return _flat_cosine(optimizer, cfg.epochs, flat, lr_min_ratio)
    if kind == "multistep":
        milestones = getattr(cfg, "lr_milestones", None) or [max(cfg.epochs - 1, 1)]
        gamma = getattr(cfg, "lr_gamma", 0.1)
        return MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
    raise ValueError(f"Unknown scheduler {kind!r}; expected 'flatcosine' or 'multistep'.")
