"""Training (Phase 4): the D-FINE loop, EMA, LR schedules, and progress visualization.

Everything here imports torch, so it is kept out of the base ``import dfine`` path
(exposed lazily). Install with ``pip install dfine[train]`` (torch + scipy matcher +
tensorboard). The public entry point is :meth:`dfine.DFINE.train`; the pieces are also
importable directly for custom loops.
"""

from __future__ import annotations

from .ema import ModelEMA
from .evaluator import COCO_STAT_NAMES, coco_val_fn, evaluate
from .logger import MetricLogger, SmoothedValue
from .scheduler import LinearWarmup, build_lr_scheduler
from .trainer import Trainer, build_param_groups, train_one_epoch
from .visualizer import TrainingVisualizer

__all__ = [
    "Trainer",
    "train_one_epoch",
    "build_param_groups",
    "ModelEMA",
    "MetricLogger",
    "SmoothedValue",
    "LinearWarmup",
    "build_lr_scheduler",
    "TrainingVisualizer",
    "evaluate",
    "coco_val_fn",
    "COCO_STAT_NAMES",
]
