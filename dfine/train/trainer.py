"""The D-FINE training loop (single-process, Path-A native model).

Mirrors upstream ``src/solver/det_engine.py::train_one_epoch`` +
``det_solver.py::fit`` with the registry/YAML/distributed layers removed:

* ``build_param_groups`` — the AdamW groups from ``configs/.../optimizer.yml``:
  backbone (non-norm) at ``lr_backbone``, norm/BN in the encoder/decoder with
  ``weight_decay=0``, everything else at the base ``lr``/``weight_decay``. The regex
  patterns are copied verbatim so grouping matches upstream.
* ``train_one_epoch`` — AMP autocast, ``sum(loss_dict)`` backward, grad clip, optimizer
  step, EMA update, per-iteration warmup, and the ``MetricLogger`` console progress.
* ``Trainer`` — wires model + criterion + optimizer + schedulers + EMA + the
  `TrainingVisualizer`, and runs ``.fit(train_loader, val_loader, epochs)``.

Validation/COCO-eval and the COCO dataset/augmentation live in later Phase-4 tasks;
`Trainer.fit` already accepts a ``val_fn`` hook so they slot in without changing the loop.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .ema import ModelEMA
from .logger import MetricLogger, SmoothedValue
from .scheduler import LinearWarmup, build_lr_scheduler
from .visualizer import TrainingVisualizer

__all__ = ["build_param_groups", "train_one_epoch", "Trainer"]

# Copied verbatim from D-FINE configs/dfine/include/optimizer.yml so the AdamW param
# grouping is identical to upstream. Upstream ships two schemes: L/X (and the base) put
# only `norm|bn` in the zero-weight-decay encoder/decoder group, while N/S/M also include
# `bias` — selected per size by `cfg.zero_wd_encdec_bias`.
_BACKBONE_NO_NORM = r"^(?=.*backbone)(?!.*norm).*$"
_ENC_DEC_NORM = r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$"
_ENC_DEC_NORM_BIAS = r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$"


def build_param_groups(model: nn.Module, cfg) -> list[dict]:
    """AdamW param groups: backbone LR, zero-WD norms, base for the rest.

    Each named parameter lands in exactly one group (first pattern wins), matching
    upstream ``get_optim_params``.
    """
    enc_dec_pattern = (
        _ENC_DEC_NORM_BIAS if getattr(cfg, "zero_wd_encdec_bias", False) else _ENC_DEC_NORM
    )
    groups: list[dict] = [
        {"params": [], "lr": cfg.lr_backbone, "pattern": _BACKBONE_NO_NORM},
        {"params": [], "weight_decay": 0.0, "pattern": enc_dec_pattern},
    ]
    default = {"params": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for g in groups:
            if re.findall(g["pattern"], name):
                g["params"].append(p)
                break
        else:
            default["params"].append(p)
    out = [{k: v for k, v in g.items() if k != "pattern"} for g in groups]
    out.append(default)
    return [g for g in out if g["params"]]


def _new_grad_scaler():
    """A CUDA AMP ``GradScaler`` via the modern ``torch.amp`` API where available.

    ``torch.cuda.amp.GradScaler`` is deprecated (torch >= 2.4); ``torch.amp.GradScaler``
    is the replacement but only exists on newer torch, so fall back for older versions.
    """
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:  # pragma: no cover - very old signature
            pass
    return torch.cuda.amp.GradScaler()


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """AdamW over :func:`build_param_groups` with the config's base LR/WD/betas."""
    return torch.optim.AdamW(
        build_param_groups(model, cfg),
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        weight_decay=cfg.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    epochs: int | None = None,
    max_norm: float = 0.0,
    print_freq: int = 10,
    ema: ModelEMA | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    warmup: LinearWarmup | None = None,
    visualizer: TrainingVisualizer | None = None,
) -> dict[str, float]:
    """Run one training epoch; return ``{meter: global_avg}`` (loss terms + lr)."""
    model.train()
    criterion.train()
    logger = MetricLogger(delimiter="  ")
    logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]" if epochs is None else f"Epoch: [{epoch}/{epochs}]"
    n_iter = len(data_loader) if hasattr(data_loader, "__len__") else 0

    for i, (samples, targets) in enumerate(logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * n_iter + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=n_iter)
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(samples, targets=targets)
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        if warmup is not None:
            warmup.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            terms = {k: v.item() for k, v in loss_dict.items()}
            raise RuntimeError(f"Loss is {loss_value}, stopping training. Loss terms: {terms}")

        logger.update(loss=loss_value, **{k: v.item() for k, v in loss_dict.items()})
        logger.update(lr=optimizer.param_groups[0]["lr"])
        if visualizer is not None and global_step % 10 == 0:
            visualizer.log_step(
                global_step,
                loss_value,
                [pg["lr"] for pg in optimizer.param_groups],
                {k: v.item() for k, v in loss_dict.items()},
            )

    stats = logger.global_avg_dict()
    print("Averaged stats:", logger)
    return stats


class Trainer:
    """Own the model + criterion + optimizer/schedulers + EMA + visualization, and fit.

    Built from a :class:`~dfine.config.DFINEConfig`; the criterion is created from the
    same config so the loss weights/denoising match the model. Call :meth:`fit` with a
    dataloader that yields ``(samples, targets)`` where ``samples`` is a ``float`` image
    batch ``BCHW`` and each ``target`` is ``{"labels": LongTensor[n], "boxes": Tensor[n,4]}``
    (cxcywh, normalized), matching the criterion's expectations.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: torch.device | None = None,
        output_dir: str | Path = "runs/train",
        *,
        use_ema: bool | None = None,
        use_amp: bool | None = None,
        visualize: bool = True,
        use_wandb: bool = False,
    ):
        from ..backends.native import DFINECriterion
        from .distributed import is_main_process, wrap_model_ddp

        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main = is_main_process()

        # ``module`` is the raw model (optimizer/EMA/checkpoints target it); ``model`` is
        # the possibly-DDP-wrapped module used for the forward/backward.
        self.module = model.to(self.device)
        self.criterion = DFINECriterion.from_config(cfg).to(self.device)
        self.optimizer = build_optimizer(self.module, cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, cfg)
        self.model = wrap_model_ddp(
            self.module,
            device=self.device,
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters,
        )

        self.output_dir = Path(output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        want_amp = cfg.use_amp if use_amp is None else use_amp
        self.scaler = _new_grad_scaler() if (want_amp and self.device.type == "cuda") else None
        want_ema = cfg.ema_decay > 0 if use_ema is None else use_ema
        self.ema = (
            ModelEMA(self.module, decay=cfg.ema_decay, warmups=cfg.ema_warmups)
            if want_ema
            else None
        )
        # Only rank 0 writes TensorBoard/loss-curve/W&B artifacts.
        self.visualizer = (
            TrainingVisualizer(
                self.output_dir,
                use_wandb=use_wandb,
                wandb_name=getattr(cfg, "exp_name", None),
            )
            if (visualize and self.is_main)
            else None
        )

    def fit(
        self,
        train_loader: Iterable,
        epochs: int | None = None,
        val_loader: Iterable | None = None,
        val_fn: Callable[[nn.Module, Iterable], dict[str, float]] | None = None,
    ) -> nn.Module:
        """Train for ``epochs`` (default ``cfg.epochs``); return the eval module (EMA if on).

        Under an active process group (multi-GPU) the loaders are sharded with a
        ``DistributedSampler``, only rank 0 saves/logs, and validation runs on all ranks
        (``faster-coco-eval`` gathers the shards); the returned module is de-paralleled.
        """
        from .distributed import barrier, de_parallel, wrap_loader_distributed

        epochs = epochs or self.cfg.epochs
        train_loader = wrap_loader_distributed(train_loader, shuffle=True)
        if val_loader is not None:
            val_loader = wrap_loader_distributed(val_loader, shuffle=False)
        warmup = (
            LinearWarmup(self.lr_scheduler, self.cfg.warmup_iters)
            if self.cfg.warmup_iters > 0
            else None
        )

        for epoch in range(epochs):
            # Let the sampler / multi-scale collate know the epoch (no-op for plain loaders).
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(epoch)
            stats = train_one_epoch(
                self.model,
                self.criterion,
                train_loader,
                self.optimizer,
                self.device,
                epoch,
                epochs=epochs,
                max_norm=self.cfg.clip_max_norm,
                ema=self.ema,
                scaler=self.scaler,
                warmup=warmup,
                visualizer=self.visualizer,
            )
            if warmup is None or warmup.finished():
                self.lr_scheduler.step()

            metrics = None
            if val_fn is not None and val_loader is not None:
                module = self.ema.module if self.ema else de_parallel(self.model)
                metrics = val_fn(module, val_loader)
            if self.visualizer is not None:
                self.visualizer.log_epoch(epoch, stats, metrics)
            if self.is_main:
                self.save_checkpoint(self.output_dir / "last.pth", epoch)
            barrier()

        if self.visualizer is not None:
            self.visualizer.close()
        return self.ema.module if self.ema else de_parallel(self.model)

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        from .distributed import de_parallel

        state = {
            "epoch": epoch,
            "model": de_parallel(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)
