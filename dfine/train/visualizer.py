"""Training-process visualization — the same signals D-FINE surfaces while training.

Upstream logs to a TensorBoard ``SummaryWriter`` (``Loss/total``, ``Loss/<term>``,
``Lr/pg_<i>``, ``Test/<metric>``) and optionally to Weights & Biases. This wraps both
behind one object plus a self-contained matplotlib loss-curve PNG, so a bare
``pip install pydfine[train]`` already draws the training progress with no extra setup.

Every backend is optional and degrades gracefully: missing ``tensorboard`` / ``wandb`` /
``matplotlib`` just disables that output with a one-time note rather than failing the run.
The console progress line (`MetricLogger`) is always on and lives in ``logger.py``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TrainingVisualizer"]

# Primary validation metric per task, highest priority first — the one drawn on the
# progress curve's second panel. Mirrors the Trainer's best-checkpoint metric selection
# (detection AP / instance-seg mask AP / semantic-seg mIoU); keep the two in sync.
_PRIMARY_METRIC_KEYS = ("AP", "mAP_50_95_mask", "mIoU")
_METRIC_LABELS = {
    "AP": ("val AP@[.50:.95]", "validation mAP"),
    "mAP_50_95_mask": ("val mask AP@[.50:.95]", "validation mask mAP"),
    "mIoU": ("val mIoU", "validation mIoU"),
}


class TrainingVisualizer:
    """Fan training scalars out to TensorBoard, a loss-curve PNG, and optional wandb.

    Args:
        output_dir: where TensorBoard events + ``loss_curve.png`` are written.
        use_tensorboard: enable the ``SummaryWriter`` (if ``tensorboard`` is installed).
        use_wandb: mirror scalars to Weights & Biases (if ``wandb`` is installed/logged in).
        wandb_project / wandb_name: passed to ``wandb.init`` when ``use_wandb``.
        plot: draw/update ``loss_curve.png`` after each epoch (if ``matplotlib`` is installed).
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "dfine",
        wandb_name: str | None = None,
        plot: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot = plot
        self._steps: list[int] = []
        self._losses: list[float] = []
        self._epoch_x: list[int] = []
        self._epoch_metric: list[float] = []
        self._metric_key: str | None = None  # locked to the task's primary metric on first val

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError:
                print(
                    "[visualizer] tensorboard not installed — skipping TB logs "
                    "(pip install pydfine[train])."
                )

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                wandb.init(project=wandb_project, name=wandb_name)
                self.wandb = wandb
            except ImportError:
                print("[visualizer] wandb not installed — skipping W&B logs.")

    @property
    def tb_logdir(self) -> Path | None:
        """The TensorBoard event directory, or ``None`` when TB logging is off/unavailable."""
        return self.output_dir / "tb" if self.writer is not None else None

    def log_step(
        self, global_step: int, total_loss: float, lrs: list[float], loss_dict: dict[str, float]
    ) -> None:
        """Record one optimizer step (called every ~10 iters by the trainer)."""
        self._steps.append(global_step)
        self._losses.append(total_loss)
        if self.writer is not None:
            self.writer.add_scalar("Loss/total", total_loss, global_step)
            for i, lr in enumerate(lrs):
                self.writer.add_scalar(f"Lr/pg_{i}", lr, global_step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f"Loss/{k}", v, global_step)
        if self.wandb is not None:
            self.wandb.log(
                {
                    "Loss/total": total_loss,
                    "lr": lrs[0],
                    "step": global_step,
                    **{f"Loss/{k}": v for k, v in loss_dict.items()},
                }
            )

    def log_epoch(
        self, epoch: int, train_stats: dict[str, float], metrics: dict[str, float] | None = None
    ) -> None:
        """Record end-of-epoch train averages + optional validation metrics.

        The progress curve's second panel tracks the task's primary validation metric —
        detection ``AP``, instance-seg ``mAP_50_95_mask``, or sem_seg ``mIoU`` — locked to
        whichever appears first so a run always plots the right score, not just detection AP.
        """
        if self.writer is not None:
            for k, v in train_stats.items():
                self.writer.add_scalar(f"Epoch/train_{k}", v, epoch)
            for k, v in (metrics or {}).items():
                self.writer.add_scalar(f"Test/{k}", v, epoch)
        if self.wandb is not None:
            payload = {f"train/{k}": v for k, v in train_stats.items()}
            payload.update({f"metrics/{k}": v for k, v in (metrics or {}).items()})
            payload["epoch"] = epoch
            self.wandb.log(payload)
        key = self._primary_metric_key(metrics)
        if key is not None:
            self._metric_key = self._metric_key or key
            if key == self._metric_key:  # ignore a stray other-task key mid-run
                self._epoch_x.append(epoch)
                self._epoch_metric.append(metrics[key])
        if self.plot:
            self._draw()

    @staticmethod
    def _primary_metric_key(metrics: dict[str, float] | None) -> str | None:
        """The task's primary validation metric present in ``metrics`` (by priority), else None."""
        if not metrics:
            return None
        return next((k for k in _PRIMARY_METRIC_KEYS if k in metrics), None)

    def _draw(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.plot = False
            print("[visualizer] matplotlib not installed — skipping loss_curve.png.")
            return
        if not self._steps:
            return
        has_metric = bool(self._epoch_x)
        nrows = 2 if has_metric else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 3.0 * nrows), squeeze=False)
        loss_ax = axes[0][0]
        loss_ax.plot(self._steps, self._losses, color="tab:blue", lw=1.2)
        loss_ax.set_xlabel("optimizer step")
        loss_ax.set_ylabel("total loss")
        loss_ax.set_title("train loss")
        loss_ax.grid(alpha=0.3)
        if has_metric:
            ylabel, title = _METRIC_LABELS.get(
                self._metric_key, ("val metric", "validation metric")
            )
            metric_ax = axes[1][0]
            metric_ax.plot(self._epoch_x, self._epoch_metric, "o-", color="tab:red", lw=1.2)
            metric_ax.set_xlabel("epoch")
            metric_ax.set_ylabel(ylabel)
            metric_ax.set_title(title)
            metric_ax.grid(alpha=0.3)
        fig.suptitle("D-FINE training progress")
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=110)
        plt.close(fig)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
