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
        # (global_step, total_loss) points for the PNG curve.
        self._steps: list[int] = []
        self._losses: list[float] = []
        self._epoch_x: list[int] = []
        self._epoch_ap: list[float] = []

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
        """Record end-of-epoch train averages + optional validation metrics."""
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
        # "AP" is the primary COCO mAP@[.50:.95] (see evaluator.COCO_STAT_NAMES).
        if metrics and "AP" in metrics:
            self._epoch_x.append(epoch)
            self._epoch_ap.append(metrics["AP"])
        if self.plot:
            self._draw()

    def _draw(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.plot = False  # only warn once, then stop trying
            print("[visualizer] matplotlib not installed — skipping loss_curve.png.")
            return
        if not self._steps:
            return
        has_ap = bool(self._epoch_x)
        nrows = 2 if has_ap else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 3.0 * nrows), squeeze=False)
        loss_ax = axes[0][0]
        loss_ax.plot(self._steps, self._losses, color="tab:blue", lw=1.2)
        loss_ax.set_xlabel("optimizer step")
        loss_ax.set_ylabel("total loss")
        loss_ax.set_title("train loss")
        loss_ax.grid(alpha=0.3)
        if has_ap:
            ap_ax = axes[1][0]
            ap_ax.plot(self._epoch_x, self._epoch_ap, "o-", color="tab:red", lw=1.2)
            ap_ax.set_xlabel("epoch")
            ap_ax.set_ylabel("val AP50:95")
            ap_ax.set_title("validation mAP")
            ap_ax.grid(alpha=0.3)
        fig.suptitle("D-FINE training progress")
        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curve.png", dpi=110)
        plt.close(fig)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
