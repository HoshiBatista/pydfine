"""Validation analytics — confusion matrix, PR curves, per-class AP.

Diagnostics you look at *after* the COCO summary to understand a trained detector:

* :class:`ConfusionMatrix` — an ``(nc+1) × (nc+1)`` predicted-vs-true count grid (last
  row/col = background: false negatives / false positives), matched by IoU. numpy-only
  core; ``.plot`` renders a heatmap.
* :func:`per_class_ap` / :func:`plot_pr_curve` — read the COCO evaluator's precision
  tensor (``coco_eval.eval['precision']``, shape ``[T,R,K,A,M]``) for per-class AP and
  precision–recall curves — no second inference pass.
* :func:`save_val_analytics` — ties them together: write ``confusion_matrix.png`` +
  ``pr_curve.png`` and log the per-class AP table.

matplotlib (the ``[train]`` extra) is imported lazily with the headless ``Agg`` backend,
so importing this module never needs a display. The confusion matrix assumes **contiguous**
class labels (``remap_mscoco_category=False``); out-of-range predictions are ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..log import LOGGER, colorstr, metrics_line, rule

__all__ = ["ConfusionMatrix", "box_iou", "per_class_ap", "plot_pr_curve", "save_val_analytics"]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of ``xyxy`` boxes → ``[len(a), len(b)]`` (0 where either is empty)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class ConfusionMatrix:
    """Predicted-vs-true detection counts, matched by IoU (ultralytics-style).

    ``matrix[p, t]`` counts predictions of class ``p`` matched to a true box of class ``t``;
    index ``nc`` (the extra last row/col) is **background** — ``matrix[nc, t]`` are missed
    GT (false negatives), ``matrix[p, nc]`` are spurious detections (false positives). Feed
    one image at a time with :meth:`process_batch`.
    """

    def __init__(self, num_classes: int, conf: float = 0.25, iou_thresh: float = 0.45):
        self.nc = int(num_classes)
        self.conf = conf
        self.iou_thresh = iou_thresh
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

    def process_batch(
        self,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        """Accumulate one image's detections (``xyxy``/score/class) against its GT."""
        bg = self.nc
        det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        det_scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
        det_classes = np.asarray(det_classes).reshape(-1).astype(int)
        gt_classes = np.asarray(gt_classes).reshape(-1).astype(int)

        keep = (det_scores >= self.conf) & (det_classes >= 0) & (det_classes < self.nc)
        db, dc = det_boxes[keep], det_classes[keep]
        gmask = (gt_classes >= 0) & (gt_classes < self.nc)
        gb, gc = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)[gmask], gt_classes[gmask]

        if len(gc) == 0:  # every kept detection is a false positive
            for c in dc:
                self.matrix[c, bg] += 1
            return
        if len(db) == 0:  # every GT is a miss
            for c in gc:
                self.matrix[bg, c] += 1
            return

        iou = box_iou(db, gb)
        matched_det: set[int] = set()
        matched_gt: set[int] = set()
        # Greedy highest-IoU-first assignment, each det/GT used once (above the threshold).
        pairs = [
            (iou[i, j], i, j)
            for i in range(len(db))
            for j in range(len(gb))
            if iou[i, j] >= self.iou_thresh
        ]
        for _, i, j in sorted(pairs, key=lambda x: x[0], reverse=True):
            if i in matched_det or j in matched_gt:
                continue
            matched_det.add(i)
            matched_gt.add(j)
            self.matrix[dc[i], gc[j]] += 1
        for i in range(len(db)):
            if i not in matched_det:
                self.matrix[dc[i], bg] += 1  # false positive
        for j in range(len(gb)):
            if j not in matched_gt:
                self.matrix[bg, gc[j]] += 1  # false negative (missed GT)

    def plot(self, save_path: str | Path, names: dict[int, str] | None = None) -> Path:
        """Save a column-normalized heatmap (per-true-class recall view) to ``save_path``."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [(names.get(i, str(i)) if names else str(i)) for i in range(self.nc)]
        labels = labels + ["background"]
        col_sums = self.matrix.sum(axis=0, keepdims=True)
        norm = np.divide(
            self.matrix, col_sums, out=np.zeros_like(self.matrix, float), where=col_sums > 0
        )

        fig, ax = plt.subplots(figsize=(max(6, self.nc * 0.6),) * 2)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_xticks(range(self.nc + 1), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(self.nc + 1), labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Confusion matrix (normalized)")
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path


def per_class_ap(coco_eval, names: dict[int, str] | None = None) -> dict[str, float]:
    """Per-class AP@[.50:.95] from a COCO evaluator's precision tensor (area=all, maxDet=100)."""
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    cat_ids = list(coco_eval.params.catIds)
    out: dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        pr = precision[:, :, k, 0, -1]
        valid = pr[pr > -1]
        ap = float(valid.mean()) if valid.size else float("nan")
        out[names.get(cat_id, str(cat_id)) if names else str(cat_id)] = ap
    return out


def plot_pr_curve(coco_eval, save_path: str | Path, names: dict[int, str] | None = None) -> Path:
    """Save per-class precision–recall curves at IoU=0.50 (+ their mean) to ``save_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    rec = coco_eval.params.recThrs  # [R] recall thresholds 0..1
    cat_ids = list(coco_eval.params.catIds)

    fig, ax = plt.subplots(figsize=(8, 6))
    per_class = precision[0, :, :, 0, -1]  # IoU=0.5 → [R, K]
    for k, cat_id in enumerate(cat_ids):
        pr = per_class[:, k]
        label = names.get(cat_id, str(cat_id)) if names else str(cat_id)
        ax.plot(rec, np.where(pr < 0, 0, pr), linewidth=1, label=label)
    mean = np.where(per_class < 0, 0, per_class).mean(axis=1)
    ax.plot(rec, mean, color="black", linewidth=3, label=f"all (mAP@.5 {mean.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision–Recall (IoU=0.50)")
    if len(cat_ids) <= 20:
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_val_analytics(coco_eval, cm: ConfusionMatrix, output_dir, names=None) -> dict[str, str]:
    """Write ``confusion_matrix.png`` + ``pr_curve.png`` and log the per-class AP table.

    Returns the written paths keyed by artifact name. matplotlib failures degrade to a
    warning (the numeric metrics from :func:`evaluate` are unaffected).
    """
    output_dir = Path(output_dir)
    artifacts: dict[str, str] = {}

    aps = per_class_ap(coco_eval, names)
    LOGGER.info(f"{rule('per-class AP', 'cyan')}  {metrics_line(aps)}")

    try:
        artifacts["confusion_matrix"] = str(cm.plot(output_dir / "confusion_matrix.png", names))
        artifacts["pr_curve"] = str(plot_pr_curve(coco_eval, output_dir / "pr_curve.png", names))
        LOGGER.info(colorstr("green", "bold", f"Analytics saved to {output_dir}"))
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        LOGGER.warning(f"could not render val plots: {exc}")
    return artifacts
