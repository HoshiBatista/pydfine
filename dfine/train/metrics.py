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

__all__ = [
    "ConfusionMatrix",
    "PRCurveMetrics",
    "WorstPredictions",
    "box_iou",
    "per_class_ap",
    "plot_pr_curve",
    "save_val_analytics",
]


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

    def __init__(self, num_classes: int, conf: float = 0.25, iou_thresh: float = 0.5):
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


class PRCurveMetrics:
    """Accumulate ``(score, is_TP)`` per class to build P/R/F1-vs-confidence curves.

    Each detection is matched (per class, highest-score-first) to a same-class GT box at
    ``iou_thresh`` — matched → true positive, else false positive. After the val pass,
    :meth:`curves` sweeps the confidence axis to give per-class + mean precision, recall and
    F1, and :meth:`best_confidence` returns the threshold that maximizes mean F1 — the
    practical "what ``conf`` should I use?" answer.
    """

    def __init__(self, num_classes: int, iou_thresh: float = 0.5):
        self.nc = int(num_classes)
        self.iou_thresh = iou_thresh
        self._tp: list[bool] = []
        self._conf: list[float] = []
        self._cls: list[int] = []
        self.n_gt = np.zeros(self.nc, dtype=np.int64)

    def process_batch(self, det_boxes, det_scores, det_classes, gt_boxes, gt_classes) -> None:
        """Match one image's detections to its GT (per class) and record ``(score, TP)``."""
        dc = np.asarray(det_classes).reshape(-1).astype(int)
        ds = np.asarray(det_scores, dtype=np.float64).reshape(-1)
        db = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        gc = np.asarray(gt_classes).reshape(-1).astype(int)
        gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)

        for c in gc:
            if 0 <= c < self.nc:
                self.n_gt[c] += 1
        if len(db) == 0:
            return

        valid = (dc >= 0) & (dc < self.nc)
        db, ds, dc = db[valid], ds[valid], dc[valid]
        iou = box_iou(db, gb) if len(gb) else np.zeros((len(db), 0))
        matched: set[int] = set()
        for idx in np.argsort(-ds):
            c = int(dc[idx])
            tp = False
            if len(gb):
                cand = [
                    j
                    for j in range(len(gb))
                    if gc[j] == c and j not in matched and iou[idx, j] >= self.iou_thresh
                ]
                if cand:
                    matched.add(max(cand, key=lambda j: iou[idx, j]))
                    tp = True
            self._tp.append(tp)
            self._conf.append(float(ds[idx]))
            self._cls.append(c)

    def curves(self, n: int = 1000):
        """Return ``(x, precision, recall, f1, classes)`` over an ``n``-point confidence grid.

        ``x`` is the confidence axis ``[n]``; ``precision``/``recall``/``f1`` are
        ``[len(classes), n]`` for the classes that have any GT.
        """
        eps = 1e-16
        x = np.linspace(0, 1, n)
        tp = np.asarray(self._tp, dtype=bool)
        conf = np.asarray(self._conf, dtype=np.float64)
        cls = np.asarray(self._cls, dtype=int)
        classes = [c for c in range(self.nc) if self.n_gt[c] > 0]

        p = np.ones((len(classes), n))
        r = np.zeros((len(classes), n))
        for ci, c in enumerate(classes):
            m = cls == c
            if not m.any():
                continue
            order = np.argsort(-conf[m])
            csorted = conf[m][order]
            tpc = tp[m][order].cumsum()
            fpc = (~tp[m][order]).cumsum()
            recall = tpc / (self.n_gt[c] + eps)
            precision = tpc / (tpc + fpc + eps)
            # curves are functions of the confidence threshold (conf sorted descending)
            r[ci] = np.interp(-x, -csorted, recall, left=0)
            p[ci] = np.interp(-x, -csorted, precision, left=1)
        f1 = 2 * p * r / (p + r + eps)
        return x, p, r, f1, classes

    def best_confidence(self) -> tuple[float, float]:
        """``(conf, mean_F1)`` at the confidence maximizing mean F1 (``(0.0, 0.0)`` if empty)."""
        x, _p, _r, f1, classes = self.curves()
        if not classes:
            return 0.0, 0.0
        mean_f1 = f1.mean(axis=0)
        i = int(mean_f1.argmax())
        return float(x[i]), float(mean_f1[i])


class WorstPredictions:
    """Keep the ``top_k`` images with the most errors (FP + FN) for visual review.

    ``add`` counts a frame's mistakes (class-aware IoU matching at ``iou_thresh``, detections
    filtered at ``conf``): unmatched GT = false negatives, unmatched detections = false
    positives, a wrong-class match counts as both. Only a bounded top-``k`` heap is kept, so
    memory stays flat over the val set. ``save`` renders each keeper — **green = ground truth,
    red = prediction** — to ``output_dir/worst/`` for spotting label errors and failure modes.
    """

    def __init__(self, conf: float = 0.25, iou_thresh: float = 0.5, top_k: int = 16):
        self.conf = conf
        self.iou_thresh = iou_thresh
        self.top_k = top_k
        self._heap: list = []  # min-heap of (n_errors, seq, record)
        self._seq = 0

    def _error_count(self, db, dc, gb, gc) -> tuple[int, int]:
        """``(false_positives, false_negatives)`` from class-aware greedy IoU matching."""
        if len(gb) == 0:
            return len(db), 0
        if len(db) == 0:
            return 0, len(gb)
        iou = box_iou(db, gb)
        pairs = [
            (iou[i, j], i, j)
            for i in range(len(db))
            for j in range(len(gb))
            if dc[i] == gc[j] and iou[i, j] >= self.iou_thresh
        ]
        md: set[int] = set()
        mg: set[int] = set()
        for _, i, j in sorted(pairs, key=lambda x: x[0], reverse=True):
            if i in md or j in mg:
                continue
            md.add(i)
            mg.add(j)
        return len(db) - len(md), len(gb) - len(mg)

    def add(self, image_path, det_boxes, det_scores, det_classes, gt_boxes, gt_classes) -> None:
        """Consider one image; keep it only if it is among the ``top_k`` worst so far."""
        import heapq

        if not image_path:
            return
        db = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        ds = np.asarray(det_scores, dtype=np.float64).reshape(-1)
        dc = np.asarray(det_classes).reshape(-1).astype(int)
        keep = ds >= self.conf
        db, ds, dc = db[keep], ds[keep], dc[keep]
        gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gc = np.asarray(gt_classes).reshape(-1).astype(int)

        fp, fn = self._error_count(db, dc, gb, gc)
        n_err = fp + fn
        if n_err == 0:
            return
        self._seq += 1
        record = (str(image_path), n_err, fp, fn, db, ds, dc, gb)
        item = (n_err, self._seq, record)
        if len(self._heap) < self.top_k:
            heapq.heappush(self._heap, item)
        elif n_err > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def save(self, output_dir, names=None) -> list[str]:
        """Render the kept worst frames (GT green, prediction red) to ``output_dir/worst/``."""
        from PIL import Image, ImageDraw

        worst_dir = Path(output_dir) / "worst"
        records = [rec for _, _, rec in sorted(self._heap, key=lambda x: x[0], reverse=True)]
        if not records:
            return []
        worst_dir.mkdir(parents=True, exist_ok=True)

        out: list[str] = []
        for rank, (path, n_err, fp, fn, db, ds, dc, gb) in enumerate(records):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:  # pragma: no cover - unreadable source is skipped
                continue
            draw = ImageDraw.Draw(img)
            for box in gb:
                draw.rectangle([float(v) for v in box], outline=(0, 200, 0), width=2)
            for box, s, c in zip(db, ds, dc):
                name = names.get(int(c), str(int(c))) if names else str(int(c))
                draw.rectangle([float(v) for v in box], outline=(255, 40, 40), width=2)
                draw.text(
                    (float(box[0]) + 2, float(box[1]) + 2), f"{name} {s:.2f}", fill=(255, 40, 40)
                )
            draw.text((4, 4), f"FP={fp}  FN={fn}   GT=green pred=red", fill=(255, 255, 0))
            dst = worst_dir / f"{rank:02d}_err{n_err}_{Path(path).stem}.jpg"
            img.save(dst)
            out.append(str(dst))
        return out


def _plot_conf_curve(x, ys, classes, save_path, ylabel, names=None):
    """Plot per-class ``ys`` (``[K, n]``) + their mean against the confidence axis ``x``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for row, c in zip(ys, classes):
        label = names.get(c, str(c)) if names else str(c)
        ax.plot(x, row, linewidth=1, label=label)
    mean = ys.mean(axis=0) if len(ys) else np.zeros_like(x)
    peak = f"  (max {mean.max():.2f} @ {x[mean.argmax()]:.3f})" if len(ys) else ""
    ax.plot(x, mean, color="black", linewidth=3, label=f"mean{peak}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{ylabel}–Confidence")
    if len(classes) <= 20:
        ax.legend(fontsize=7, loc="lower center")
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


def save_val_analytics(
    coco_eval,
    cm: ConfusionMatrix,
    output_dir,
    names=None,
    prm: PRCurveMetrics | None = None,
    worst: WorstPredictions | None = None,
) -> dict[str, str]:
    """Write the analytics artifacts and log the per-class AP table + best confidence.

    Produces ``confusion_matrix.png`` + ``pr_curve.png``, and — when given — the
    ``f1_curve.png``/``p_curve.png``/``r_curve.png`` conf sweep (from ``prm``, plus a logged
    "best confidence") and a ``worst/`` gallery of the highest-error frames (from ``worst``).
    Returns the written paths keyed by artifact name; matplotlib failures degrade to a
    warning (numeric metrics are unaffected).
    """
    output_dir = Path(output_dir)
    artifacts: dict[str, str] = {}

    aps = per_class_ap(coco_eval, names)
    LOGGER.info(f"{rule('per-class AP', 'cyan')}  {metrics_line(aps)}")
    if prm is not None:
        conf, f1 = prm.best_confidence()
        LOGGER.info(colorstr("cyan", "bold", f"Best confidence: {conf:.3f}  (mean F1 {f1:.3f})"))
    if worst is not None:
        saved = worst.save(output_dir, names)
        if saved:
            artifacts["worst"] = str(Path(output_dir) / "worst")
            LOGGER.info(colorstr("cyan", "bold", f"Worst {len(saved)} frames → {output_dir}/worst"))

    try:
        artifacts["confusion_matrix"] = str(cm.plot(output_dir / "confusion_matrix.png", names))
        artifacts["pr_curve"] = str(plot_pr_curve(coco_eval, output_dir / "pr_curve.png", names))
        if prm is not None:
            x, p, r, f1, classes = prm.curves()
            artifacts["f1_curve"] = str(
                _plot_conf_curve(x, f1, classes, output_dir / "f1_curve.png", "F1", names)
            )
            artifacts["p_curve"] = str(
                _plot_conf_curve(x, p, classes, output_dir / "p_curve.png", "Precision", names)
            )
            artifacts["r_curve"] = str(
                _plot_conf_curve(x, r, classes, output_dir / "r_curve.png", "Recall", names)
            )
        LOGGER.info(colorstr("green", "bold", f"Analytics saved to {output_dir}"))
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        LOGGER.warning(f"could not render val plots: {exc}")
    return artifacts
