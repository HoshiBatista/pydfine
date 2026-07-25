# Validation & analytics

Score a trained detector with the standard COCO metrics, and — optionally — render the
diagnostic plots you look at *after* the numbers to understand **where** the model fails.

- [`DFINE.val`](#running-validation) — the one-call entry point (COCO metrics, optional plots).
- [Analytics artifacts](#analytics-artifacts) — confusion matrix, P/R/F1-vs-confidence curves,
  per-class AP, and a worst-predictions gallery.
- [API reference](#api-reference) — the underlying `evaluate` / metric classes.

---

## Running validation

```python
from dfine import DFINE

model = DFINE.from_pretrained("dfine-l")

# A COCO dataset root: val/ + annotations/instances_val.json (a stock MS-COCO
# val2017/ layout is auto-detected too).
metrics = model.val(data="coco/")
print(metrics["AP"])        # primary mAP@[.50:.95]
print(metrics["AP50"])      # mAP@0.50
```

`val` returns the **12 standard COCO metrics** keyed by name (`COCO_STAT_NAMES`):

| Key | Meaning |
| --- | --- |
| `AP` | mAP averaged over IoU 0.50–0.95 (**the** headline number) |
| `AP50`, `AP75` | mAP at IoU 0.50 / 0.75 |
| `AP_small`, `AP_medium`, `AP_large` | mAP by GT object area |
| `AR_1`, `AR_10`, `AR_100` | average recall given 1 / 10 / 100 detections per image |
| `AR_small`, `AR_medium`, `AR_large` | average recall by object area |

### Label spaces must line up

The single most common validation footgun: predicted class ids must match the
`category_id` values in your ground-truth annotations.

!!! warning "Stock MS-COCO uses **sparse** ids (1..90)"
    pydfine predicts **contiguous** `0..N-1` labels by default. To score against stock
    MS-COCO ground truth, build the model with `remap_mscoco_category=True` so the
    postprocessor emits the sparse ids the annotations use:

    ```python
    model = DFINE.from_pretrained("dfine-l", remap_mscoco_category=True)
    metrics = model.val(data="coco/", remap_mscoco_category=True)
    ```

    Datasets produced by [`yolo_to_coco`](data.md) are already **0-indexed contiguous**, so
    the default `remap_mscoco_category=False` is correct for them.

### Bring your own loader

```python
from dfine.train.dataset import build_coco_val_dataloader

loader = build_coco_val_dataloader("coco/", cfg=model.config, batch_size=8)
metrics = model.val(val_loader=loader)   # loader.dataset must carry the GT `.coco`
```

### From the CLI

```bash
dfine val dfine-l --data coco/
dfine val dfine-l --data coco/ --plots --output-dir runs/val   # + analytics plots
```

---

## Analytics artifacts

Pass `plots=True` to also write a diagnostic bundle under `output_dir` (default
`runs/val`). This needs `matplotlib` (the `[train]` extra) and assumes **contiguous**
labels (`remap_mscoco_category=False`).

```python
metrics = model.val(data="coco/", plots=True, output_dir="runs/val")
```

```
runs/val/
├── confusion_matrix.png    # predicted-vs-true grid (which classes get confused)
├── pr_curve.png            # per-class precision–recall at IoU 0.50
├── f1_curve.png            # F1 vs confidence  (its peak → best operating point)
├── p_curve.png             # precision vs confidence
├── r_curve.png             # recall vs confidence
└── worst/                  # highest-error frames, GT green / pred red
    ├── 00_err7_000042.jpg
    └── ...
```

The per-class AP table and the recommended confidence are also **logged**:

```
per-class AP  person 0.72  car 0.65  dog 0.51  ...
Best confidence: 0.31  (mean F1 0.68)
Analytics saved to runs/val
```

!!! info "Consistent IoU matching (0.5)"
    All three custom accumulators — [`ConfusionMatrix`][dfine.train.metrics.ConfusionMatrix],
    [`PRCurveMetrics`][dfine.train.metrics.PRCurveMetrics], and
    [`WorstPredictions`][dfine.train.metrics.WorstPredictions] — match a detection to a
    ground-truth box at **IoU ≥ 0.5** (a common single-threshold TP definition). This is a
    coarser, more interpretable view than COCO's IoU 0.50–0.95 sweep, and it is deliberately
    the **same** across every plot so the artifacts agree with each other. The numeric COCO
    metrics above are independent — they come from the COCO evaluator, not these classes.

### Reading each plot

| Plot | What it answers | How to read it |
| --- | --- | --- |
| **Confusion matrix** | *Which classes does the model mix up?* | Column-normalized (per true class). Bright off-diagonal cells = systematic confusion; the last row/column is **background** (false negatives / false positives). |
| **F1–confidence** | *What `conf` should I deploy at?* | The black mean curve peaks at the best trade-off — that x-value is `Best confidence`. |
| **P–confidence** | *How clean are detections as I raise the bar?* | Precision rises with confidence; find where it saturates. |
| **R–confidence** | *How much do I miss as I raise the bar?* | Recall falls with confidence; the crossover with precision is the F1 peak. |
| **PR curve** | *Per-class quality at IoU 0.50* | Area under each curve ≈ that class's AP50; the mean line is mAP@.5. |
| **`worst/` gallery** | *Where are the failures / label errors?* | Frames with the most FP+FN, **GT in green, predictions in red** — spot mislabeled data and hard cases fast. |

### Pick a deployment confidence

The F1-vs-confidence sweep gives a data-driven answer to "what `conf` do I pass to
`predict`?" — no eyeballing:

```python
from dfine.train.dataset import build_coco_val_dataloader
from dfine.train.evaluator import evaluate
from dfine.train.metrics import PRCurveMetrics

loader = build_coco_val_dataloader("coco/", cfg=model.config)

# evaluate(plots=True) logs "Best confidence: X" for you; to compute it directly:
prm = PRCurveMetrics(num_classes=model.config.num_classes)   # matches at IoU 0.5
# ... feed prm.process_batch(det_boxes, det_scores, det_classes, gt_boxes, gt_classes) ...
best_conf, mean_f1 = prm.best_confidence()
results = model.predict("street.jpg", conf=best_conf)
```

### Validate every epoch during training

`DFINE.train` auto-wires COCO validation whenever a val loader is present (built from
`data=`, or passed explicitly) — the `AP` curve is logged and streamed to TensorBoard, and
`best.pth` is saved on each improvement:

```python
model.train(data="coco/", epochs=72)          # val/ split validated each epoch
```

To customize, pass your own hook — see [`coco_val_fn`][dfine.train.evaluator.coco_val_fn]:

```python
from dfine.train.evaluator import coco_val_fn

val_fn = coco_val_fn(model.postprocessor, model.device)
# trainer.fit(train_loader, val_loader=loader, val_fn=val_fn)
```

---

## API reference

### DFINE.val

::: dfine.model.DFINE.val
    options:
      show_root_heading: true
      show_root_full_path: false

### evaluate

::: dfine.train.evaluator.evaluate

::: dfine.train.evaluator.coco_val_fn

### Metrics

::: dfine.train.metrics.ConfusionMatrix

::: dfine.train.metrics.PRCurveMetrics

::: dfine.train.metrics.WorstPredictions

::: dfine.train.metrics.per_class_ap

::: dfine.train.metrics.box_iou
