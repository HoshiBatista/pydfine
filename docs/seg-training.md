# Segmentation training

Train **instance** (`task="segment"`) or **semantic** (`task="sem_seg"`) segmentation from a
single YOLO-style dataset root, through the same `DFINE.train(...)` façade as detection. The
losses, matcher mask costs, and validation metrics are all config-driven — no YAML.

```bash
pip install pydfine[train]        # torch + matcher (scipy) + cv2 + torchmetrics (mask AP)
```

## Dataset layout

A seg dataset is an `images/` folder plus a parallel `labels/` folder (the label is resolved
per image by swapping `/images/` → `/labels/`). Train/val is split from `images/{train,val}`
subdirs if present, otherwise from a deterministic `val_split` fraction of the flat root.

**Instance (`segment`)** — `labels/<stem>.txt`, one object per line in **normalized**
coordinates:

```
cls x1 y1 x2 y2 … xN yN     # a polygon (≥3 points) → rasterized to an instance mask
cls xc yc w h               # a plain box (5 cols) → box only, empty mask
```

```
dataset/
  images/  a.jpg  b.jpg  …
  labels/  a.txt  b.txt  …          # YOLO-Seg polygons
```

**Semantic (`sem_seg`)** — `labels/<stem>.png`, a single-channel `uint8` label map where the
pixel value is the class id; `ignore_index` pixels (default `255`) are excluded from loss and
metrics:

```
dataset/
  images/  a.jpg  b.jpg  …
  labels/  a.png  b.png  …          # class-id maps (contiguous 0..num_classes-1)
```

**Explicit train/val split** — drop images and labels into `train/` and `val/` subdirs and the
split is used verbatim (any `val_split` is ignored):

```
dataset/
  images/  train/ …   val/ …
  labels/  train/ …   val/ …
```

## Instance segmentation

Fine-tune from the released D-FINE-seg weights (recommended — the mask head starts trained),
or build fresh with your own class count.

```python
from dfine import DFINE

# Fine-tune the pretrained instance-seg model (needs the pydfine[hf] extra for the weights).
model = DFINE.from_pretrained("dfine-seg-l")        # dfine-seg-{n,s,m,l,x}
model.train(data="dataset/", epochs=50, batch_size=8)

# …or from scratch for a custom class count:
model = DFINE(size="l", task="segment", num_classes=10)
model.train(data="dataset/", epochs=100)
```

Each epoch the val split is scored with **COCO mask AP** — `mAP_50_95_mask`, `mAP_50_mask`,
`mAP_75_mask` — and the best `mAP_50_95_mask` checkpoint is saved to `runs/train/best.pth`.

## Semantic segmentation

```python
from dfine import DFINE

model = DFINE(size="l", task="sem_seg", num_classes=19)   # e.g. Cityscapes
model.train(data="dataset/", epochs=100, batch_size=8)
```

`sem_seg` initializes the shared mask fuser from the instance-seg checkpoint when available
(the neck/classifier train from scratch). Each epoch the val split is scored with **mIoU** and
`pixel_acc`; the best `mIoU` checkpoint is saved.

## Train/val split & evaluation

`DFINE.train` auto-builds the val loader from the same `data=` root and picks the right metric
by task — you don't pass a `val_loader`:

```python
model.train(data="dataset/", epochs=50, val_split=0.2)   # 20% held out for val (default)
model.train(data="dataset/", epochs=50, val_split=0.0)   # train on everything, no eval
```

The split is **deterministic** (seeded), so runs are reproducible. `images/{train,val}` subdirs
always win over `val_split`.

## Bringing your own dataloaders

For full control, build the loaders yourself and pass them in. `build_seg_dataloaders` returns a
`(train, val)` pair; `cfg=model.config` inherits `task` / `num_classes` / `imgsz` /
`sem_seg_ignore_index`:

```python
from dfine.train.seg_dataset import build_seg_dataloaders

train_loader, val_loader = build_seg_dataloaders(
    "dataset/", cfg=model.config, batch_size=8, val_split=0.2
)
model.train(train_loader=train_loader, val_loader=val_loader, epochs=50)
```

Each batch is `(images, targets)` — `images` is `[B, 3, imgsz, imgsz]`; each target carries
`boxes` + `labels` + `masks` `[N, imgsz, imgsz]` (segment) or `sem_mask` `[imgsz, imgsz]`
(sem_seg). Any dataloader yielding that shape works.

## Loss & matcher knobs

All defaults follow D-FINE-seg and live on `DFINEConfig` — override them at construction
(`DFINE(size="l", task="segment", loss_mask_dice_weight=2.0)`):

| Task | Parameter | Default | Role |
|------|-----------|---------|------|
| segment | `loss_mask_bce_weight` | `1.0` | box-cropped BCE on the instance masks |
| segment | `loss_mask_dice_weight` | `1.0` | box-cropped Dice on the instance masks |
| segment | `cost_mask` / `cost_mask_dice` | `1.0` / `1.0` | mask focal / Dice cost in the matcher |
| sem_seg | `loss_ce_weight` | `1.0` | dense pixel cross-entropy |
| sem_seg | `loss_dice_weight` | `1.0` | multi-class soft Dice |
| sem_seg | `loss_aux_weight` | `0.4` | auxiliary deep-supervision CE |
| sem_seg | `sem_seg_ignore_index` | `255` | label id excluded from loss + mIoU |

Detection stays byte-identical: mask losses/costs only activate for `task="segment"`, and the
sem_seg criterion is only used for `task="sem_seg"`. Optimizer, LR schedule, EMA, and AMP are the
same as the [detection trainer](api/model.md). See the [config reference](CONFIG_REFERENCE.md)
for the full parameter list.

## Export the trained model

The trained model exports to a task-aware ONNX graph (masks for `segment`, a fused-argmax label
map for `sem_seg`) — see the [Export guide](api/export.md):

```python
model.export(imgsz=640)     # segment → (labels, boxes, scores, masks); sem_seg → sem_seg map
```
