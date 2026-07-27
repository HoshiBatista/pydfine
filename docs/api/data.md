# Data & convert

Bring a YOLO detection dataset into the COCO layout `DFINE.train(data=...)` and
`DFINE.val(data=...)` consume — no manual reshuffling.

## The two layouts

YOLO stores one `.txt` per image (`class cx cy w h`, normalized, class 0-indexed), with
images under `images/<split>/` and labels under the mirror `labels/<split>/`:

```
yolo/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml                      # optional: class names + split paths
```

`yolo_to_coco` writes the COCO layout D-FINE trains on:

```
coco/
  train/  val/                   # images
  annotations/
    instances_train.json
    instances_val.json
```

Category ids stay **0-indexed** (= the YOLO class id), so they line up with the model's
contiguous labels under the default `remap_mscoco_category=False`.

## Quickstart

```python
from dfine import yolo_to_coco

written = yolo_to_coco("yolo/", "coco/")
# {"train": "coco/annotations/instances_train.json", "val": ".../instances_val.json"}
```

Then train straight on the output:

```python
from dfine import DFINE
import json

num_classes = len(json.load(open(written["train"]))["categories"])
model = DFINE(size="s", num_classes=num_classes, imgsz=640)
model.train(data="coco/", epochs=100)
```

Or from the shell:

```bash
dfine convert yolo/ coco/ --names cat dog bird
```

## Where splits and class names come from

- **Class names:** an explicit `class_names=[...]` wins; otherwise `data.yaml`'s `names`
  (list or `{id: name}` dict) is used; otherwise names are inferred as `class_<i>` from
  the label ids.
- **Splits:** an explicit `splits={...}` wins; otherwise the `train`/`val`/`test` paths
  declared in `data.yaml` are used (resolved relative to the yaml, including the common
  Roboflow `../valid/images` form); otherwise folders are auto-detected
  (`images/<split>` and `<split>/images`, with `valid`/`validation` accepted as val-split
  aliases). A split declared in `data.yaml` but not found on disk, or a missing `val`
  split, logs a warning instead of being silently dropped.

!!! tip "Roboflow / Ultralytics exports"
    These declare their splits in `data.yaml` (often `val: ../valid/images`) and name the
    validation folder `valid`. `yolo_to_coco` reads those paths and folder aliases
    directly, so a stock Roboflow export converts both splits with no extra flags.

## Common variations

```python
# Explicit class names (skip data.yaml)
yolo_to_coco("yolo/", "coco/", class_names=["cat", "dog", "bird"])

# Point at split image dirs yourself (relative to the root, or absolute)
yolo_to_coco("yolo/", "coco/", splits={"train": "images/train", "val": "images/val"})

# Symlink images instead of copying (saves disk on large datasets)
yolo_to_coco("yolo/", "coco/", copy_images=False)

# Rename the output split folders
yolo_to_coco("yolo/", "coco/", split_names={"train": "train2017", "val": "val2017"})
```

Segmentation-style rows (a class id followed by polygon points) are accepted too — their
bounding box is derived. The converter is torch-free (only needs Pillow for image sizes,
and PyYAML to read a `data.yaml`).

## API

::: dfine.convert.yolo_to_coco
