# Examples cookbook

Task-oriented recipes for pydfine. Each block is self-contained and uses only the
public API. Runnable versions of most of these live in
[`templates/`](https://github.com/HoshiBatista/pydfine/tree/main/templates) — copy a
script and adjust.

!!! tip "Resolution is a build-time choice"
    D-FINE bakes the input size into the encoder's positional embeddings, so set `imgsz`
    when you **construct** the model — not on `predict`/`export`. Passing a different
    `predict(imgsz=…)` raises; rebuild with `DFINE(size=…, imgsz=…)` instead.

## Contents

- [Predict on one image](#predict-on-one-image)
- [Batch a folder or glob](#batch-a-folder-or-glob)
- [Work with the Results object](#work-with-the-results-object)
- [Export predictions to other formats](#export-predictions-to-other-formats)
- [Video and tracking](#video-and-tracking)
- [Count objects crossing a line](#count-objects-crossing-a-line)
- [Train on a COCO dataset](#train-on-a-coco-dataset)
- [Train from a YOLO dataset](#train-from-a-yolo-dataset)
- [Fine-tune on custom classes](#fine-tune-on-custom-classes)
- [Validate with analytics](#validate-with-analytics)
- [Export for deployment](#export-for-deployment)
- [Run an exported ONNX graph](#run-an-exported-onnx-graph)
- [Fully custom architecture](#fully-custom-architecture)
- [Config as YAML](#config-as-yaml)
- [Segmentation](#segmentation)
- [Train segmentation](#train-segmentation)
- [Benchmark and inspect](#benchmark-and-inspect)
- [Interop: supervision](#interop-supervision)
- [CLI equivalents](#cli-equivalents)

---

## Predict on one image

```python
from dfine import DFINE

model = DFINE.from_pretrained("dfine-s")  # build + download + load, one line
result = model.predict("street.jpg", conf=0.4)[0]

print(result.verbose())  # "3 persons, 1 car"
for xyxy, conf, cls in result.boxes:
    print(result.names[int(cls)], float(conf), xyxy.tolist())

result.save("out.jpg")  # annotated copy
```

`from_pretrained` reads the size and class count from the checkpoint name. See
`dfine models` (or [`list_checkpoints()`](api/config.md)) for the catalogue —
`dfine-{n,s,m,l,x}` plus `-obj365` / `-obj2coco` variants and `dfine-seg-*`.

## Batch a folder or glob

A directory source runs over every image in it (sorted); a glob runs over the matches;
a list can mix folders, globs, and explicit paths.

```python
results = model.predict(
    "images/",  # or "images/*.jpg", or ["a.jpg", "b.png", "more/"]
    conf=0.3,
    save=True,  # annotated images -> runs/detect/predict/
    save_txt=True,  # YOLO-format labels -> .../labels/
    save_crop=True,  # per-detection crops -> .../crops/
    save_conf=True,  # append score to the label rows
)
print(sum(len(r.boxes) for r in results), "detections total")
```

Outputs go to a fresh, auto-incremented run dir (`predict`, `predict2`, …) so runs never
clobber each other.

## Work with the Results object

```python
r = model.predict("street.jpg")[0]

r.boxes.xyxy  # (N, 4) float tensor, original-image pixels
r.boxes.conf  # (N,)   scores
r.boxes.cls  # (N,)   integer class ids
r.boxes.id  # (N,) track ids after predict_video(track=True), else None
r.names  # {id: name} mapping
len(r.boxes)  # detection count

img = r.plot()  # HxWx3 uint8 RGB ndarray (annotated)
r.save("out.jpg")  # plot + write
r.save_txt("out.txt", save_conf=True)
r.save_crop("crops/", file_name="street.jpg")
```

## Export predictions to other formats

One `Results` exports to the ecosystems you already use, so pydfine drops into an
existing pipeline without glue code (`results_interop.py`).

```python
r = model.predict("street.jpg", conf=0.4)[0]

r.summary(decimals=2)  # list[dict] — JSON-serializable (r.tojson() for the string)
r.to_coco(image_id=1)  # COCO result dicts for pycocotools / faster-coco-eval
r.to_pandas()  # DataFrame: xmin,ymin,xmax,ymax,confidence,class,name  (pydfine[interop])
r.to_supervision()  # supervision.Detections for its annotators/zones  (pydfine[interop])
```

For a `task="segment"` result these also carry the masks (RLE in `to_coco`, a bool
`mask` array in `to_supervision`).

## Video and tracking

```python
# Write an annotated mp4 (source resolution/fps).
model.predict_video("in.mp4", output="out.mp4", conf=0.3)

# Persistent IDs across frames with ByteTrack (needs pydfine[track]).
model.predict_video("in.mp4", output="tracked.mp4", track=True)

# Stream per-frame Results for custom logic — nothing is written.
for result in model.predict_video("in.mp4", stream=True, track=True):
    ids = result.boxes.id
    ...  # count, crop, forward frames elsewhere
```

## Count objects crossing a line

Stream tracked frames and tally a crossing when a track's center moves across a line.
A stable ByteTrack `id` means each object is counted once — the skeleton for
people-counting / traffic flow (`track_and_count.py`).

```python
model = DFINE.from_pretrained("dfine-s")
prev_y, down, up = {}, 0, 0

for r in model.predict_video("traffic.mp4", stream=True, track=True, conf=0.4):
    line_y = 0.5 * r.orig_shape[0]  # orig_shape is (H, W)
    if r.boxes.id is None:
        continue
    for i, (xyxy, _conf, cls) in enumerate(r.boxes):
        tid = int(r.boxes.id[i])
        cy = float((xyxy[1] + xyxy[3]) / 2)
        was = prev_y.get(tid)
        if was is not None and was < line_y <= cy:
            down += 1
        elif was is not None and was > line_y >= cy:
            up += 1
        prev_y[tid] = cy

print(f"down: {down}  up: {up}")
```

## Train on a COCO dataset

```python
from dfine import DFINE

model = DFINE(size="l", num_classes=80, imgsz=640)
model.train(
    data="coco/",  # train/ + val/ + annotations/instances_{train,val}.json
    epochs=72,
    batch_size=8,
    output_dir="runs/train",
)
metrics = model.val(data="coco/")
print(metrics["AP"])
```

- **Stock 80-class MS-COCO** (sparse category ids): add
  `remap_mscoco_category=True` to both `train` and `val`.
- **Multi-GPU**: `model.train(data="coco/", devices=4)` spawns one DDP worker per GPU —
  no `torchrun`. (Multi-GPU needs `data=`, not an in-memory loader.)
- **Resume**: `model.train(..., resume=True)` continues from `output_dir/last.pth`.
- **Per-epoch analytics**: `val_plots=True` renders the full plot bundle each epoch.

Checkpoints (rank 0): `last.pth` each epoch, `best.pth` when the primary metric improves.

## Train from a YOLO dataset

Convert once, then train on the COCO output. The converter reads `images/<split>` +
`labels/<split>`, picks up class names and split paths from `data.yaml` (including
Roboflow's `valid/` split), and writes 0-indexed categories that line up with the
model's labels.

```python
from dfine import DFINE, yolo_to_coco

written = yolo_to_coco("yolo/", "coco/")  # {"train": ".../instances_train.json", ...}

model = DFINE(size="s", num_classes=3, imgsz=640)
model.train(data="coco/", epochs=100)
```

Or from the shell: `dfine convert yolo/ coco/`.

## Fine-tune on custom classes

Build at your class count — the ImageNet-pretrained backbone gives you strong features
to fine-tune from (this is transfer learning, not training from scratch).

```python
names = ["cat", "dog", "bird"]
model = DFINE(size="s", num_classes=len(names), class_names=names, imgsz=640)
model.train(data="coco/", epochs=48, batch_size=8)
model.predict("sample.jpg", conf=0.4, save=True)
```

!!! note "Reusing a released detector head"
    `model.load("dfine-s")` is a strict load — it only fits a model with the **same**
    class count (80 for COCO). To transfer the full COCO detector, keep
    `num_classes=80`; for your own classes, use the ImageNet-backbone path above.

## Validate with analytics

```python
metrics = model.val(data="coco/", plots=True, output_dir="runs/val")
print(metrics["AP"], metrics["AP50"], metrics["AP75"])
# runs/val/{confusion_matrix,pr_curve,f1_curve,p_curve,r_curve}.png + worst/
```

`val` returns the 12 standard COCO metrics keyed by name (`AP` is mAP@[.50:.95]). See
[Validation & analytics](api/validation.md) for how to read each plot and pick a
deployment confidence from the F1 curve.

## Export for deployment

```python
model = DFINE.from_pretrained("dfine-s")

onnx = model.export(format="onnx", simplify=True)  # dynamic batch by default
ts = model.export(format="torchscript")  # torch-only, fixed batch/imgsz
```

The graph fuses the postprocessor: it takes `(images, orig_target_sizes)` and returns
`(labels, boxes, scores)` already scaled to the original image. Build a TensorRT engine
from the ONNX:

```python
from dfine.export import tensorrt_command

print(tensorrt_command(onnx, fp16=True))  # trtexec ... --fp16
```

## Run an exported ONNX graph

At serve time you need neither torch nor pydfine — just a runtime. Preprocessing mirrors
`predict` (resize-to-square, `/255`, CHW) so results match the torch path
(`deploy_onnxruntime.py`).

```python
import numpy as np, onnxruntime as ort
from PIL import Image

img = Image.open("street.jpg").convert("RGB")
x = np.asarray(img.resize((640, 640), Image.BILINEAR), np.float32) / 255.0
images = x.transpose(2, 0, 1)[None]  # 1CHW
sizes = np.array([[img.width, img.height]], dtype=np.int64)  # original (W, H)

sess = ort.InferenceSession("dfine-s.onnx", providers=["CPUExecutionProvider"])
labels, boxes, scores = sess.run(None, {"images": images, "orig_target_sizes": sizes})
# boxes[0] are already in original-image pixels; filter by scores[0] >= conf.
```

The batch dim is dynamic by default, so one graph serves any batch size. `imgsz` must
match the model you exported (the encoder's positional embeddings are baked at that size).

## Fully custom architecture

No preset — every knob is a typed kwarg. See the [Config reference](CONFIG_REFERENCE.md)
for all fields and defaults.

```python
from dfine import DFINE, DFINEConfig

model = DFINE(
    num_classes=3,
    class_names=["cat", "dog", "bird"],
    backbone="hgnetv2_b0",
    hidden_dim=256,
    encoder_layers=1,
    decoder_layers=4,
    num_levels=3,
    num_points=[3, 6, 3],
    reg_max=32,
    num_denoising=100,
    imgsz=640,
)

# Or: start from a preset and override just what you need.
cfg = DFINEConfig.preset("s", num_classes=3, class_names=["cat", "dog", "bird"])
model = DFINE(config=cfg)
```

## Config as YAML

pydfine is config-first: the whole architecture is a typed `DFINEConfig`. Freeze one to
YAML for versioning/sharing, then rebuild the exact model — no kwargs to remember
(`config_as_yaml.py`). This is interop only; the user path never needs YAML.

```python
from dfine import DFINE, DFINEConfig

cfg = DFINEConfig.preset("l", num_classes=3, imgsz=640)
cfg.to_yaml("model.yaml")  # write it (or `text = cfg.to_yaml()` for the string)

cfg = DFINEConfig.from_yaml("model.yaml")  # path or a YAML string
model = DFINE(config=cfg)  # the exact same architecture
```

## Segmentation

The same façade covers instance and semantic segmentation — predictions come back at the
original image scale. Needs `pydfine[hf]`.

```python
# Instance segmentation — masks + boxes (dfine-seg-{n,s,m,l,x})
model = DFINE.from_pretrained("dfine-seg-l")
r = model.predict("street.jpg", conf=0.4)[0]
r.boxes.xyxy  # (N, 4) original-scale boxes
r.masks.data  # (N, H, W) bool masks, 1:1 with boxes
r.plot()  # boxes + per-instance mask overlays

# Semantic segmentation — dense label map, no boxes
model = DFINE(size="l", task="sem_seg", num_classes=19)
r = model.predict("street.jpg")[0]
r.sem_seg.data  # (H, W) uint8 class ids (255 = void)
```

## Train segmentation

Both seg tasks train from a single YOLO-style root (`images/` + `labels/`) through the
same `train` façade — polygon `.txt` for instance, class-id `.png` for semantic
(`train_segmentation.py`). The val split is auto-scored with mask AP / mIoU.

```python
# Instance: fine-tune the released mask head (needs pydfine[hf] for the weights).
model = DFINE.from_pretrained("dfine-seg-l")
model.train(data="dataset/", epochs=50, batch_size=8)

# Semantic: build fresh at your class count (e.g. 19-class Cityscapes).
model = DFINE(size="l", task="sem_seg", num_classes=19)
model.train(data="dataset/", epochs=100, batch_size=8)
```

See the [segmentation training guide](seg-training.md) for the dataset layout, the
train/val split rules, and the loss/matcher knobs.

## Benchmark and inspect

```python
model = DFINE.from_pretrained("dfine-s")

model.info(verbose=True)  # layers / params / gradients / GFLOPs + per-module split
model.benchmark(runs=100, batch=1)  # {"ms_per_image", "fps", "device", ...}
```

`benchmark` times the compute-bound model forward (the postprocessor is excluded).
GFLOPs need `thop` installed.

## Interop: supervision

```python
r = model.predict("street.jpg")[0]
detections = r.to_supervision()  # supervision.Detections (needs pydfine[interop])
```

Use it with the [`supervision`](https://supervision.roboflow.com/) annotators, zones, and
trackers.

## CLI equivalents

Everything above has a shell counterpart:

```bash
dfine models                             # list presets + checkpoints
dfine predict dfine-s img.jpg --conf 0.4 # detect + save
dfine predict dfine-s imgs/ --save-txt   # batch a folder, write labels
dfine train l --data coco/ --epochs 72 --devices 4
dfine val dfine-l --data coco/ --plots
dfine export dfine-m --format onnx --simplify
dfine convert yolo/ coco/                # YOLO -> COCO
dfine benchmark dfine-s --runs 100
dfine info dfine-l --verbose
```
