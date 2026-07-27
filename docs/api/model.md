# DFINE

The public, backend-agnostic detector class: build from typed params, load released
weights, and `predict` / `train` / `val` / `export` / `benchmark`. Backend details never
leak through its kwargs.

For task-oriented, copy-paste recipes see the **[examples cookbook](../examples.md)** and
the runnable **[`templates/`](https://github.com/HoshiBatista/pydfine/tree/main/templates)**.

## Build a model

```python
from dfine import DFINE, DFINEConfig

# From a released checkpoint — size + num_classes inferred, weights downloaded + loaded.
model = DFINE.from_pretrained("dfine-s")

# From a size preset (ImageNet backbone); every field is overridable inline.
model = DFINE(size="l", num_classes=80, imgsz=640, device="cuda")

# From a bare size + your own weights.
model = DFINE(size="m", num_classes=3).load("runs/train/best.pth")

# From a fully custom config object.
cfg = DFINEConfig.preset("s", num_classes=3, class_names=["cat", "dog", "bird"])
model = DFINE(config=cfg)
```

!!! warning "imgsz is a build-time choice"
    The encoder's positional embeddings are precomputed for the model's `imgsz`, so
    `predict`/`export`/`benchmark` must use that same size. To run at a different
    resolution, rebuild: `DFINE(size=…, imgsz=…)`.

## Predict

```python
results = model.predict("street.jpg", conf=0.4)      # list[Results], one per image
r = results[0]
r.boxes.xyxy, r.boxes.conf, r.boxes.cls              # tensors, original-image scale
r.save("out.jpg")

# Folders, globs, and lists all work; save flags write a run dir.
model.predict("images/", save=True, save_txt=True, save_crop=True)
```

See [Results & Boxes](results.md) for the returned containers.

## Train, validate, export

```python
# Fine-tune on a COCO root (loaders built for you). Multi-GPU: devices=N.
model.train(data="coco/", epochs=72, batch_size=8, devices=2)

# 12 COCO metrics (+ optional analytics plots).
metrics = model.val(data="coco/", plots=True)
print(metrics["AP"])

# Deployable graph (postprocessor fused in).
model.export(format="onnx", simplify=True)
```

## Inspect and time

```python
model.info(verbose=True)             # layers / params / gradients / GFLOPs
model.benchmark(runs=100, batch=1)   # {"ms_per_image", "fps", "device", ...}
```

## Video

```python
model.predict_video("in.mp4", output="out.mp4", track=True)   # annotated mp4 + IDs
for r in model.predict_video("in.mp4", stream=True):          # per-frame Results
    ...
```

## API reference

::: dfine.model.DFINE
