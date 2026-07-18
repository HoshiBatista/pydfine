# pydfine

A batteries-included, **config-first** Python library for the **D-FINE** real-time
object detector ([Peterande/D-FINE](https://github.com/Peterande/D-FINE), ICLR 2025
Spotlight), with an `ultralytics`-style developer experience.

**Design goal:** the entire model — backbone, encoder, decoder, losses, denoising,
training, augmentation — is configured through **typed Python parameters on one class**
([`DFINEConfig`](api/config.md)). No YAML on the user path, no config-registry
indirection, no `torchrun` incantations.

## Install

```bash
pip install pydfine              # core (config + CLI, torch-free)
pip install pydfine[torch]       # + inference (model build + predict)
pip install pydfine[train]       # + training / COCO val
pip install pydfine[export]      # + ONNX export
pip install pydfine[track]       # + ByteTrack on predict_video
```

## Quickstart

```python
from dfine import DFINE

# Presets fill sensible defaults; every field is overridable inline.
model = DFINE(size="l", num_classes=80, device="cuda")

results = model.predict("street.jpg", conf=0.4)
results[0].save("out.jpg")

model.train(data="dataset/", epochs=72)   # fine-tune on a COCO dataset
metrics = model.val(data="dataset/")      # COCO metrics
model.export(format="onnx")               # deployable ONNX graph
```

Load released COCO weights in one line:

```python
model = DFINE.from_pretrained("dfine-s")   # resolve + download + strict-load
```

## CLI

```bash
dfine models                       # list presets + known checkpoints
dfine predict dfine-s img.jpg      # detect and save annotated output
dfine val dfine-l --data coco/     # COCO metrics
dfine train n --data coco/         # fine-tune
dfine export dfine-m               # ONNX
dfine convert yolo/ coco/          # YOLO dataset -> COCO layout
```

## Learn more

- [Architecture](ARCHITECTURE.md) — how the model works + module→param map.
- [Config reference](CONFIG_REFERENCE.md) — every parameter and per-size preset.
- **API** — [`DFINE`](api/model.md), [`DFINEConfig`](api/config.md),
  [`Results`/`Boxes`](api/results.md), [tracking](api/tracking.md),
  [data & convert](api/data.md), [export](api/export.md).
