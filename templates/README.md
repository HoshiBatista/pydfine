# pydfine templates

Copy-paste starting points for common pydfine tasks. Every file is a **runnable
script** with an `argparse` CLI — grab one, adjust the defaults, and go. They use only
the public API (`from dfine import DFINE, DFINEConfig, yolo_to_coco`), so nothing here
reaches into internals.

```bash
pip install pydfine[torch]          # inference templates
pip install pydfine[train]          # training / validation templates
pip install pydfine[export]         # export template
pip install pydfine[track]          # video tracking template
```

| Template | What it shows | Extra |
|---|---|---|
| [`predict_image.py`](predict_image.py) | Detect on one image, read boxes, save an annotated copy. | `[torch]` |
| [`predict_folder.py`](predict_folder.py) | Batch a folder/glob, save images + YOLO labels + crops. | `[torch]` |
| [`predict_video.py`](predict_video.py) | Annotate a video, optionally with ByteTrack IDs. | `[video]` / `[track]` |
| [`train_coco.py`](train_coco.py) | Fine-tune a preset on a COCO dataset root. | `[train]` |
| [`train_from_yolo.py`](train_from_yolo.py) | Convert a YOLO dataset then train on it. | `[train]` |
| [`train_segmentation.py`](train_segmentation.py) | Train instance / semantic segmentation from a YOLO-style root. | `[train]` |
| [`finetune_custom_classes.py`](finetune_custom_classes.py) | Start from COCO weights, retrain the head for your classes. | `[train]` |
| [`validate.py`](validate.py) | COCO metrics + the analytics plot bundle. | `[train]` |
| [`export_onnx.py`](export_onnx.py) | Export ONNX / TorchScript for deployment. | `[export]` |
| [`deploy_onnxruntime.py`](deploy_onnxruntime.py) | Run an exported ONNX graph with onnxruntime (no torch at serve time). | onnxruntime |
| [`custom_architecture.py`](custom_architecture.py) | Build a fully custom model from `DFINEConfig` — no preset. | `[torch]` |
| [`config_as_yaml.py`](config_as_yaml.py) | Freeze a config to YAML and rebuild the exact model. | `[train]` |
| [`segmentation.py`](segmentation.py) | Instance + semantic segmentation inference. | `[hf]` |
| [`results_interop.py`](results_interop.py) | Export predictions to DataFrame / COCO JSON / supervision. | `[interop]` |
| [`track_and_count.py`](track_and_count.py) | Count objects crossing a line via streaming ByteTrack. | `[track]` |
| [`benchmark_and_info.py`](benchmark_and_info.py) | Measure latency/FPS and print a model summary. | `[torch]` |

## The one rule to remember

D-FINE bakes the input resolution into the encoder's positional embeddings, so the
image size is a **build-time** choice, not a call-time one. Pick it when you construct
the model:

```python
model = DFINE(size="l", num_classes=80, imgsz=640)  # <- set imgsz here
model.predict("img.jpg")  # not here
```

Passing `predict(imgsz=...)` / `export(imgsz=...)` that differs from the model's own
`imgsz` raises — rebuild instead.

## See also

- [Examples cookbook](../docs/examples.md) — the same recipes with explanation.
- [`DFINE` API](../docs/api/model.md) · [Config reference](../docs/CONFIG_REFERENCE.md)
