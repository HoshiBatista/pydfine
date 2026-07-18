# dfine

A batteries-included Python library for the **D-FINE** real-time object detector
([Peterande/D-FINE](https://github.com/Peterande/D-FINE), ICLR 2025 Spotlight),
with an `ultralytics`-style developer experience.

📖 **Documentation:** <https://hoshibatista.github.io/pydfine/>

**Design goal:** the entire model — backbone, encoder, decoder, losses, denoising,
training, augmentation — is configured through **typed Python parameters on one
class**. No YAML files, no config-registry indirection, no `torchrun` incantations.

```python
from dfine import DFINE

# Presets fill sensible defaults; every single field is overridable inline.
model = DFINE(
    size="l",                 # n | s | m | l | x  -> sets backbone, dims, depths
    num_classes=80,
    num_queries=300,
    hidden_dim=256,
    reg_max=32,               # Fine-grained Distribution Refinement bins
    backbone="hgnetv2_b4",
    backbone_pretrained=True,
    device="cuda",
)

results = model.predict("street.jpg", conf=0.4)
results[0].save("out.jpg")

model.train(data="dataset/", epochs=72, imgsz=640, batch=32)
metrics = model.val()
model.export(format="onnx")
```

Fully custom architecture, no preset:

```python
model = DFINE(
    num_classes=3,
    backbone="hgnetv2_b0", use_lab=True, freeze_at=-1,
    hidden_dim=256, encoder_dim_feedforward=1024, encoder_layers=1, nhead=8,
    decoder_layers=4, eval_idx=-1, num_levels=3, num_points=[3, 6, 3],
    reg_max=32, reg_scale=4.0, lqe_layers=2,
    num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
    class_names=["cat", "dog", "bird"],
)
```

## Status

Inference is complete and bit-exact with upstream; the training stack (loop, data,
augmentation, COCO `val`, multi-GPU DDP, visualization) is in — only `export` remains.
Done so far:

- **Config-first core** — `DFINEConfig` (every model/training param as a typed field),
  verified `n/s/m/l/x` presets, validation, checkpoint registry, `dfine models` CLI.
- **Native model port (Path A)** under `dfine/backends/native/` — the full
  **backbone → encoder → decoder** stack ported from upstream `src/` with the
  YAML/registry layer stripped: `HGNetv2`, `HybridEncoder`, and `DFINETransformer`
  (FDR head, LQE, contrastive denoising). Layer/param names preserved so released
  `.pth` load unchanged. Each module builds from the config via `from_config(cfg)`.

- **Working inference** — assembled `DFINE` model + `DFINEPostProcessor`, upstream
  `.pth` loading (`registry`/`downloads`, `from_pretrained`), and the public
  `DFINE(...).predict(...) -> Results` API (`.boxes.xyxy/.conf/.cls`, `.plot()/.save()`).

- **Video** — `DFINE.predict_video(source, output=...)` writes an annotated mp4, or
  `stream=True` yields per-frame `Results` (needs `pip install dfine[video]`).

- **Training loss** — `HungarianMatcher` + `DFINECriterion` (VFL + L1 + GIoU + FGL +
  DDF) ported and wired from the config; consumes the decoder's training-mode output.

- **Training loop** — `DFINE.train(train_loader, epochs=...)` runs the ported D-FINE
  loop (AdamW param groups, EMA, AMP, grad clip, warmup + flat-cosine LR) with the same
  **progress visualization as upstream**: a live console readout (`MetricLogger`) plus
  TensorBoard scalars and a `loss_curve.png` under `output_dir` (needs
  `pip install dfine[train]`; W&B optional).

- **COCO data + augmentation** — `dfine.train.dataset.build_coco_dataloader(img_folder,
  ann_file, cfg=...)` gives a ready `(images, targets)` loader (contiguous-label remap,
  multi-scale collate); pass `transforms=dfine.train.augment.train_transforms(imgsz,
  stop_epoch=...)` for D-FINE's full augment pipeline (photometric distort, zoom-out,
  IoU-crop, H-flip) with the two-phase no-aug tail. Feeds straight into `DFINE.train`.

- **Have a YOLO dataset?** Convert it once — `dfine convert path/to/yolo path/to/coco`
  (or `dfine.yolo_to_coco(...)`) — then `DFINE.train(data="path/to/coco")`. It reads the
  `images/<split>` + `labels/<split>` layout (and `data.yaml` names) and writes the COCO
  layout with 0-indexed categories that line up with the model's labels.

Only `export` (Phase 3) remains. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full
status.

```python
from dfine import DFINEConfig

cfg = DFINEConfig.preset("l", num_classes=3)   # verified upstream defaults
cfg = DFINEConfig.preset("n")                  # 2-level, hidden_dim=128
```

The ported modules already run end-to-end (needs the `torch` extra installed):

```python
import torch
from dfine import DFINEConfig
from dfine.backends.native import HGNetv2, HybridEncoder, DFINETransformer

cfg = DFINEConfig.preset("l", num_classes=80)
backbone = HGNetv2.from_config(cfg).eval()
encoder = HybridEncoder.from_config(cfg).eval()
decoder = DFINETransformer.from_config(cfg).eval()

out = decoder(encoder(backbone(torch.randn(1, 3, cfg.imgsz, cfg.imgsz))))
# out["pred_logits"]: (1, 300, 80)   out["pred_boxes"]: (1, 300, 4)  [cxcywh, 0..1]
```

> The one-class `DFINE(...)` façade at the top of this README works today for
> **inference** (`predict`/`load`/`from_pretrained`), **training** — both
> `train(data="coco/", epochs=...)` (a standard COCO root; the loaders are built for
> you) and `train(train_loader, epochs=...)` (a hand-built loader) — and **COCO
> evaluation** (`val(data="coco/")` → the 12 named COCO metrics, also run each epoch
> during `train`), all with the `dfine[train]` extra. **Multi-GPU** is a single kwarg:
> `train(data="coco/", devices=N)` spawns one DDP worker per GPU (or launch with
> `torchrun` and call `train(...)` as usual). `export` is phase-stubbed and raises a
> clear "arriving in Phase N" until implemented.

## Why this exists

Upstream D-FINE is an excellent research repo, but using it means editing YAML,
copying config include-trees, and launching scripts. This library turns all of that
into one importable, fully-typed class with presets — so a developer can go from
`pip install` to a trained custom detector without touching a config file.

## For contributors and AI agents

This project is built to be developed largely by coding agents (Claude Code / any
agent that reads `AGENTS.md`). Start here:

| File | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Canonical agent guide** — architecture, conventions, workflow, commands, definition of done. Read first. |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code–specific notes; defers to `AGENTS.md`. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How D-FINE works and how we re-shape it into Python. |
| [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) | Every model parameter, default, and per-size preset. The heart of the "one class, many params" design. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased, checkbox task plan. |

## License & attribution

pydfine is licensed under the [Apache License 2.0](LICENSE).

It is a **derivative work of D-FINE**
([Peterande/D-FINE](https://github.com/Peterande/D-FINE), Apache-2.0, © 2024 The
D-FINE Authors): the model is a native port of upstream `src/`, with layer and
parameter names preserved so released `.pth` checkpoints load unchanged. Every ported
module under `dfine/backends/native/` carries a per-file header crediting its source
and describing the changes. See [`NOTICE`](NOTICE) for the full attribution, including
D-FINE's own lineage (RT-DETR, DETR, PaddleDetection).
