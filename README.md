# dfine

A batteries-included Python library for the **D-FINE** real-time object detector
([Peterande/D-FINE](https://github.com/Peterande/D-FINE), ICLR 2025 Spotlight),
with an `ultralytics`-style developer experience.

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

Early construction, but the model core is in. Done so far:

- **Config-first core** — `DFINEConfig` (every model/training param as a typed field),
  verified `n/s/m/l/x` presets, validation, checkpoint registry, `dfine models` CLI.
- **Native model port (Path A)** under `dfine/backends/native/` — the full
  **backbone → encoder → decoder** stack ported from upstream `src/` with the
  YAML/registry layer stripped: `HGNetv2`, `HybridEncoder`, and `DFINETransformer`
  (FDR head, LQE, contrastive denoising). Layer/param names preserved so released
  `.pth` load unchanged. Each module builds from the config via `from_config(cfg)`.

Next: postprocessor + a single assembled `DFINE` model, then `predict`/`train`/
`val`/`export`. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full status.

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

> The one-class `DFINE(...)` façade at the top of this README is the target public
> API; it lands once the modules above are assembled behind the postprocessor.

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

## License

Apache-2.0. Ported/vendored upstream code retains D-FINE's Apache-2.0 license and attribution.
