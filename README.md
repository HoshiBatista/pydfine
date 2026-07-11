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

Early construction. Done so far: the **config-first core** — `DFINEConfig` (every
model/training param as a typed field), verified `n/s/m/l/x` presets, validation, the
checkpoint registry, and the `dfine models` CLI. The inference/training backends
(`predict`/`train`/`val`/`export`) are next. See [`docs/ROADMAP.md`](docs/ROADMAP.md)
for the full status.

```python
from dfine import DFINEConfig

cfg = DFINEConfig.preset("l", num_classes=3)   # verified upstream defaults
cfg = DFINEConfig.preset("n")                  # 2-level, hidden_dim=128
```

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
