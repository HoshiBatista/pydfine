# AGENTS.md — build guide for coding agents

This is the canonical instruction file for any AI agent (Claude Code, etc.) working
in this repo. Read it fully before writing code. `CLAUDE.md` defers to this file.

---

## 1. Mission

Build a `pip install`-able Python library that wraps **D-FINE**
(github.com/Peterande/D-FINE) with an **ultralytics-style** API, where the whole
model and training recipe is configured through **typed Python parameters on one
class (`DFINE`)** — **no YAML files, no config-registry indirection**.

A developer should be able to:

```python
from dfine import DFINE
model = DFINE(size="l", num_classes=80)      # or fully custom kwargs
model.predict(...); model.train(...); model.val(...); model.export(...)
```

## 2. Non-negotiable design principles

1. **Config-as-Python.** All model/training options are fields on a frozen
   `DFINEConfig` dataclass, surfaced as `DFINE(...)` constructor kwargs. Never
   require the user to read or write a `.yml`. (A `DFINEConfig.from_yaml()` /
   `.to_yaml()` interop helper is allowed, but must never be on the critical path.)
2. **No registry / DI magic.** Upstream builds modules via a custom
   `register`/`create` + `YAMLConfig` system in `src/core`. We **delete that layer**
   and instantiate every `nn.Module` directly from dataclass fields.
3. **Presets, not lock-in.** `size="n|s|m|l|x"` fills the per-variant defaults
   (see `docs/CONFIG_REFERENCE.md`), but **every** field remains overridable inline.
4. **Weight-compatible.** A model built from a preset must load upstream pretrained
   `.pth` checkpoints and reproduce their reported COCO AP. Porting must preserve
   layer names/shapes (or provide a documented remap). This is the correctness bar.
5. **Ultralytics-shaped ergonomics.** `predict/train/val/export`, a `Results`
   object with `.boxes.xyxy/.conf/.cls`, `.plot()/.save()`, auto weight download.
6. **Typed and documented.** Full type hints; every public param has a docstring
   line and appears in `docs/CONFIG_REFERENCE.md`.
7. **Small, testable units.** Each ported module gets a shape/parity test.

## 3. Two build paths — we chose Path A (see decision below)

- **Path A — Port upstream `src/` into pure Python.** Copy the `nn.Module`s
  (HGNetv2, HybridEncoder, DFINETransformer, matcher, criterion, postprocessor),
  strip the registry decorators, and wire them from `DFINEConfig`. Maximum control,
  exact weight parity with upstream `.pth`.
- **Path B — Build on `transformers` `DFineModel`/`DFineConfig`.** HuggingFace already
  ported D-FINE YAML-free (`DFineConfig` is literally the flat param class we want,
  and pretrained weights load by name). Wrapping it gives us the model internals for
  free; we own only the ergonomics layer (config presets, predict/train/val/export,
  Results, augmentation, downloads).

**Decision (2026-07-11): we are building Path A** — porting upstream `src/` directly
into `dfine/backends/native/` for byte-exact parity with the released `.pth`. This
means `transformers` is **not** a dependency. Progress so far: the full inference
stack is done — `HGNetv2`, `HybridEncoder`, `DFINETransformer` (FDR/LQE/denoising),
`DFINEPostProcessor`, the assembled `DFINE` module, and the `.pth` loader — plus the
public `DFINE.predict`/`predict_video` API, the checkpoint registry + download/cache,
the training loss (`HungarianMatcher` + `DFINECriterion`), and now the single-process
training loop (`dfine/train/`: param groups, EMA, AMP, warmup + flat-cosine LR, and
upstream-style progress visualization — console `MetricLogger` + TensorBoard +
`loss_curve.png`) behind `DFINE.train(train_loader, ...)`. Next is the COCO
dataset/augmentation (so `train(data=...)` needs no hand-built loader) and `.val()`.
See `docs/ROADMAP.md`.

The `dfine/backends/` boundary is still kept so a `transformers` wrapper (Path B)
could be added later without touching the public `DFINE` API. Whichever path: the
**public API and parameter names must not depend on the backend.**

## 4. Repository layout (target)

Legend: ✅ done · ⬜ planned (target path shown).

```
dfine/
  __init__.py          # ✅ lazy exports DFINE/Results/Boxes (base import stays torch-free)
  config.py            # ✅ DFINEConfig dataclass + SIZE_PRESETS + validation
  model.py             # ✅ DFINE: predict/predict_video/train + input loading/preprocess;
                       #    load/from_pretrained; val/export are phase stubs
  results.py           # ✅ Results / Boxes (.boxes.xyxy/.conf/.cls, .plot()/.save())
  registry.py          # ✅ checkpoint catalogue: name -> CheckpointSpec(size,dataset,
                       #    num_classes,url); resolve_weights/config_for
  downloads.py         # ✅ weight cache/download (atomic, $DFINE_CACHE_DIR)
  backends/
    __init__.py        # backend package docstring
    native/            # ✅ Path A port — full inference + loss
      common.py        #   FrozenBatchNorm2d
      ops.py           #   get_activation, inverse_sigmoid, deformable attn core, ...
      box_ops.py       #   box conversions, IoU/GIoU
      dfine_utils.py   #   FDR weighting_function / distance2bbox / bbox2distance
      denoising.py     #   contrastive denoising (training)
      coco.py          #   MS-COCO id/name maps
      dist.py          #   single-process world-size/rank shim (criterion)
      hgnetv2.py       #   HGNetv2 backbone (B0-B6)
      hybrid_encoder.py#   HybridEncoder (AIFI + CCFM/GELAN)
      dfine_decoder.py #   DFINETransformer (FDR head, LQE)
      postprocessor.py #   DFINEPostProcessor (top-k decode to xyxy)
      dfine.py         #   assembled DFINE (backbone+encoder+decoder) + .load()
      loader.py        #   upstream .pth -> native modules (strict load)
      matcher.py       #   HungarianMatcher (LSAP)
      criterion.py     #   DFINECriterion (VFL/L1/GIoU/FGL/DDF)
    # transformers.py  # Path B wrapper — optional, not planned yet
  train/               # ◐ Phase 4 — loop done; dataset/augment/val planned
    trainer.py         # ✅ train_one_epoch + Trainer, param groups, checkpointing
    ema.py             # ✅ ModelEMA (weight EMA)
    scheduler.py       # ✅ LinearWarmup + flat-cosine/multistep LR
    logger.py          # ✅ MetricLogger/SmoothedValue — console progress readout
    visualizer.py      # ✅ TrainingVisualizer — TensorBoard + loss_curve.png (+ W&B)
    dataset.py         # ✅ CocoDetection + build_coco_dataloader (multi-scale collate)
    augment.py         # ⬜ RandomPhotometricDistort, ZoomOut, IoUCrop, MultiScale...
  export/              # ⬜ Phase 3
    onnx.py            #   ONNX export (+ optional onnxsim); TRT/OpenVINO helpers
  cli.py               # ✅ `dfine models`; predict/train/val/export are phase stubs
docs/                  # ARCHITECTURE.md, CONFIG_REFERENCE.md, ROADMAP.md
tests/                 # parity + unit tests
```

## 5. Architecture in one paragraph

D-FINE = HGNetV2 backbone → HybridEncoder (AIFI + CCFM/GELAN fusion) → DFINETransformer
decoder. The decoder's novelty is **FDR**: box regression is a distribution over
`reg_max` bins per edge, refined residually across decoder layers with a non-uniform
weighting function (`reg_scale`, `up`/`down` bounds). **GO-LSD** distills the final
layer's distribution into earlier layers (DDF loss). See `docs/ARCHITECTURE.md` for
the full data flow and the module→param map.

## 6. Conventions

- Python ≥3.9, `torch`/`torchvision`. Type hints everywhere. `from __future__ import annotations`.
- Formatting: `ruff format` + `ruff check`. Line length 100.
- Docstrings: short one-liner per public function/param (Google-ish).
- No global state; device is explicit. No prints in library code — use `logging`.
- Public names are stable contracts; don't rename params casually.
- Keep functions small; a reviewer should grasp each in <30s.
- Every new module ships with a test in `tests/`.

## 7. How to work (loop)

1. Read `docs/ROADMAP.md`, pick the **lowest unchecked task in the current phase**.
2. Read the relevant section of `docs/ARCHITECTURE.md` + `docs/CONFIG_REFERENCE.md`.
3. If porting from upstream, fetch the exact upstream module and match shapes/names.
4. Write the code + a test. Run tests. Iterate until green.
5. Update `docs/CONFIG_REFERENCE.md` if you added/changed a param.
6. Check the box in `docs/ROADMAP.md` and note anything surprising.
7. Keep diffs focused — one task per change set.

## 8. Commands

```bash
pip install -e ".[dev]"        # editable install with dev extras (incl. torch)
ruff format . && ruff check .  # format + lint
pytest -q                      # run tests
pytest -q -k parity            # weight/output parity tests only
dfine models                   # sanity: list presets
```

Optional extras (on top of core): `dfine[torch]` (inference: torch/torchvision/
pillow), `dfine[video]` (`predict_video`: OpenCV), `dfine[train]` (scipy matcher +
COCO eval), `dfine[export]` (ONNX). `[dev]` pulls torch + headless OpenCV + scipy so
the whole suite runs. Base `pip install dfine` (config/CLI only) needs no torch.

## 9. Definition of Done (per task)

- [ ] Code typed, formatted, lint-clean.
- [ ] Test added and passing (shape/parity/behavior as appropriate).
- [ ] No YAML on the user's critical path; no registry/`create()` calls.
- [ ] Public param names match `docs/CONFIG_REFERENCE.md`.
- [ ] For model modules: loads matching upstream weights OR has a documented remap.
- [ ] Roadmap checkbox ticked.

## 10. Guardrails (do NOT)

- Do **not** reintroduce YAML configs or the upstream registry into the public path.
- Do **not** change public parameter names/semantics to match an internal backend;
  adapt the backend instead.
- Do **not** fabricate default values — verify against upstream `D-FINE/src/` or its
  `configs/*.yml`. If unsure, mark `# TODO(verify)` and open a roadmap note.
- Do **not** vendor code without keeping its Apache-2.0 `LICENSE` + attribution.
- Do **not** hard-code paths, secrets, or a specific CUDA device.

## 11. Provenance / references

- Upstream: https://github.com/Peterande/D-FINE (Apache-2.0)
- Paper: https://arxiv.org/abs/2410.13842 (hyperparameter tables in appendix)
- HF port (flat config, no YAML): https://huggingface.co/docs/transformers/en/model_doc/d_fine
- Related wrappers for reference: ArgoHA/D-FINE-seg, Intellindust-AI-Lab/DEIM(v2)
