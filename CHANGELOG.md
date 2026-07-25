# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is `0.x`, minor/patch boundaries are best-effort and the public API may still shift.

## [Unreleased]

### Added

- **Segmentation** — instance and semantic segmentation behind the same `DFINE(task=...)`
  façade: native mask losses + mask costs in the Hungarian matcher, `SemSegCriterion`
  (CE + soft Dice + `ignore_index`), YOLO-style seg datasets with train/val split, ONNX
  export for `segment`/`sem_seg`, and mask-AP / mIoU validation. Numeric-parity-tested
  against [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg).
- **Validation analytics** (`model.val(..., plots=True)`) — confusion matrix, per-class AP,
  P/R/F1-vs-confidence curves with a best-confidence recommendation, and a worst-predictions
  (FP/FN) gallery. New [`docs/api/validation.md`](docs/api/validation.md) page.
- **TorchScript export** — `model.export(format="torchscript")` alongside ONNX.
- **Results ergonomics** — `save_txt()` (YOLO labels), `save_crop()` (per-detection crops),
  `summary()` / `tojson()`, `verbose()` per-class summary, COCO-RLE masks in `to_coco()`, and
  auto-incremented run directories (`--project`/`--name`/`--save-*`).
- **Model utilities** — `DFINE.info()` (`dfine info`) model summary and `DFINE.benchmark()`
  (`dfine benchmark`) inference-speed measurement.
- **Predict over a directory or glob** source.
- **Resume training** — `DFINE.train(resume=...)` continues an interrupted run; periodic
  checkpoints + tqdm progress bar.
- Community health files: `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and more
  README badges.

### Fixed

- `ConfusionMatrix` IoU-matching threshold aligned to `0.5`, consistent with the P/R/F1 and
  worst-predictions accumulators (COCO numeric metrics unaffected).
- Fixed a same-stem filename collision in the save plumbing; guarded `benchmark`.

### Changed

- Test suite is warning-free: unclosed file handles closed, and torch's own legacy
  TorchScript-based ONNX export deprecations / tracer warnings are filtered narrowly so
  genuinely new warnings still surface.

## [0.0.1] - 2026-07-18

Initial public release on [PyPI](https://pypi.org/project/pydfine/) — feature-complete
across roadmap phases 0–6.

### Added

- **Config-first core** — `DFINEConfig`, a frozen dataclass exposing every model/training
  parameter as a typed field, with `preset(size, **overrides)` for `n/s/m/l/x`, validation,
  and optional YAML interop.
- **Native model port (Path A)** — `HGNetv2`, `HybridEncoder`, and `DFINETransformer`
  (FDR head, LQE, contrastive denoising) ported from upstream `src/` with the YAML/registry
  layer stripped and layer/param names preserved. **Bit-exact parity** with upstream `.pth`
  across all sizes (`max|Δ| = 0`).
- **Inference** — the one-class public API `DFINE(...).predict(...) -> list[Results]`
  (`.boxes.xyxy/.conf/.cls`, `.plot()/.save()`), `from_pretrained`, and video inference.
- **Training** — the ported D-FINE loop (AdamW param groups, EMA, AMP, grad clip,
  warmup + flat-cosine LR), COCO datasets + full augmentation pipeline, live console +
  TensorBoard visualization, and single-kwarg **multi-GPU DDP** (`devices=N`).
- **COCO validation** — `model.val()` returns the 12 named COCO metrics, also run each epoch
  during training.
- **ONNX export** — dynamic-batch graph via `model.export(format="onnx")`, with TensorRT /
  OpenVINO downstream notes.
- **Tracking** — optional ByteTrack tracker on `predict_video`.
- **Results interop** — `to_supervision()`, `to_coco()`, `to_pandas()`.
- **YOLO → COCO converter** — `dfine convert` / `dfine.yolo_to_coco(...)`.
- **CLI** — `dfine models/predict/val/train/export/convert`.
- **Docs site** (MkDocs Material) and API reference.

[Unreleased]: https://github.com/HoshiBatista/pydfine/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/HoshiBatista/pydfine/releases/tag/v0.0.1
