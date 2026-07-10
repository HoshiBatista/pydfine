# Roadmap

Phased, checkbox task plan. Agents: work the **lowest unchecked task in the active
phase**, one task per change set. Tick boxes as you go and leave a dated note if you
discover something that changes later phases.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Phase 0 — Project scaffolding
- [x] Package skeleton (`dfine/`, `pyproject.toml`, CLI).
- [x] `Results`/`Boxes`, weight download/cache, input loading, COCO names.
- [x] Agent docs: `README`, `AGENTS.md`, `CLAUDE.md`, `docs/*`.
- [ ] Add `dev` extras (`ruff`, `pytest`) + `ruff`/`pytest` config in `pyproject.toml`.
- [ ] Add `tests/` with a smoke test and CI workflow.

## Phase 1 — Config-first core (the headline feature)
- [ ] Implement `dfine/config.py`: `DFINEConfig` frozen dataclass with **every** field
      from `docs/CONFIG_REFERENCE.md`, full type hints + one-line docstrings.
- [ ] `DFINEConfig.preset(size, **overrides)` + `SIZE_PRESETS` table (§11 of reference).
- [ ] Config validation (`__post_init__`): ranges, list lengths, cross-field checks
      (e.g. `len(in_channels)==len(feat_strides)==num_levels`).
- [ ] `DFINEConfig.from_yaml()/.to_yaml()` interop (optional path only).
- [ ] Tests: preset field values match reference; validation rejects bad configs.

## Phase 2 — Backend abstraction + working inference
- [ ] `dfine/backends/__init__.py`: `Backend` protocol (`build`, `predict`, `state_dict`,
      `load_state_dict`, `to_deploy`) + `get_backend(config)`.
- [ ] `dfine/backends/transformers.py` (**Path B**): build `DFineForObjectDetection`
      from `DFINEConfig`, load pretrained by name, expose the deploy contract from
      `docs/ARCHITECTURE.md` §3.
- [ ] Rewrite `DFINE` (`model.py`) to build via `get_backend(config)` — remove the
      YAML/`YAMLConfig` path from the public flow.
- [ ] `DFINE.predict()/__call__` returns `Results`; batched; conf filter.
- [ ] `DFINE.predict_video()`.
- [ ] Parity test: `DFINE(size="s").load("dfine-s")` on a sample image ≈ upstream boxes.

## Phase 3 — Export
- [ ] `dfine/export/onnx.py`: dynamic-batch ONNX with `(images, orig_target_sizes)`
      signature + optional `onnxsim`.
- [ ] `DFINE.export(format="onnx")`; smoke test that onnxruntime runs the graph.
- [ ] Helpers/docs for TensorRT (`trtexec --fp16`) and OpenVINO downstream.

## Phase 4 — Training
- [ ] `train/dataset.py`: COCO-format dataset + dataloader (`remap_mscoco_category`).
- [ ] `train/augment.py`: PhotometricDistort, ZoomOut, IoUCrop, HFlip, MultiScale;
      two-phase schedule (advanced → `no_aug_epoch` tail).
- [ ] `train/trainer.py`: AdamW param groups (backbone vs norm/bias), EMA, AMP,
      grad clip, warmup + flat-cosine scheduler, checkpointing/resume.
- [ ] `DFINE.train(data=..., epochs=..., imgsz=..., batch=...)` single-GPU path.
- [ ] Multi-GPU launch (wrap `torchrun`) behind the same `.train()` call.
- [ ] `DFINE.val()` via COCO evaluator → returns metrics dict.
- [ ] Overfit-one-batch test (loss → ~0) as a training smoke test.

## Phase 5 — Native backend (Path A) — optional, for full ownership/parity
- [ ] Port `HGNetv2` into `dfine/backends/native/backbone.py` (strip registry).
- [ ] Port `HybridEncoder`.
- [ ] Port `DFINETransformer` (+ FDR head, LQE, denoising).
- [ ] Port `HungarianMatcher` + `DFINECriterion` (VFL/L1/GIoU/FGL/DDF).
- [ ] Port `DFINEPostProcessor`.
- [ ] Weight-remap loader: upstream `.pth` → native modules; parity test per size.
- [ ] Make native the default backend once parity holds across n/s/m/l/x.

## Phase 6 — Polish
- [ ] `Results` interop: `to_supervision()`, `to_coco()`, `to_pandas()`.
- [ ] Optional ByteTrack tracker on `predict_video`.
- [ ] Docs site / API reference generation.
- [ ] Publish to PyPI (choose final package name; update imports).

---

## Notes / decisions log
- (add dated notes here as you learn things that affect later phases)
- Backend default is **Path B (transformers)** until Phase 5 parity lands; keep the
  public `DFINE(...)` API identical across backends.
