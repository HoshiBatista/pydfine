# Roadmap

Phased, checkbox task plan. Agents: work the **lowest unchecked task in the active
phase**, one task per change set. Tick boxes as you go and leave a dated note if you
discover something that changes later phases.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Phase 0 — Project scaffolding
- [x] Package skeleton (`dfine/`, `pyproject.toml`, CLI `dfine models`).
- [ ] `Results`/`Boxes`, weight download/cache, input loading, COCO names. *(moved to
      Phase 2 — these need the backend; only `registry.py` exists so far.)*
- [x] Agent docs: `README`, `AGENTS.md`, `CLAUDE.md`, `docs/*`.
- [x] Add `dev` extras (`ruff`, `pytest`) + `ruff`/`pytest` config in `pyproject.toml`.
- [x] Add `tests/` with a smoke test.
- [x] CI workflow (`.github/workflows/ci.yml`) running `ruff` + `pytest` (py3.9–3.13).
- [x] Repo tooling: `.pre-commit-config.yaml`, PR/issue templates, `dependabot.yml`,
      `CONTRIBUTING.md`.

## Phase 1 — Config-first core (the headline feature)
- [x] Implement `dfine/config.py`: `DFINEConfig` frozen dataclass with **every** field
      from `docs/CONFIG_REFERENCE.md`, full type hints + one-line docstrings.
- [x] `DFINEConfig.preset(size, **overrides)` + `SIZE_PRESETS` table (§11 of reference).
- [x] Config validation (`__post_init__`): ranges, list lengths, cross-field checks
      (e.g. `len(in_channels)==len(feat_strides)==num_levels`).
- [ ] `DFINEConfig.from_yaml()/.to_yaml()` interop (optional path only).
- [x] Tests: preset field values match reference; validation rejects bad configs.

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
- ~~Backend default is Path B (transformers) until Phase 5 parity lands.~~
- **2026-07-11 — DECISION: go Path A (native port) directly**, per repo owner. We
  port the needed modules out of `D-FINE/src/` and rewrite them YAML/registry-free,
  wired from `DFINEConfig`. `transformers` is therefore **not** a dependency (upstream
  `src/` never imports it). Phase 5 (native backend) is effectively pulled forward to
  become the primary work; the `backends/` boundary is kept so a wrapper could still
  be added later. Deps split into `requirements*.txt`: core = torch/torchvision/
  numpy/pillow; train adds scipy (matcher) + faster-coco-eval (val).
- **2026-07-11** — Implemented `config.py`. Preset values were verified field-by-field
  against `D-FINE/configs/dfine/*.yml` (not the paper tables). Corrections vs the old
  `CONFIG_REFERENCE.md` §11, now applied there too:
  - **N is a 2-level model**: `num_levels=2`, `hidden_dim=128`, `in_channels=[512,1024]`,
    `feat_strides=[16,32]`, `use_encoder_idx=[1]`, `encoder_dim_feedforward=512`,
    `feat_channels=[128,128]`, `num_points=[6,6]`, `return_idx=[2,3]`. (Was listed as
    3-level/256.)
  - **M** uses `use_lab=True`, `in_channels=[384,768,1536]`, `depth_mult=0.67`.
  - Backbone LRs: N 4e-4, S 1e-4, M 2e-5, L 1.25e-5, X 2.5e-6. Epochs: N 160, S/M 132,
    L/X 80. Loss weights: fgl 0.15, ddf 1.5. Matcher alpha 0.25 (criterion alpha 0.75).
  - Docs were at repo root but every link points to `docs/*`; moved the three specs
    into `docs/`. README's "inference-core scaffold" claim was aspirational — no
    `dfine/` existed; scaffold is now real but Results/Boxes/downloads remain Phase 2.
