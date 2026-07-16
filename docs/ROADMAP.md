# Roadmap

Phased, checkbox task plan. Agents: work the **lowest unchecked task in the active
phase**, one task per change set. Tick boxes as you go and leave a dated note if you
discover something that changes later phases.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Phase 0 — Project scaffolding
- [x] Package skeleton (`dfine/`, `pyproject.toml`, CLI `dfine models`).
- [x] `Results`/`Boxes`, weight download/cache, input loading, COCO names. *(done in
      Phase 2: `results.py`, `downloads.py`, `model.py`.)*
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

## Phase 2 — Assembled model + working inference (native, Path A)
Note: much of the native port is done under Phase 5 already. This phase wires the
ported modules into one model behind the public API.
- [x] `native/postprocessor.py`: port `DFINEPostProcessor` (top-k decode to xyxy in
      original scale). Done under Phase 5.
- [x] `native/dfine.py`: assemble backbone+encoder+decoder into one `nn.Module`
      (`from_config`, `forward = decoder(encoder(backbone(x)))`, `.deploy()`,
      `.load()`). Postproc stays separate (matches upstream). 674 params for N.
- [x] Weight-remap loader (`native/loader.py`): `load_checkpoint` /
      `extract_state_dict` unwrap upstream `.pth` (EMA-preferred, `module.`-strip)
      and `strict=True` load — no key remap needed (names preserved).
- [x] `dfine/model.py` — the public `DFINE` class; `predict()/__call__` returns
      `Results`; batched; conf filter; input loading (path/PIL/ndarray/list) +
      `Resize(imgsz)+ToTensor` preprocessing (matches upstream `torch_inf.py`).
      Config-first ctor (`DFINE(size=..., **overrides)`), `.load(name|path)`,
      `.from_pretrained(name)`, device auto-select; `train/val/export/predict_video`
      are phase-stubbed. Exposed lazily from `dfine/__init__.py` (base import stays
      torch-free).
- [x] `Results`/`Boxes` (`.boxes.xyxy/.conf/.cls`, `.plot()/.save()`, `__len__`,
      iterate) in `dfine/results.py`. Weight download/cache = `dfine/downloads.py`.
- [x] `DFINE.predict_video()` — frame-by-frame detect over a video (OpenCV);
      writes an annotated mp4 (orig res/fps) or `stream=True` yields per-frame
      `Results`. Lazy `cv2` import (`dfine[video]` extra); real round-trip + stream
      + missing-cv2 + bad-source tests (headless opencv in dev deps so CI runs them).
- [x] Parity test: native port reproduces genuine upstream output. `test_parity.py`
      (gated on `DFINE_WEIGHTS_DIR`) compares our model+postprocessor against a
      committed fixture generated from real upstream (`scripts/gen_parity_fixture.py`)
      on a deterministic seeded input. **Bit-exact (max|Δ|=0.0)** across all 5 sizes
      (n/s/m/l/x) for raw boxes, final boxes, scores, and labels.

## Phase 3 — Export
- [ ] `dfine/export/onnx.py`: dynamic-batch ONNX with `(images, orig_target_sizes)`
      signature + optional `onnxsim`.
- [ ] `DFINE.export(format="onnx")`; smoke test that onnxruntime runs the graph.
- [ ] Helpers/docs for TensorRT (`trtexec --fp16`) and OpenVINO downstream.

## Phase 4 — Training
- [x] `train/trainer.py`: the D-FINE loop ported single-process — `build_param_groups`
      (backbone vs enc/dec-norm regex, verbatim from `optimizer.yml`), `train_one_epoch`
      (AMP, `sum(loss_dict)` backward, grad clip, EMA, per-iter warmup), and a `Trainer`
      that wires model+criterion+optimizer+scheduler+EMA and runs `.fit()` with
      per-epoch `last.pth` checkpointing. `train/ema.py` (`ModelEMA`), `train/scheduler.py`
      (`LinearWarmup` + flat-cosine/multistep).
- [x] **Training-process visualization (like upstream D-FINE):** `train/logger.py`
      (`MetricLogger`/`SmoothedValue` — the console `Epoch [i/N] eta … loss … lr … mem`
      readout) + `train/visualizer.py` (`TrainingVisualizer`: TensorBoard scalars
      `Loss/*`,`Lr/*`,`Test/*` + a matplotlib `loss_curve.png` + optional W&B, all
      optional/graceful). Added `matplotlib` to the `[train]` extra.
- [x] `DFINE.train(train_loader, epochs=…, output_dir=…, val_fn=…)` single-GPU path
      (loader-based). Verified end-to-end: writes `last.pth` + `loss_curve.png` + TB events.
- [x] Overfit-one-batch test (loss drops sharply) + param-group/EMA/warmup/scheduler/logger
      units — `tests/test_trainer.py` (green).
- [x] `train/dataset.py`: COCO-format dataset + dataloader ported from upstream
      `src/data` (detection path). `CocoDetection` (on `faster_coco_eval`'s parser) +
      `_PrepareCocoTarget` (xywh→xyxy clamp/keep, contiguous-label remap), a minimal
      resize+tensor+`cxcywh`-normalize `default_transforms`, `BatchImageCollateFunction`
      (multi-scale jitter, epoch-gated) + a `set_epoch`-forwarding dataloader, and the
      config-first `build_coco_dataloader(img_folder, ann_file, cfg=…)`. Yields
      `(images, targets)` consumable directly by `DFINE.train`. Tested against a
      synthetic on-disk COCO set (output contract + remap + multiscale + one train step);
      `faster-coco-eval` added to the `[dev]` extra so CI runs it.
- [x] `DFINE.train(data="path/to/coco", …)` sugar: `dataset.build_coco_dataloaders`
      resolves the standard COCO root layout (`train2017/` + `annotations/
      instances_train2017.json`, optional `val2017/`) into a train loader (two-phase
      augmentation + multi-scale) and, when present, a val loader; `DFINE.train` calls
      it when `data=` is given (mutually exclusive with `train_loader=`; `batch_size`/
      `num_workers`/`augment`/`remap_mscoco_category` tune the build). Tested: split
      resolution, val auto-discovery/absence, augmented + no-aug builds, missing-root/
      -split errors (`test_dataset.py`), and a real 1-epoch `train(data=…)` writing
      `last.pth` (`test_model.py`).
- [x] `train/augment.py`: ported D-FINE's train pipeline — RandomPhotometricDistort,
      RandomZoomOut, RandomIoUCrop (with a `p` wrapper), SanitizeBoundingBoxes,
      RandomHorizontalFlip, Resize, then the shared tensor/`cxcywh`-normalize tail. The
      two-phase schedule is `TrainCompose` + `stop_epoch`: the advanced ops
      (photometric/zoomout/IoU-crop, `ADVANCED_OPS`) switch off once
      `epoch >= stop_epoch` (pass `cfg.epochs - cfg.no_aug_epoch`). `set_epoch` is
      forwarded loader→dataset→compose. Plug in via
      `build_coco_dataloader(transforms=train_transforms(imgsz, stop_epoch=…))`.
      Tested: output contract, stop-epoch skip logic, epoch forwarding, and an
      augmented train step (multi-scale collate is Phase-4's `dataset.py`, done).
- [x] Multi-GPU launch (wrap `torchrun`) behind the same `.train()` call. New
      `train/distributed.py` ports the single-node pieces of upstream `dist_utils`
      (rank/world-size, group setup/teardown, DDP+SyncBN model wrap, `DistributedSampler`
      loader wrap, and a `spawn` that launches one worker per GPU). The `Trainer` is now
      DDP-aware (optimizer/EMA/checkpoints target the de-paralleled module; only rank 0
      saves/logs; loaders sharded; val runs on all ranks). `DFINE.train(devices=N)` is
      the launcher (spawns workers, each rebuilds from `config`+`data`, rank 0's weights
      reloaded after); a `torchrun --nproc_per_node=N` script that calls `train(...)`
      also works (each worker joins the existing group). Added `sync_bn`/
      `find_unused_parameters` config fields (upstream defaults True/False). Tested:
      no-group defaults + 1-process gloo group (DDP/sampler wrap) always-on, plus a
      gated 2-process CPU/gloo spawn end-to-end (`DFINE_TEST_MULTIGPU=1`).
- [x] `DFINE.val()` via COCO evaluator → returns metrics dict (slots into the existing
      `Trainer.fit(val_fn=…)` hook). `train/evaluator.py` ports upstream
      `det_engine.evaluate` (single-process): runs the model over a COCO val loader,
      decodes with the postprocessor, scores against the loader's ground-truth `.coco`
      via `faster-coco-eval`, and returns the 12 named COCO stats (`COCO_STAT_NAMES`;
      `AP` = primary mAP). `coco_val_fn(postprocessor, device)` wraps it as the
      `Trainer.fit(val_fn=…)` closure; `DFINE.train` now auto-wires it whenever a val
      loader is present and no `val_fn` was passed. `DFINE.val(data=… | val_loader=…)`
      + `build_coco_val_dataloader` (val-only loader from a COCO root). Tested:
      perfect-prediction replay → AP==1.0, train-mode restore, the closure, the
      non-COCO-loader error (`test_evaluator.py`), and `DFINE.val(data=…)` + val-during-
      train (`test_model.py`).

## Phase 5 — Native backend (Path A) — **primary path** (decision 2026-07-11)
- [x] Port `HGNetv2` into `dfine/backends/native/hgnetv2.py` (strip registry; +
      `from_config`, name normalization, `out_channels`/`out_strides`; `common.py`
      holds `FrozenBatchNorm2d`). Shape tests green for all presets.
- [x] Port `HybridEncoder` into `dfine/backends/native/hybrid_encoder.py` (AIFI +
      CCFM/GELAN; `get_activation` moved to `ops.py`; `+ from_config`). Shape +
      backbone→encoder integration tests green for all presets.
- [x] Port `DFINETransformer` (+ FDR head, LQE, denoising) into
      `dfine/backends/native/dfine_decoder.py`, with `box_ops.py`, `dfine_utils.py`,
      `denoising.py`, and extended `ops.py` (inverse_sigmoid, bias_init,
      deformable_attention_core_func_v2). Added `decoder_dim_feedforward` config field
      (512 for N, else 1024). Shape + full backbone→encoder→decoder pipeline tests green.
- [x] Port `HungarianMatcher` + `DFINECriterion` (VFL/L1/GIoU/FGL/DDF) into
      `native/matcher.py` + `native/criterion.py` (+ `native/dist.py` shim; registry/
      `src.core` stripped; `from_config` added). scipy imported lazily (train-only).
      Tested: matcher 1-to-1 + top-k; criterion end-to-end on the real train-mode
      decoder output — finite loss dict (final+aux+enc+pre+dn terms) that backprops
      to the decoder; also the `num_denoising=0` path.
- [x] Port `DFINEPostProcessor` into `dfine/backends/native/postprocessor.py`
      (registry/`src.core` stripped; `+ from_config`). Added `coco.py` with the
      MS-COCO category maps for the `remap_mscoco_category` branch. Decode +
      full pipeline + deploy-mode tests green.
- [x] Weight-remap loader: upstream `.pth` → native modules (`native/loader.py` +
      assembled `native/dfine.py`). Offline round-trip test + opt-in real-`.pth`
      strict-parity test (`DFINE_TEST_CKPT`/`DFINE_TEST_SIZE`). Verified against a
      real released-format N checkpoint: strict load, 0 missing/0 unexpected.
- [x] Per-size parity across n/s/m/l/x. Catalogue (`registry.py`) now carries
      `num_classes` per checkpoint + `resolve_weights(size, dataset)` /
      `config_for()` "which model to use" logic; `downloads.py` caches assets;
      `DFINE.from_pretrained(name)` ties it together. Parametrized parity test
      (`test_per_size_coco_parity`, gated on `DFINE_WEIGHTS_DIR`) now runs green
      against all 5 downloaded COCO `.pth` (n/s/m/l/x): 0 missing / 0 unexpected,
      finite forward pass. Uncovered + fixed an X-only arch bug — see 2026-07-14 note.
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
- **2026-07-12** — Ported `DFINEPostProcessor` (native). `use_focal_loss` isn't a
  config field: D-FINE fixes it True globally, so `from_config` hard-codes True.
  Added `backends/native/coco.py` (COCO id/name maps) for the `remap_mscoco_category`
  branch; it's imported lazily so non-COCO models never touch it. The postprocessor
  does **not** clamp boxes to the frame (upstream doesn't either) — xyxy corners can
  fall outside `[0, size]`.
- **2026-07-12** — Assembled model + weight loader. `native/dfine.py` mirrors upstream
  (`backbone`/`encoder`/`decoder` attr names → checkpoints load with no remap);
  postproc stays a separate module. `native/loader.py` unwraps upstream `.pth`
  (prefers `ema.module`, strips `module.`) and does a `strict=True` load. **Gotcha:**
  the decoder registers `anchors`/`valid_mask` as *persistent* buffers sized to
  `eval_spatial_size`, so `imgsz` must match the checkpoint's train resolution (640,
  the preset default = all official COCO releases) or strict load fails on those two
  buffers. Parity proven offline against a real released-format N `.pth` (0 missing/0
  unexpected).
- **2026-07-12** — Checkpoint "which model to use" logic. `registry.py` now maps each
  released asset to a `CheckpointSpec(size, dataset, num_classes, filename, url)`.
  Three dataset variants: `coco`/`obj2coco` → 80 classes, `obj365` → 366 (the *only*
  dataset-dependent arch diff; reg_scale=8 is X-only and already in that preset).
  **Availability is not uniform: N is COCO-only** — upstream released no obj2coco/obj365
  for N, so `resolve_weights("n", "obj365")` raises listing what N *does* have.
  `config_for()` builds the matching config (wires obj365's 366-class head);
  `downloads.py` caches assets (`$DFINE_CACHE_DIR`, atomic .part rename);
  `DFINE.from_pretrained(name, **overrides)` = resolve→config→download→strict-load
  (sets `backbone_pretrained=False` so the ImageNet backbone isn't fetched then
  overwritten). Per-size COCO parity test is parametrized + gated on `DFINE_WEIGHTS_DIR`;
  run it with the 5 downloaded COCO `.pth` to close the parity tick.
- **2026-07-12** — Public API landed (`model.py` + `results.py`), the headline
  `DFINE(...)` façade from the README. `DFINE(size=..., **overrides)` is config-first
  (device is a runtime kwarg, not a config field); `.predict(source, conf, imgsz)`
  loads path/PIL/ndarray/list → `Resize(imgsz)+ToTensor` (no mean/std norm, matches
  upstream `torch_inf.py`) → native model → postprocessor → `list[Results]`. Boxes
  come back in original pixel scale (postprocessor already rescales). `.load()` takes a
  catalogue name **or** a local path. Names default to COCO-80 when `num_classes==80`
  and no `class_names`. Public symbols are lazy-loaded in `__init__.py` so a bare
  `import dfine` stays torch-free. Next open Phase-2 items: `predict_video` + the
  sample-image upstream-boxes parity test.
- **2026-07-12** — `predict_video` done (OpenCV, lazy import, `dfine[video]` extra);
  writes annotated mp4 or `stream=True` yields per-frame `Results`.
- **2026-07-12** — Loss ported: `native/matcher.py` (`HungarianMatcher`) +
  `native/criterion.py` (`DFINECriterion`) + `native/dist.py` (single-process
  `get_world_size`/`is_dist_available_and_initialized` shim). `from_config` uses
  upstream's fixed `losses=['vfl','boxes','local']`, `boxes_weight_format=None`
  (VFL computes its own IoU), matcher costs 2/5/2 (α=0.25), criterion α=0.75.
  The criterion consumes the decoder's **training-mode** dict as-is (needs
  `.train()`: `pred_corners`/`ref_points`/`up`/`reg_scale` + `aux/enc/pre/dn`
  outputs) — verified it backprops to the decoder. `scipy` is train-only, imported
  lazily inside the matcher. Two unused upstream helpers (`feature_loss_function`,
  `get_gradual_steps`) were not ported (not on the loss path). Next Phase-4: dataset/
  dataloader → augment → trainer (`.train()`), then `.val()` (COCO eval).
- **2026-07-14** — Per-size parity tick closed. Downloaded the 5 COCO `.pth`
  (n/s/m/l/x) and ran `test_per_size_coco_parity` — all strict-load with 0 missing /
  0 unexpected. This surfaced an **X-only architecture bug**: X sets the *encoder*
  `hidden_dim=384` but upstream leaves the *decoder* `DFINETransformer.hidden_dim` at
  the base **256** (only `feat_channels` becomes `[384]*3`, which the decoder's
  `input_proj` maps 384→256). Our config used one `hidden_dim` for both, so X built its
  `dec_bbox_head`/heads at 384 and the strict load failed on shape mismatch. Fix: added
  a `decoder_hidden_dim` config field (defaults to `hidden_dim` via `__post_init__`;
  set to 256 in the X preset); the decoder `from_config` now reads
  `cfg.decoder_hidden_dim`, the encoder keeps `cfg.hidden_dim`. n/s/m/l unaffected
  (encoder==decoder dim there). Parity tests are gated on `DFINE_WEIGHTS_DIR`, so CI
  without weights stays green.
- **2026-07-14** — Numeric parity vs genuine upstream (Phase 2 tick). Ran the real
  `D-FINE/src` model (via `YAMLConfig`) on a deterministic seeded input and saved a
  compact fixture per size (`tests/data/parity_<size>.pt`, ~16 KB: raw pred_boxes +
  final labels/boxes/scores; raw pred_logits dropped since final labels/scores are
  argmax/sigmoid+topk over them). `tests/test_parity.py` builds our native port from
  the same COCO `.pth` and asserts a match — **bit-exact, max|Δ|=0.0 for all of
  n/s/m/l/x**. Two setup notes: (1) upstream's COCO YAML turns on
  `remap_mscoco_category` (1..90 ids); our library standardizes on contiguous 0..79
  labels + separate name mapping, so the generator forces upstream's remap off for an
  apples-to-apples compare. (2) Upstream's full stack (tensorboard/transformers/
  calflops via `profiler_utils`) is needed only to *generate* the fixture — the test
  imports none of it; `transformers`/`calflops` were uninstalled from the dev venv
  after (they are not deps; `tensorboard`/`faster-coco-eval` are legit train extras).
  This closes both remaining parity items — the port is proven, not just asserted.
- **2026-07-15** — Ported the Phase-4 training loop into `dfine/train/` and cross-checked
  it against upstream `D-FINE/src`. Verified faithful: the AdamW param grouping is
  bit-exact vs upstream `get_optim_params` (replayed its regex — 186 backbone / 98
  zero-WD enc·dec-norm / 144 default, disjoint, all covered); `train_one_epoch`,
  `ModelEMA` (decay ramp + update rule), and `LinearWarmup` (factor formula) match
  `det_engine`/`optim` numerically. Single-process simplifications only: dropped
  `dist_utils` all-reduce/`de_parallel`/`reduce_dict` (no-ops at world-size 1); AMP uses
  `device_type=device.type` (more correct than upstream's `str(device)`).
  **INTENTIONAL DEVIATION (kept, per repo owner):** the default `scheduler="flatcosine"`
  adds a cosine decay over the trailing `no_aug_epoch` epochs. Upstream configures
  `MultiStepLR(milestones=[500])`, which never fires (all recipes are 72–160 epochs) —
  i.e. upstream's LR is effectively flat with no annealing. Our flat body matches; the
  cosine no-aug tail is an added enhancement, not parity. For an exact upstream schedule
  use `scheduler="multistep"` with a milestone beyond `epochs`. Documented in
  `dfine/train/scheduler.py` + the `scheduler` field in `dfine/config.py`.
- **2026-07-16** — `DFINE.train(data="coco/")` path sugar landed. New
  `train/dataset.py::build_coco_dataloaders(data_root, …)` resolves the standard
  MS-COCO layout (`train2017/`, `annotations/instances_train2017.json`, optional
  `val2017/`+`instances_val2017.json`; split names overridable) into
  `(train_loader, val_loader)`. The train loader gets the full two-phase augmentation
  (`augment.train_transforms`, `stop_epoch = cfg.epochs − cfg.no_aug_epoch`) + the
  existing multi-scale collate; the val loader is plain-resize and is `None` when no
  val split is on disk. `DFINE.train` now takes `data=` (mutually exclusive with
  `train_loader=`, raises if both/neither) plus `batch_size`/`num_workers`/`augment`/
  `remap_mscoco_category` passthroughs; auto-built val loader fills `val_loader` when
  not supplied. `build_coco_dataloaders` is imported lazily inside `train()` (keeps
  `faster-coco-eval` off the base train import). Remaining Phase-4: `.val()` (COCO eval)
  + multi-GPU.
- **2026-07-16** — `DFINE.val()` landed (COCO eval). New `train/evaluator.py::evaluate`
  ports upstream `det_engine.evaluate` down to the single-process detection path: eval
  the model over a COCO val loader, decode with the postprocessor, score against the
  loader's ground-truth `.coco` with `faster-coco-eval` (the same evaluator upstream
  wraps), and return the classic 12-element COCO summary as a named `dict[str, float]`
  (`COCO_STAT_NAMES`; `AP` = mAP@[.50:.95]). `coco_val_fn(postprocessor, device)` is the
  `Trainer.fit(val_fn=…)` closure; `DFINE.train` auto-wires it when a val loader exists
  and no `val_fn` is passed (so `train(data="coco/")` validates each epoch and logs
  `Test/*` + the mAP curve). `DFINE.val(data=… | val_loader=…)` builds a val-only loader
  via new `dataset.build_coco_val_dataloader`. **Label-space gotcha:** the
  postprocessor's `remap_mscoco_category` (from `cfg`) decides whether predicted labels
  are contiguous `0..N-1` or sparse MS-COCO ids, and they must match the GT JSON's
  `category_id`s — stock MS-COCO GT is sparse, so build the model with
  `remap_mscoco_category=True` to score it. Fixed the visualizer's AP key (`AP50:95`
  placeholder → `AP`). Remaining Phase-4: multi-GPU only.
- **2026-07-16** — Multi-GPU training landed — **Phase 4 is complete.** New
  `train/distributed.py` ports upstream `dist_utils` (single-node, torchrun-free):
  `setup_distributed`/`cleanup_distributed` (env-driven, `nccl`|`gloo`), rank/world-size
  queries, `wrap_model_ddp` (DDP + SyncBN, SyncBN GPU-only), `wrap_loader_distributed`
  (`DistributedSampler`, forwards `set_epoch`), and `spawn` (mp.spawn, one proc/GPU, auto
  MASTER_ADDR/PORT). `Trainer` is DDP-aware: it keeps `self.module` (raw) vs `self.model`
  (DDP) — optimizer/EMA/checkpoints/param-groups use the de-paralleled module, EMA
  unwraps DDP in `update`, loaders are sharded in `fit`, val runs on all ranks
  (faster-coco-eval gathers shards), and only rank 0 writes checkpoints/visualizer.
  `DFINE.train(devices=N)` is the launcher: snapshot weights → `spawn` N workers (each
  rebuilds `DFINE(config=…)`, loads the snapshot, trains under DDP) → reload rank 0's
  `last.pth`; requires `data=` (loaders can't cross `spawn`). A `torchrun
  --nproc_per_node=N` script calling `train(...)` also works — `launched_via_torchrun()`
  makes each worker join the existing group and bind its `LOCAL_RANK` GPU instead of
  spawning. Added `DFINE(config=…)` ctor path + `sync_bn`/`find_unused_parameters` config
  fields (upstream defaults True/False). **Verified** the 2-process CPU/gloo spawn
  end-to-end (writes `last.pth`, parent reloads); a real bug — SyncBN needs GPU modules —
  was caught by running it and fixed (guard SyncBN to CUDA). Only multi-GPU launch is CI-
  gated (`DFINE_TEST_MULTIGPU=1`); the helper units run on CPU always.
