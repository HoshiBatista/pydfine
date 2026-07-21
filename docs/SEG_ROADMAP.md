# Segmentation integration — analysis & roadmap

Goal: bring **instance segmentation** (`task="segment"`) and **semantic segmentation**
(`task="sem_seg"`) into `pydfine`, in the same config-first, parity-driven style as the
detection path. Source of the work: the [`ArgoHA/D-FINE-seg`](https://github.com/ArgoHA/D-FINE-seg)
framework (cloned at `./D-FINE-seg/`).

---

## 1. What D-FINE-seg is (and is not)

- A **separate, from-scratch framework** by ArgoHA — *not* a fork of Peterande/D-FINE.
  Its README states: "the detection core follows the D-FINE paper; segmentation heads,
  training, export and inference are implemented from scratch."
- Hydra/YAML-driven (one big `config.yaml`), `uv`-managed, task flag `task: detect |
  segment | sem_seg`, five sizes `n/s/m/l/x`.
- Ships **its own retrained weights** on Hugging Face (`ArgoSA/D-FINE-seg`):
  - detection: `dfine_<size>_{coco,obj2coco}.pt`
  - instance seg: `dfine_seg_<size>_coco.pt` (includes a trained `MaskDecoder`)
- Extra machinery beyond the model: Muon+Adan optimizers, mosaic/affine augs, ByteTrack,
  SAM3 auto-labeling, multi-backend export (ONNX/TRT/OpenVINO/CoreML/LiteRT), INT8.

### Parity implications (read this first)

`pydfine`'s discipline is **bit-exact parity vs Peterande's released `.pth`**. D-FINE-seg
uses **different weights** (ArgoSA HF) and a **different config surface**. Therefore:

- The seg **parity anchor is D-FINE-seg's own torch output**, not Peterande's. We generate
  fixtures by running `./D-FINE-seg` and match our port against them — same method as
  `tests/test_parity.py`, new anchor.
- We **map** D-FINE-seg's Hydra config onto typed `DFINEConfig` params. We do **not** adopt
  Hydra, Muon, or its training loop wholesale — only what's needed for parity + a clean API.
- **Open question to verify early:** does D-FINE-seg's *detection-core* param naming match
  `pydfine`'s native modules (so `dfine_seg_<size>_coco.pt`'s backbone/encoder/decoder keys
  strict-load into our modules)? The seg checkpoints are the only ones that carry mask
  weights, so this gates everything. Task S1 below settles it.

---

## 2. Architecture delta vs current `pydfine` (what to port)

All new arch lives in `D-FINE-seg/src/d_fine/arch/dfine_decoder.py`. Three additions:

1. **`MaskDecoder`** (`in_chs`, `out_ch=mask_dim`): fuses HybridEncoder PAN outputs
   (stride 8/16/32) to a single **1/4-resolution** mask-feature map. Lateral 1×1 + GroupNorm
   per level, bilinear-sum fusion, 3×3 smooth conv, bilinear ×2 upsample. Pure conv — no
   registry.
2. **`DFINETransformer` mask branch** (guarded by `enable_mask_head`):
   - `self.mask_decoder = MaskDecoder(...)`
   - `self.mask_head = MLP(hidden_dim, hidden_dim, mask_dim, num_layers=3)` — per-query mask
     embedding.
   - inference mask logits = `einsum("bqc,bchw->bqhw", mask_embed * C^-0.5, mask_feat)`.
   - decoder called with `return_queries=True` to expose `hs` (per-query features).
3. **`SemSegDecoder`** (replaces `DFINETransformer` entirely when `task="sem_seg"`): reuses
   `MaskDecoder` (attr name kept so pretrained fuser transfers) → 2× conv-GN-ReLU neck →
   Dropout2d → 1×1 classifier → bilinear ×4 to input res. No queries, no matcher, no NMS.

**Low-level-feature plumbing** (nano-specific, but general): when a mask head is on and the
encoder has no stride-8 level (nano is 2-level, strides 16/32), the backbone returns an extra
leading **stride-8** feature (`HGNetv2.return_idx` gets `1` prepended) and the assembled model
peels it off as `low_level_feat` and threads it into the mask decoder. `mask_low_level_ch` =
backbone stage2 out channels.

**Per-size mask dims** (from `configs.py`): `mask_dim = 128` for **n**, `256` for s/m/l/x.
Backbones/encoder/decoder dims otherwise identical to `pydfine`'s existing presets (B0/B2/B4/B5,
same hidden_dim/feat_channels/return_idx) — good, the detection core is the one we already ship.

**Training-only bits** (needed only for Phase T): mask losses in `DFINECriterion`
(box-cropped BCE + Dice, `loss_masks`), mask supervision inside contrastive denoising, Hungarian
matcher mask costs (`dice_cost` + `sigmoid_focal_cost`), and `SemSegCriterion` (CE + multi-class
soft Dice with `ignore_index`).

---

## 3. Public API shape (config-first, backend-agnostic)

Keep the headline surface typed and small:

```python
model = DFINE(size="s", task="segment")          # or task="sem_seg"
model = DFINE.from_pretrained("dfine_seg_s_coco") # ArgoHA HF weights
results = model.predict("img.jpg")               # list[Results]
r = results[0]
r.masks            # Masks object: .data [N,H,W] bool/float, original scale
r.boxes            # unchanged (segment still yields boxes)
r.plot()           # boxes + mask overlays
```

New `DFINEConfig` fields (typed, defaulted so detect is unchanged):
`task: Literal["detect","segment","sem_seg"] = "detect"`, `mask_dim: int` (preset per size),
`enable_mask_head: bool` (derived from task), plus sem_seg knobs (`ignore_index=255`,
`sem_seg_class_weights`). `mask_low_level_ch` is derived, not user-set.

New checkpoint catalogue entries in `registry.py` for `dfine_seg_<size>_coco` + a HF download
source (`huggingface_hub`, new optional dep) alongside the existing GitHub-release source.

---

## 4. Roadmap (phased, parity-gated)

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Work the lowest unchecked task.

### Phase S — Instance-seg inference (highest value, most parity-tractable)
- [x] **S1 Weight-compat probe** *(2026-07-21)*. Downloaded `dfine_seg_n_coco.pt` from
      `ArgoSA/D-FINE-seg` and diffed against a native nano model: **all 674 native keys present,
      0 shape mismatches, 0 renames**. The checkpoint adds exactly **21 keys**, all under
      `decoder.mask_decoder.*` (MaskDecoder) + `decoder.mask_head.layers.*` (3-layer MLP).
      Conclusion: the seg det-core is byte-compatible with our native modules — porting the two
      mask modules + a `strict=False`-then-verify load is all that's needed. Plan validated.
- [x] **S2 Port `MaskDecoder`** *(2026-07-21)* → `dfine/backends/native/mask_decoder.py`
      (GroupNorm fuser; `lateral`/`bn`/`fusion_conv`/`fusion_norm`/`up_conv`/`bn1` names verbatim).
      Exported from `native/__init__`. Tests (`test_mask_decoder.py`): per-size forward shape
      (output = 2× finest input res, `out_ch` channels) **+ a strict-load parity test** — the
      module `load_state_dict(strict=True)`s the real `decoder.mask_decoder.*` sub-dict from
      `dfine_seg_n_coco.pt` with 0 missing/0 unexpected. Full suite 210 passed / 12 skipped.
- [x] **S3 Mask branch in native `DFINETransformer`** *(2026-07-21)*. Added `enable_mask_head`/
      `mask_dim`/`mask_low_level_ch` ctor args (+ keyword-only `from_config` kwargs, default
      off); `mask_decoder` (`MaskDecoder`) + `mask_head` (3-layer MLP) modules;
      `_should_do_masks` + `_mask_logits_from_h` (`einsum` per-query masks, `mask_dim^-0.5`
      scale); `TransformerDecoder.forward` gained `return_queries` → returns per-layer `hs`
      (now a 7-tuple); `forward(low_level_feat=...)` computes `pred_masks` (sigmoid at eval,
      logits at train) and threads aux/dn masks via an extended `_set_aux_loss2`. **Detection
      path byte-identical when off** (guarded everywhere). Tests (`test_seg_decoder.py`):
      det-keys-unchanged-when-off, eval mask forward shape `[B,Q,H/4,W/4]` in [0,1], **+
      whole-decoder strict-load of `dfine_seg_n_coco.pt`'s `decoder.*` (0 missing/0
      unexpected)**. Assembled model still detection-only until S4. Suite 213 passed / 12 skipped.
- [x] **S4 Assembled seg model + config fields** *(2026-07-21)*. Added `DFINEConfig.task`
      (`detect`/`segment`/`sem_seg`, validated) + `mask_dim` (128 for N, else 256) + derived
      `enable_mask_head` property (**S5's config half, pulled forward**). `DFINE._seg_wiring(cfg)`
      centralizes the segmentation wiring: when the mask head is on and the encoder has no
      stride-8 level (nano), the backbone emits an extra stride-8 feature and `mask_low_level_ch`
      = B0 stage2 channels; `from_config` passes them to backbone/decoder. `HGNetv2.from_config`
      gained a `return_idx` override (no frozen-config mutation). `DFINE.forward` peels the extra
      leading backbone feature into `low_level_feat`. **Detection path byte-identical** (wiring
      no-ops for `task="detect"`). `docs/CONFIG_REFERENCE.md` updated. Verified: a `task="segment"`
      nano model strict-loads the full `dfine_seg_n_coco.pt` (0 missing/0 unexpected) and its
      forward yields `pred_masks [1,300,H/4,W/4]` in [0,1]. Tests `test_seg_model.py` (5).
      Suite 218 passed / 12 skipped.
- [x] **S5 (remaining) Registry + HF weights** *(2026-07-21)*. `CheckpointSpec` gained
      `task`/`source`/`repo_id`; added `dfine-seg-{n,s,m,l,x}` entries (`source="hf"`,
      `repo_id="ArgoSA/D-FINE-seg"`, `task="segment"`, 80-class) — all 5 confirmed present on HF.
      `config_for` now propagates `spec.task` (wires the mask head). `downloads.download_weights`
      branches on `source`: HF specs fetch via `huggingface_hub.hf_hub_download` (lazy import,
      new `[hf]` extra; also in `[dev]`), GitHub specs unchanged. `native/loader` already handles
      ArgoHA's plain state_dict (bare-tensor-dict branch). **`DFINE.from_pretrained("dfine-seg-n")`
      works end-to-end** (download → build w/ mask head → strict-load → mask forward) — verified by
      a (cache-gated) test. `docs/CONFIG_REFERENCE.md` done in S4. Registry stays torch-free (base
      import check green). Tests: seg specs present/wired per size, source-aware URL check,
      from_pretrained e2e. Suite 224 passed / 12 skipped. **Phase S inference chain wired up to
      here; S6 (Results.masks) + S7 (parity) close it out.**
- [x] **S6 Mask postprocessing + `Results.masks`** *(2026-07-21)*. The postprocessor now
      surfaces each detection's `query_index` (from its top-k `index // num_classes`), so
      `predict` gathers the surviving queries' masks from `outputs["pred_masks"]`, resizes them
      bilinearly to the original image size, thresholds (`mask_thresh=0.5`), and clips each to its
      box (`_cleanup_masks`, matching D-FINE-seg's inference). New `Masks` container (`data`
      `[N,H,W]` bool, original scale; aligned 1:1 with `Boxes`); `Results.masks`; `plot()` alpha-
      overlays masks in each detection's color under the boxes; `to_supervision()` attaches a bool
      `mask` array. `Masks` exported lazily from `dfine`. **Detection path unchanged** (no
      `pred_masks` → `masks=None`). Tests: `Masks` container + masked `plot`/`to_supervision`
      (`test_results.py`), and `predict(task="segment")` → N masks aligned with N boxes at
      original scale + detection-has-no-masks (`test_seg_model.py`). Base import torch-free.
      Suite 230 passed / 12 skipped. *(COCO-RLE `to_coco` deferred — needs an RLE encoder;
      not on the critical path.)*
- [x] **S7 Instance-seg parity test** *(2026-07-21)*. `scripts/gen_seg_parity_fixture.py`
      (dev-only) runs genuine **D-FINE-seg** (`build_model`, strict-load) on a seeded input and
      saves raw `pred_logits`/`pred_boxes` (all queries) + an 8-query slice of instance-mask maps
      (fp16, ~0.5 MB) to `tests/data/seg_parity_<size>.pt`. `tests/test_seg_parity.py` builds our
      port from the *same* seg checkpoint, feeds the *same* input, and asserts it reproduces those
      numbers (gated on the seg `.pt` being in the HF cache; D-FINE-seg not imported). **Result on
      nano: bit-exact** — `pred_logits`/`pred_boxes` max-abs diff `0.0`; masks max-abs `2.4e-4`,
      i.e. only the fixture's fp16 storage rounding (< fp16 eps `9.8e-4`), so the port is byte-for-
      byte faithful up to storage. Fixtures for `s/m/l/x` regenerate the same way (test skips until
      present). **Phase S (instance segmentation, inference) complete.** Suite 231 passed / 16
      skipped.

### Phase SS — Semantic-seg inference
- [ ] **SS1 Port `SemSegDecoder`** (`native/sem_seg_decoder.py`, reuses `MaskDecoder`).
- [ ] **SS2 Assemble sem_seg model** (decoder-slot swap when `task="sem_seg"`; argmax
      postprocess → `[H,W]` uint8 label map at original res).
- [ ] **SS3 `Results` sem_seg surface** (`r.sem_seg` label map, palette overlay plot).
- [ ] **SS4 sem_seg weights + parity** (HF `dfine_seg_*`; fixture vs D-FINE-seg).

### Phase TS — Segmentation training (largest, optional / later)
- [ ] **TS1 Mask losses** in native criterion (box-cropped BCE + Dice; DN mask supervision).
- [ ] **TS2 Matcher mask costs** (`dice_cost` + `sigmoid_focal_cost`).
- [ ] **TS3 `SemSegCriterion`** (CE + soft Dice + `ignore_index`).
- [ ] **TS4 Seg datasets**: YOLO polygon labels → instance masks; PNG masks → sem_seg targets;
      wire into `build_coco_dataloaders`/a new loader. (Muon/Adan/mosaic are **out of scope** —
      reuse pydfine's AdamW trainer; note the deviation, like the flatcosine one.)
- [ ] **TS5 `DFINE.train(task=...)`** end-to-end (overfit-one-batch mask-loss-drops test).

### Phase XS — Export & polish (later)
- [ ] **XS1 ONNX export** with fused mask/argmax graph (segment + sem_seg output contracts).
- [ ] **XS2 Docs**: API pages, README seg quickstart, `CONFIG_REFERENCE` seg fields.
- [ ] **XS3 Attribution**: NOTICE/README credit to `ArgoHA/D-FINE-seg` (Apache-2.0) +
      per-file provenance headers on ported modules.

---

## 5. Scope decisions / non-goals (proposed)

- **Do not** adopt Hydra, Muon, Adan, mosaic, SAM3, multi-backend export beyond ONNX, or the
  `.npy` multi-channel path in v1. They're framework features orthogonal to "config-first
  D-FINE in Python." Revisit per-item if asked.
- **Inference-first.** Land Phase S (instance seg) and SS (sem seg) — real user value,
  bounded, parity-checkable — before the much larger training phase.
- **Attribution is mandatory** (Apache-2.0): credit ArgoHA/D-FINE-seg for all ported seg code.

## 6. Locked decisions (owner, 2026-07-21)
1. **Instance seg first** — Phase S is the active phase; sem_seg (SS) follows.
2. **Inference only for now** — load ArgoHA's pretrained seg weights + predict masks. Phase TS
   (training) is deferred until explicitly requested.
3. **Add `huggingface_hub`** (new optional dep, e.g. `[hf]` extra) to auto-download
   `dfine_seg_<size>_coco.pt` from `ArgoSA/D-FINE-seg` on first use.

Active task: **S1 — weight-compat probe**.
