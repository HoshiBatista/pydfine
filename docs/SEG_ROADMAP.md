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
- [x] **SS1 Port `SemSegDecoder`** *(2026-07-21)* → `dfine/backends/native/sem_seg_decoder.py`
      (reuses the native `MaskDecoder` fuser; `conv_gn_act` neck helper). Attr names verbatim
      (`mask_decoder`/`neck`/`dropout`/`classifier`/`aux_head`) so the trained fuser transfers
      from `dfine_seg_<size>_coco.pt` while neck/classifier/aux stay from-scratch. `forward`
      returns `{"sem_seg_logits": [B,C,H,W]}` (1/4 logits bilinearly ×4 to input res), adding
      `sem_seg_logits_aux` (deep supervision on the finest PAN feat) only in training. Added
      `from_config(cfg, *, mask_low_level_ch=None)` (num_classes/feat_channels/mask_dim from cfg;
      neck_dim/dropout/aux = upstream defaults). Exported from `native/__init__`. Tests
      (`test_sem_seg_decoder.py`, 5): eval/train forward shapes + aux, nano low-level plumbing,
      `from_config` dims, **+ the reused fuser strict-loads `dfine_seg_n_coco.pt`'s
      `decoder.mask_decoder.*` (0 missing/0 unexpected)**. Assembled model unchanged until SS2.
      Suite 241 passed / 16 skipped.
- [x] **SS2 Assemble sem_seg model** *(2026-07-21)*. `DFINE.from_config` swaps the whole
      decoder slot to `SemSegDecoder` when `task="sem_seg"` (detect/segment keep
      `DFINETransformer`). Added a `DFINEConfig.uses_mask_fuser` property (segment **or**
      sem_seg) and generalized `_seg_wiring` to it, so the nano low-level stride-8 plumbing
      now fires for sem_seg too (`return_idx` prepend + `mask_low_level_ch`); `forward` peels
      the extra feat unchanged. New `SemSegPostProcessor` (`native/sem_seg_postprocessor.py`,
      exported): argmax over classes → per-image NEAREST resize to original `(W,H)` →
      uint8 `[H,W]` label map (no letterbox, matching pydfine's resize path + upstream
      `process_sem_seg`). **Detect/segment paths unchanged.** Tests (`test_sem_seg_model.py`,
      7): wiring fires/no-ops per size, decoder swap + forward `[B,C,H,W]` at input res,
      detect/segment still `DFINETransformer`, postproc argmax+resize to `(H0,W0)` uint8 +
      identity-size argmax match, **+ (cache-gated) the assembled model's reused fuser
      strict-loads `dfine_seg_n_coco.pt`'s `decoder.mask_decoder.*`**. Base import torch-free.
      Suite 248 passed / 16 skipped.
- [x] **SS3 `Results` sem_seg surface** *(2026-07-21)*. New `SemSeg` container (`results.py`,
      lazily exported from `dfine`): `data` uint8 `[H,W]` at original scale, `.shape`, repr with
      class count (255 = void, excluded). `Results` gained `sem_seg` (+ repr note); `plot()`
      overlays each class with a palette color (alpha 0.5, void pixels untouched), before/besides
      the box+mask drawing. `DFINE.__init__` builds a `SemSegPostProcessor` when `task="sem_seg"`;
      `predict` branches to it → `_to_semseg_results` wraps each `[H0,W0]` label map in a boxless
      `Results` (no `Masks`). **Detect/segment predict unchanged** (`sem_seg=None`). Tests
      (`test_sem_seg_predict.py`, 5): `SemSeg` shape/repr, palette plot tints classes & skips void,
      boxless result, `predict(task="sem_seg")` → uint8 label map at original `(H,W)` with valid
      ids, detect has no `sem_seg`. Base import torch-free. Suite 253 passed / 16 skipped.
- [x] **SS4 sem_seg parity** *(2026-07-21)*. `scripts/gen_semseg_parity_fixture.py` (dev-only)
      runs genuine D-FINE-seg `SemSegDecoder` on small synthetic pyramids and stores — for a
      nano low-level-feat case and a plain stride-8 case — the decoder weights + inputs +
      output logits (`tests/data/semseg_parity.pt`, ~0.5 MB, self-contained). `test_semseg_parity.py`
      builds our port, strict-loads the same weights, feeds the same inputs, and asserts
      **bit-exact** logits — max-abs diff `0.0` for both cases — plus a postproc argmax match.
      The sem_seg forward is dimension-independent, so tiny channels exercise every path; the
      fixture needs no checkpoint and D-FINE-seg is not imported. **Note:** there are no released
      *trained* sem_seg weights (HF `dfine_seg_*_coco.pt` is instance-seg), so parity is
      "same weights → same output"; the trained-fuser transfer is pinned by the SS1/SS2
      strict-load tests. **Phase SS (semantic segmentation, inference) complete.** Suite 256
      passed / 16 skipped.

### Phase TS — Segmentation training (largest, optional / later)
- [x] **TS1 Mask losses** *(2026-07-23)*. Ported D-FINE-seg's instance-mask losses into
      `DFINECriterion`: `loss_masks` (YOLO-style **box-cropped BCE + soft Dice** on the matched
      queries' 1/4-res mask logits — GT masks bilinear-resized to mask space, loss computed only
      inside each GT box and normalized by box area) with `_prepare_target_masks` /
      `_prepare_target_boxes_for_masks` / `_cropped_bce_loss` / `_cropped_dice_loss` helpers.
      Registered `"masks"` in `loss_map`; the final/aux/DN layers already carry `pred_masks`, and
      a dedicated `dn_pred_masks` block adds the `_dn_final` DN mask-supervision term.
      `from_config` appends `"masks"` + `loss_mask_bce`/`loss_mask_dice` weights (both `1.0`, new
      config fields) **only when `cfg.enable_mask_head`** — detection is byte-identical (no mask
      loss, no mask weights). Tests: segment criterion emits finite, differentiable mask losses
      (incl. `_aux_`/`_dn_final`) with gradient reaching the mask branch; detect stays mask-free.
      Attribution added to the criterion module docstring (© ArgoHA, Apache-2.0). base import
      still torch-free; suite 267 passed / 16 skipped. Matcher mask costs are **TS2**.
- [x] **TS2 Matcher mask costs** *(2026-07-23)*. Ported D-FINE-seg's instance-mask matching
      costs into `HungarianMatcher`: module-level `dice_cost` (pairwise `1 - Dice` from mask
      probs) and `sigmoid_focal_cost` (pairwise pixel-wise focal from logits, pixel-normalized),
      added into the assignment cost via a new `_add_mask_cost` in-place helper. It resizes GT
      masks (bilinear) to the prediction's `(Hm, Wm)`, drops leading denoising queries if present,
      and accumulates `cost_mask_dice·dice + cost_mask·focal` per batch element. `from_config`
      sets `cost_mask`/`cost_mask_dice` (both `1.0`, new config fields) **only when
      `cfg.enable_mask_head`**; the helper is also a no-op unless the outputs carry `pred_masks`
      and some target has masks — so detection matching is byte-identical (cost weights `0`).
      Tests: `dice_cost`/`sigmoid_focal_cost` value checks, segment `from_config` gating, and a
      controlled class/box tie broken by the mask cost (seg matcher picks the matching-mask query;
      detect matcher ignores masks). Attribution added to the matcher module docstring (© ArgoHA,
      Apache-2.0). base import torch-free; suite 270 passed / 16 skipped.
- [ ] **TS3 `SemSegCriterion`** (CE + soft Dice + `ignore_index`).
- [ ] **TS4 Seg datasets**: YOLO polygon labels → instance masks; PNG masks → sem_seg targets;
      wire into `build_coco_dataloaders`/a new loader. (Muon/Adan/mosaic are **out of scope** —
      reuse pydfine's AdamW trainer; note the deviation, like the flatcosine one.)
- [ ] **TS5 `DFINE.train(task=...)`** end-to-end (overfit-one-batch mask-loss-drops test).

### Phase XS — Export & polish (later)
- [x] **XS1 ONNX export** *(2026-07-23)*. `export_onnx`/`tensorrt_command` gained a `task`
      arg selecting the output contract: `segment` → `(labels, boxes, scores, masks)` where
      `masks` are the top-k queries' sigmoid maps at 1/4 res (postprocessor deploy path
      gathers them; threshold/resize/box-clip on host); `sem_seg` → single `images` input →
      `sem_seg` `[N, H, W]` uint8 label map (argmax **fused into the graph** at network res,
      host resizes nearest). Added `SegInstanceDeployModel`/`SemSegDeployModel` +
      `_build_deploy` picking the deploy module and I/O names by task; `DFINE.export` passes
      `task=self.config.task`. Detect export is byte-identical (task defaults `detect`).
      onnxruntime-verified (sem_seg >99.9% pixel agreement; segment structural + sorted-scores
      parity — numeric mask parity is covered bit-exactly by `test_seg_parity.py`).
- [x] **XS2 Docs** *(2026-07-22)*. Added a **Segmentation** quickstart to `README.md`
      (instance via `from_pretrained("dfine-seg-l")` → `r.masks`/`r.boxes`/`r.plot()`;
      sem_seg via `DFINE(task="sem_seg")` → `r.sem_seg`; notes on the `[hf]` extra,
      original-scale outputs, and parity). Documented the `Masks` and `SemSeg` containers
      in `docs/api/results.md` (mkdocstrings `:::` blocks + a prose intro mapping
      task→attribute). Polished `CONFIG_REFERENCE` §1: `mask_dim` now noted as used by
      **both** segment and sem_seg, plus a note that `enable_mask_head`/`uses_mask_fuser`/
      `mask_low_level_ch` are derived (not user-set) and detect stays byte-identical.
      Docs build clean under `mkdocs --strict`. Docs-only.
- [x] **XS3 Attribution** *(2026-07-22)*. `NOTICE` gained a dedicated D-FINE-seg section
      (© ArgoHA, Apache-2.0) explaining it is an independent from-scratch framework and
      listing exactly which modules pydfine ports (`MaskDecoder`, the decoder mask branch,
      `SemSegDecoder`, the sem_seg postprocessor) + the HF weights (`ArgoSA/D-FINE-seg`).
      `README.md` "License & attribution" now credits D-FINE-seg alongside D-FINE. Verified
      every ported seg module already carries a per-file provenance header naming its source
      file, license, © ArgoHA, and changes (`mask_decoder.py`, `sem_seg_decoder.py`,
      `sem_seg_postprocessor.py`, the mask-branch note in `dfine_decoder.py`, the HF-weights
      note in `registry.py`); added a matching note to `dfine.py`'s `_seg_wiring` assembly.
      Docs-only — base import torch-free, ruff clean, suite unchanged.

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
