# Config reference — `DFINEConfig` / `DFINE(...)` parameters

This is the **source of truth** for public parameter names, groups, and defaults.
Every param here becomes a field on `DFINEConfig` and a `DFINE(...)` kwarg.

Defaults below are taken from upstream `D-FINE/configs/*.yml` + `src/` + the paper.
Values marked `⚠verify` must be confirmed against upstream before shipping. The
**left column is our public name**; where it differs from the upstream constructor
arg, the description notes the upstream name.

Presets (`size=`) set the size-dependent fields (§9). Anything set explicitly in
`DFINE(...)` overrides the preset.

---

## 1. Top-level / task

| Param | Type | Default | Description |
|---|---|---|---|
| `size` | str\|None | None | `"n"\|"s"\|"m"\|"l"\|"x"` preset selector. None = fully manual. |
| `task` | str | `"detect"` | `"detect"\|"segment"` (instance masks)\|`"sem_seg"` (per-pixel). |
| `num_classes` | int | 80 | Number of object classes. |
| `class_names` | list[str]\|None | None | Optional names; defaults to COCO-80 when `num_classes==80`. |
| `imgsz` | int | 640 | Square inference/training resolution (`eval_spatial_size`). |
| `device` | str | "cpu" | `"cpu"\|"cuda"\|"cuda:0"\|"mps"`. |
| `remap_mscoco_category` | bool | False | COCO-id remap; keep False for custom datasets. |
| `mask_dim` | int | 256 | Mask-feature dim of the shared fuser (**128 for N**); used by `task="segment"` **and** `task="sem_seg"`. |

The mask head is enabled automatically for `task="segment"`/`"sem_seg"` (the derived
`enable_mask_head`/`uses_mask_fuser` properties) — you don't set it. `mask_low_level_ch`
(the backbone stride-8 channel count threaded into the fuser on models with no native 1/8
level, e.g. nano) is derived too. Detection is byte-identical when `task="detect"`.

## 2. Backbone (HGNetV2)

| Param | Type | Default | Description |
|---|---|---|---|
| `backbone` | str | `"hgnetv2_b4"` | Variant `b0..b5` (maps to upstream `HGNetv2.name` = `B0..B5`). |
| `backbone_pretrained` | bool | True | Load ImageNet-pretrained backbone weights. |
| `return_idx` | list[int] | `[1,2,3]` | Backbone stages fed to the encoder. |
| `freeze_at` | int | -1 | Freeze stages up to this index (-1 = none). |
| `freeze_stem_only` | bool | False | Freeze only the stem (used by X/obj365). |
| `freeze_norm` | bool | False | Freeze BatchNorm in the backbone. |
| `use_lab` | bool | False | Learnable-affine block; **True for N/S**. |
| `backbone_local_dir` | str\|None | None | Local dir for cached backbone weights. |

`in_channels` (encoder input) is backbone-dependent and set by the preset:
B0 `[256,512,1024]` (S) or `[512,1024]` (N, 2-level), B2 `[384,768,1536]` (M),
B4/B5 `[512,1024,2048]` (L/X). See the §11 per-size table for the full mapping.

## 3. Encoder (HybridEncoder)

| Param | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | int | 256 | Encoder embedding dim (384 for X). |
| `in_channels` | list[int] | `[512,1024,2048]` | From backbone; preset-set. |
| `feat_strides` | list[int] | `[8,16,32]` | Strides of the pyramid levels. |
| `use_encoder_idx` | list[int] | `[2]` | Which level runs AIFI self-attn. |
| `encoder_layers` | int | 1 | AIFI transformer layers (`num_encoder_layers`). |
| `nhead` | int | 8 | Encoder attention heads. |
| `encoder_dim_feedforward` | int | 1024 | AIFI FFN dim (`dim_feedforward`; 2048 for X). |
| `encoder_dropout` | float | 0.0 | Encoder dropout. |
| `enc_act` | str | "gelu" | AIFI activation. |
| `encoder_expansion` | float | 1.0 | CCFM/GELAN channel expansion (0.5 for S). |
| `depth_mult` | float | 1.0 | GELAN depth multiplier (0.34 for S). |
| `encoder_act` | str | "silu" | Fusion/GELAN activation (`act`). |

## 4. Decoder (DFINETransformer)

| Param | Type | Default | Description |
|---|---|---|---|
| `decoder_hidden_dim` | int \| None | None | Decoder embedding dim; `None` resolves to `hidden_dim`. **256 for X**, where the encoder runs at 384 but the decoder stays 256. |
| `num_queries` | int | 300 | Object queries. |
| `decoder_dim_feedforward` | int | 1024 | Decoder FFN dim (512 for N; separate from `encoder_dim_feedforward`). |
| `decoder_layers` | int | 6 | Decoder layers (`num_layers`; 4 for M, 3 for S/N). |
| `eval_idx` | int | -1 | Layer used at eval; negative = from end (aux-layer scaling). |
| `num_levels` | int | 3 | Multi-scale feature levels. |
| `feat_channels` | list[int] | `[256,256,256]` | Per-level channels (`[384]*3` for X). |
| `num_points` | list[int] | `[3,6,3]` | Deformable sampling points per level (`[6,6]` for N). |
| `decoder_nhead` | int | 8 | Decoder attention heads (upstream `nhead`). |
| `decoder_offset_scale` | float | 0.5 | Deformable-attn offset scale. Upstream hard-codes 0.5; not currently wired to the module. |
| `decoder_method` | str | "default" | `"default"\|"discrete"` deformable sampling (upstream `cross_attn_method`). |
| `layer_scale` | float | 1.0 | Hidden-dim scale for the wide (aux) decoder layers. |

### 4a. Fine-grained Distribution Refinement (FDR)

| Param | Type | Default | Description |
|---|---|---|---|
| `reg_max` | int | 32 | Bins per box edge in the regression distribution. |
| `reg_scale` | float | 4.0 | Weighting-function scale (8 for X and obj365 variants). |

> `up` (upper bound of W(n)) is **not** a config field: upstream fixes it as a frozen
> `nn.Parameter(0.5)` inside `DFINETransformer`, so we do the same. There is no `down`
> parameter — the lower bound is derived symmetrically inside `weighting_function`.

### 4b. Location Quality Estimator (LQE)

| Param | Type | Default | Description |
|---|---|---|---|
| `lqe_hidden_dim` | int | 64 | LQE MLP hidden dim (upstream hard-codes 64). |
| `lqe_layers` | int | 2 | LQE MLP layers (upstream hard-codes 2). |

## 5. Denoising (contrastive DN queries)

| Param | Type | Default | Description |
|---|---|---|---|
| `num_denoising` | int | 100 | Denoising queries. |
| `label_noise_ratio` | float | 0.5 | Label-flip noise. |
| `box_noise_scale` | float | 1.0 | Box perturbation scale. |

## 6. Matcher (Hungarian) — training only

| Param | Type | Default | Description |
|---|---|---|---|
| `cost_class` | float | 2.0 | Classification match cost. |
| `cost_bbox` | float | 5.0 | L1 box match cost. |
| `cost_giou` | float | 2.0 | GIoU match cost. |
| `matcher_alpha` | float | 0.25 | Matcher focal alpha (distinct from criterion `alpha`). |
| `matcher_gamma` | float | 2.0 | Matcher focal gamma. |

## 7. Losses (DFINECriterion) — training only

| Param | Type | Default | Description |
|---|---|---|---|
| `loss_vfl_weight` | float | 1.0 | Varifocal classification loss. |
| `loss_bbox_weight` | float | 5.0 | L1 box loss. |
| `loss_giou_weight` | float | 2.0 | GIoU loss. |
| `loss_fgl_weight` | float | 0.15 | Fine-grained localization (DFL) loss. |
| `loss_ddf_weight` | float | 1.5 | GO-LSD decoupled distillation loss. |
| `focal_alpha` | float | 0.75 | Focal/VFL alpha (criterion `alpha`). |
| `focal_gamma` | float | 2.0 | Focal/VFL gamma (⚠verify). |
| `ddf_temperature` | float | 0.05 | GO-LSD temperature (`T_init≈5e-2`). |
| `aux_loss` | bool | True | Supervise auxiliary decoder layers. |

## 8. Postprocessor

| Param | Type | Default | Description |
|---|---|---|---|
| `num_top_queries` | int | 300 | Top-k detections kept. |
| `conf` | float | 0.4 | Default score threshold at predict time. |

## 9. Training / optimization (`DFINE.train`)

| Param | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | 72 | Total epochs (L/X: 72+2; M/S: 120+4). |
| `batch` | int | 32 | Total batch size. |
| `lr` | float | 2.5e-4 | Base LR (L/X 2.5e-4; M/S 2e-4). |
| `lr_backbone` | float | 1.25e-4 | Backbone LR (smaller models use higher). |
| `weight_decay` | float | 1.25e-4 | AdamW weight decay (M/S 1e-4). |
| `betas` | tuple | (0.9, 0.999) | AdamW betas. |
| `clip_max_norm` | float | 0.1 | Grad clip (⚠verify). |
| `warmup_iters` | int | 500 | LR warmup iterations. |
| `scheduler` | str | "flatcosine" | `"flatcosine"\|"multistep"`. |
| `ema_decay` | float | 0.9999 | Weight-EMA decay. |
| `ema_warmups` | int | 2000 | EMA warmup steps (⚠verify per size). |
| `use_amp` | bool | True | Mixed precision. |
| `no_aug_epoch` | int | 2 | Trailing epochs with advanced augs off (M/S: 4). |
| `seed` | int | 0 | RNG seed. |
| `workers` | int | 4 | Dataloader workers. |
| `checkpoint_freq` | int | 1 | Save every N epochs. |
| `sync_bn` | bool | True | Convert BN→SyncBN under multi-GPU DDP (GPU-only; no-op single-process/CPU). |
| `find_unused_parameters` | bool | False | DDP: allow params that get no gradient (slower). |

## 10. Augmentation (`train/augment.py`)

| Param | Type | Default | Description |
|---|---|---|---|
| `aug_photometric` | bool | True | RandomPhotometricDistort. |
| `aug_zoom_out` | bool | True | RandomZoomOut. |
| `aug_iou_crop` | bool | True | RandomIoUCrop. |
| `aug_hflip` | bool | True | RandomHorizontalFlip. |
| `multiscale` | bool | True | RandomMultiScaleInput. |
| `base_size` | int | 640 | Multi-scale base size. |
| `base_size_repeat` | int\|None | 3 | Multi-scale repeat factor. |

---

## 11. Per-size presets (`SIZE_PRESETS`)

**Verified** against `D-FINE/configs/dfine/dfine_hgnetv2_{n,s,m,l,x}_coco.yml` +
`include/dfine_hgnetv2.yml` + `include/optimizer.yml` (2026-07-11). This table now
matches `dfine/config.py::SIZE_PRESETS` exactly. **Note N is structurally different:
it is a 2-level pyramid at `hidden_dim=128`, not 3-level/256.**

| field | n | s | m | l | x |
|---|---|---|---|---|---|
| `backbone` | hgnetv2_b0 | hgnetv2_b0 | hgnetv2_b2 | hgnetv2_b4 | hgnetv2_b5 |
| `use_lab` | True | True | True | False | False |
| `num_levels` | **2** | 3 | 3 | 3 | 3 |
| `hidden_dim` | **128** | 256 | 256 | 256 | 384 |
| `decoder_hidden_dim` | 128 | 256 | 256 | 256 | **256** |
| `return_idx` | [2,3] | [1,2,3] | [1,2,3] | [1,2,3] | [1,2,3] |
| `in_channels` | [512,1024] | [256,512,1024] | [384,768,1536] | [512,1024,2048] | [512,1024,2048] |
| `feat_strides` | [16,32] | [8,16,32] | [8,16,32] | [8,16,32] | [8,16,32] |
| `feat_channels` | [128,128] | [256]*3 | [256]*3 | [256]*3 | [384]*3 |
| `num_points` | [6,6] | [3,6,3] | [3,6,3] | [3,6,3] | [3,6,3] |
| `use_encoder_idx` | [1] | [2] | [2] | [2] | [2] |
| `encoder_dim_feedforward` | 512 | 1024 | 1024 | 1024 | 2048 |
| `decoder_layers` | 3 | 3 | 4 | 6 | 6 |
| `depth_mult` | 0.5 | 0.34 | 0.67 | 1.0 | 1.0 |
| `encoder_expansion` | 0.34 | 0.5 | 1.0 | 1.0 | 1.0 |
| `reg_scale` | 4 | 4 | 4 | 4 | 8 |
| `freeze_at` / `freeze_norm` | -1 / F | -1 / F | -1 / F | 0 / T | 0 / T |
| `epochs` (incl. no-aug tail) | 160 | 132 | 132 | 80 | 80 |
| `lr` | 8e-4 | 2e-4 | 2e-4 | 2.5e-4 | 2.5e-4 |
| `lr_backbone` | 4e-4 | 1e-4 | 2e-5 | 1.25e-5 | 2.5e-6 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1.25e-4 | 1.25e-4 |
| `base_size_repeat` | None | 20 | 6 | 4 | 3 |

## 12. Checkpoint presets (weights)

The registry maps preset+dataset names to upstream release URLs (see
`dfine/registry.py`), e.g. `dfine-l` (COCO), `dfine-l-obj2coco`, `dfine-l-obj365`.
Building `DFINE(size="l")` + `.load("dfine-l")` must reproduce upstream COCO AP —
this is the parity test in `tests/`.
