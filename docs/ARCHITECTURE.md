# Architecture

How D-FINE works, and how we reshape it from a YAML/registry research repo into a
single typed-Python class. Grounded in the upstream configs and the paper
(arxiv 2410.13842); verify exact shapes against upstream `src/` while porting.

## 1. Pipeline overview

```
image ─► HGNetV2 backbone ─► multi-scale feats {S8, S16, S32}
                               │
                               ▼
                    HybridEncoder (AIFI + CCFM/GELAN)
                               │  256-d (384-d for X) feature pyramid
                               ▼
                    DFINETransformer decoder  (N layers, 300 queries)
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                     ▼
   class logits (VFL)                  box = distribution over reg_max bins
                                        (Fine-grained Distribution Refinement)
                               │
                               ▼
                    DFINEPostProcessor ─► (labels, boxes_xyxy, scores)
```

Key idea (**FDR**): instead of regressing 4 box offsets directly, each box edge is a
**probability distribution over `reg_max` bins**, decoded through a non-uniform
**Weighting Function** (`reg_scale`, bounds `up`/`down`). The decoder refines these
distributions **residually** layer by layer. **GO-LSD** then distills the final
layer's sharpened distribution back into earlier layers via a decoupled distillation
(DDF) loss — improving localization at no inference cost.

## 2. Modules and where their parameters come from

| Module | Upstream class | Role | Key params (see CONFIG_REFERENCE) |
|---|---|---|---|
| Backbone | `HGNetv2` | ImageNet-pretrained conv backbone, B0–B5 | `backbone/name`, `return_idx`, `freeze_at`, `freeze_norm`, `freeze_stem_only`, `use_lab`, `pretrained` |
| Encoder | `HybridEncoder` | AIFI transformer on the top level + CCFM/GELAN cross-scale fusion | `hidden_dim`, `in_channels`, `feat_strides`, `use_encoder_idx`, `num_encoder_layers`, `nhead`, `dim_feedforward`, `dropout`, `enc_act`, `expansion`, `depth_mult`, `act` |
| Decoder | `DFINETransformer` | Deformable-attn DETR decoder w/ FDR + LQE | `num_queries`, `num_layers`, `eval_idx`, `num_levels`, `feat_channels`, `num_points`, `reg_max`, `reg_scale`, `decoder_offset_scale`, `decoder_method`, `lqe_hidden_dim`, `lqe_layers`, `layer_scale`, `up`/`down` |
| Denoising | (in decoder) | Contrastive denoising queries for faster convergence | `num_denoising`, `label_noise_ratio`, `box_noise_scale` |
| Matcher | `HungarianMatcher` | Bipartite matching pred↔gt | `cost_class`, `cost_bbox`, `cost_giou` |
| Loss | `DFINECriterion` | VFL + L1 + GIoU + FGL(DFL) + DDF(GO-LSD) | loss weights, focal `alpha`/`gamma`, DDF temperature |
| Postproc | `DFINEPostProcessor` | top-k decode to xyxy in original scale | `num_top_queries`, `remap_mscoco_category` |

### Native port status (Path A) — where each module lives now

We port these into `dfine/backends/native/`, one file per upstream module, with the
registry/YAML stripped and a `from_config(cfg)` constructor added. Status:

| Upstream class | Our file | Status |
|---|---|---|
| `HGNetv2` | `native/hgnetv2.py` | ✅ ported + shape tests |
| `HybridEncoder` | `native/hybrid_encoder.py` | ✅ ported + shape tests |
| `DFINETransformer` | `native/dfine_decoder.py` | ✅ ported + shape tests |
| `HungarianMatcher` | `native/matcher.py` | ⬜ next (training) |
| `DFINECriterion` | `native/criterion.py` | ⬜ next (training) |
| `DFINEPostProcessor` | `native/postprocessor.py` | ⬜ next (inference) |

Shared helpers: `native/common.py` (FrozenBatchNorm2d), `native/ops.py`
(activations, deformable-attn core, inverse_sigmoid…), `native/box_ops.py`,
`native/dfine_utils.py` (FDR), `native/denoising.py`.

## 3. Data flow contract (what the deploy model returns)

The deploy-mode forward (used for inference and export) is:

```
labels, boxes, scores = postprocessor(model(images), orig_target_sizes)
# images: (N,3,H,W) float in [0,1] after Resize((imgsz,imgsz)) + ToTensor
# orig_target_sizes: (N,2) = [[w,h], ...]
# boxes: xyxy in ORIGINAL image pixels (postproc rescales); scores: 0..1
```

This is already reproduced in the scaffold's `predictor.py`; the native backend must
preserve this exact contract so `Results` needs no rescaling.

## 4. From YAML+registry → typed Python

Upstream composition (to be removed from the user path):

- `configs/**.yml` include-trees define modules by string `type:` and kwargs.
- `src/core/yaml_config.py` (`YAMLConfig`) + `register`/`create` resolve those strings
  into instantiated `nn.Module`s (a hand-rolled dependency-injection container).

Our replacement:

- A single frozen `DFINEConfig` dataclass (all fields typed, defaults from presets).
- `dfine/backends/*` builds the modules **directly** from the dataclass — plain
  constructor calls, no string lookup. Example target:

```python
cfg = DFINEConfig.preset("l", num_classes=80, reg_max=32)
backbone = HGNetv2(name=cfg.backbone_name, return_idx=cfg.return_idx,
                   freeze_at=cfg.freeze_at, use_lab=cfg.use_lab,
                   pretrained=cfg.backbone_pretrained)
encoder  = HybridEncoder(in_channels=cfg.in_channels, hidden_dim=cfg.hidden_dim, ...)
decoder  = DFINETransformer(num_queries=cfg.num_queries, num_layers=cfg.decoder_layers,
                            reg_max=cfg.reg_max, reg_scale=cfg.reg_scale, ...)
```

## 5. Size variants (the presets)

From the paper's hyperparameter table + configs (details in CONFIG_REFERENCE):

| size | backbone | embed/hidden | ffn | decoder layers | notes |
|---|---|---|---|---|---|
| N | HGNetV2-B0 (light) | 256 | 1024 | 3 | `use_lab=True`, smallest GELAN |
| S | HGNetV2-B0 | 256 | 1024 | 3 | `use_lab=True`, `depth_mult=0.34`, `expansion=0.5` |
| M | HGNetV2-B2 | 256 | 1024 | 4 | |
| L | HGNetV2-B4 | 256 | 1024 | 6 | |
| X | HGNetV2-B5 | 384 | 2048 | 6 | `reg_scale=8`, `feat_channels=[384]*3` |

`in_channels` differ by backbone: `[256,512,1024]` for B0, `[512,1024,2048]` for B4/B5.

## 6. Training recipe (for `DFINE.train`)

- Optimizer AdamW with **param groups** (backbone vs norm/bias) — upstream expresses
  these as regex; we express them as structured Python rules in `train/trainer.py`.
- EMA of weights (decay ~0.9999, warmups), AMP, grad clip.
- Scheduler: flat-cosine / multistep with warmup.
- Two-phase augmentation: advanced augs (PhotometricDistort, ZoomOut, IoUCrop,
  MultiScaleInput) for most epochs, then a short **no-aug** tail (`no_aug_epoch`/
  `stop_epoch`) to stabilize. See `train/augment.py`.
- Data is COCO-format (images/ + annotations/*.json). Custom datasets set
  `remap_mscoco_category=False`.

## 7. Export

Deploy graph → ONNX (dynamic batch), then TensorRT (`trtexec --fp16`) or OpenVINO
downstream. Keep the two-input signature `(images, orig_target_sizes)` so exported
graphs match the torch path.
