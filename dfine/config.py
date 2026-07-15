"""Typed, config-first surface for D-FINE.

``DFINEConfig`` is a frozen dataclass holding **every** model + training option as a
plain Python field. ``DFINE(...)`` kwargs map 1:1 onto these fields, so there is no
YAML and no registry on the user path (see ``AGENTS.md`` §2).

Preset values are verified against the upstream configs in
``D-FINE/configs/dfine/*.yml`` (not the paper tables) so a preset reproduces the
matching released checkpoint's architecture. ``docs/CONFIG_REFERENCE.md`` is the
human-readable description of each field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

# Preset selector values accepted by ``size=`` / ``DFINEConfig.preset``.
SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class DFINEConfig:
    """All D-FINE model + training options as one typed, frozen dataclass.

    Build with a preset via :meth:`preset` (fills size-dependent fields) or construct
    directly for a fully custom architecture. Every field is overridable inline.
    """

    # --- 1. Top-level / task -------------------------------------------------
    size: str | None = None  # "n"|"s"|"m"|"l"|"x" preset selector; None = manual
    num_classes: int = 80  # number of object classes
    class_names: list[str] | None = None  # optional label names (len == num_classes)
    imgsz: int = 640  # square inference/training resolution (eval_spatial_size)
    device: str = "cpu"  # "cpu"|"cuda"|"cuda:0"|"mps"
    remap_mscoco_category: bool = False  # COCO-id remap; keep False for custom data

    # --- 2. Backbone (HGNetV2) ----------------------------------------------
    backbone: str = "hgnetv2_b4"  # variant hgnetv2_b0..b5
    backbone_pretrained: bool = True  # load ImageNet-pretrained backbone weights
    return_idx: list[int] = field(default_factory=lambda: [1, 2, 3])  # stages -> encoder
    freeze_at: int = -1  # freeze stages up to this index (-1 = none)
    freeze_stem_only: bool = False  # freeze only the stem (L/X/obj365)
    freeze_norm: bool = False  # freeze BatchNorm in the backbone
    use_lab: bool = False  # learnable-affine block; True for N/S/M
    backbone_local_dir: str | None = None  # local dir for cached backbone weights

    # --- 3. Encoder (HybridEncoder) -----------------------------------------
    hidden_dim: int = 256  # encoder/decoder embedding dim (128 for N, 384 for X)
    in_channels: list[int] = field(default_factory=lambda: [512, 1024, 2048])  # from backbone
    feat_strides: list[int] = field(default_factory=lambda: [8, 16, 32])  # pyramid strides
    use_encoder_idx: list[int] = field(default_factory=lambda: [2])  # levels that run AIFI
    encoder_layers: int = 1  # AIFI transformer layers (num_encoder_layers)
    nhead: int = 8  # encoder attention heads
    encoder_dim_feedforward: int = 1024  # AIFI FFN dim (2048 for X, 512 for N)
    encoder_dropout: float = 0.0  # encoder dropout
    enc_act: str = "gelu"  # AIFI activation
    encoder_expansion: float = 1.0  # CCFM/GELAN channel expansion (0.5 S, 0.34 N)
    depth_mult: float = 1.0  # GELAN depth multiplier (0.34 S, 0.67 M, 0.5 N)
    encoder_act: str = "silu"  # fusion/GELAN activation (upstream `act`)

    # --- 4. Decoder (DFINETransformer) --------------------------------------
    decoder_hidden_dim: int | None = None  # decoder embed dim; defaults to hidden_dim
    # (256 for X, where the encoder runs at 384 but the decoder stays 256)
    num_queries: int = 300  # object queries
    decoder_dim_feedforward: int = 1024  # decoder FFN dim (512 for N; separate from enc)
    decoder_layers: int = 6  # decoder layers (num_layers; 4 for M, 3 for S/N)
    eval_idx: int = -1  # layer used at eval; negative = from end
    num_levels: int = 3  # multi-scale feature levels
    feat_channels: list[int] = field(default_factory=lambda: [256, 256, 256])  # per-level ch
    num_points: list[int] = field(default_factory=lambda: [3, 6, 3])  # deformable pts/level
    decoder_nhead: int = 8  # decoder attention heads
    decoder_offset_scale: float = 0.5  # deformable-attn offset scale
    decoder_method: str = "default"  # "default"|"discrete" deformable sampling
    query_select_method: str = "default"  # "default"|"agnostic" query selection
    layer_scale: float = 1.0  # hidden-dim scale for later decoder layers

    # 4a. Fine-grained Distribution Refinement (FDR)
    reg_max: int = 32  # bins per box edge in the regression distribution
    reg_scale: float = 4.0  # weighting-function scale (8 for X / obj365)

    # 4b. Location Quality Estimator (LQE)
    lqe_hidden_dim: int = 64  # LQE MLP hidden dim
    lqe_layers: int = 2  # LQE MLP layers

    # --- 5. Denoising (contrastive DN queries) ------------------------------
    num_denoising: int = 100  # denoising queries
    label_noise_ratio: float = 0.5  # label-flip noise
    box_noise_scale: float = 1.0  # box perturbation scale

    # --- 6. Matcher (Hungarian) — training only -----------------------------
    cost_class: float = 2.0  # classification match cost
    cost_bbox: float = 5.0  # L1 box match cost
    cost_giou: float = 2.0  # GIoU match cost
    matcher_alpha: float = 0.25  # matcher focal alpha
    matcher_gamma: float = 2.0  # matcher focal gamma

    # --- 7. Losses (DFINECriterion) — training only -------------------------
    loss_vfl_weight: float = 1.0  # varifocal classification loss
    loss_bbox_weight: float = 5.0  # L1 box loss
    loss_giou_weight: float = 2.0  # GIoU loss
    loss_fgl_weight: float = 0.15  # fine-grained localization (DFL) loss
    loss_ddf_weight: float = 1.5  # GO-LSD decoupled distillation loss
    focal_alpha: float = 0.75  # VFL alpha
    focal_gamma: float = 2.0  # VFL gamma
    aux_loss: bool = True  # supervise auxiliary decoder layers

    # --- 8. Postprocessor ----------------------------------------------------
    num_top_queries: int = 300  # top-k detections kept
    conf: float = 0.4  # default score threshold at predict time

    # --- 9. Training / optimization -----------------------------------------
    epochs: int = 72  # total epochs (includes no-aug tail)
    batch: int = 32  # total batch size
    lr: float = 2.5e-4  # base LR
    lr_backbone: float = 1.25e-5  # backbone LR
    weight_decay: float = 1.25e-4  # AdamW weight decay
    betas: tuple[float, float] = (0.9, 0.999)  # AdamW betas
    clip_max_norm: float = 0.1  # grad clip
    warmup_iters: int = 500  # LR warmup iterations
    scheduler: str = "flatcosine"  # "flatcosine"(default; adds a cosine no-aug tail —
    # intentional deviation from upstream's effectively-flat MultiStepLR)|"multistep"
    ema_decay: float = 0.9999  # weight-EMA decay
    ema_warmups: int = 1000  # EMA warmup steps
    use_amp: bool = True  # mixed precision
    no_aug_epoch: int = 2  # trailing epochs with advanced augs off
    seed: int = 0  # RNG seed
    workers: int = 4  # dataloader workers
    checkpoint_freq: int = 1  # save every N epochs

    # --- 10. Augmentation ----------------------------------------------------
    aug_photometric: bool = True  # RandomPhotometricDistort
    aug_zoom_out: bool = True  # RandomZoomOut
    aug_iou_crop: bool = True  # RandomIoUCrop
    aug_hflip: bool = True  # RandomHorizontalFlip
    multiscale: bool = True  # RandomMultiScaleInput
    base_size: int = 640  # multi-scale base size
    base_size_repeat: int | None = 3  # multi-scale repeat factor

    # ------------------------------------------------------------------ API --
    def __post_init__(self) -> None:
        if self.decoder_hidden_dim is None:
            object.__setattr__(self, "decoder_hidden_dim", self.hidden_dim)
        self._validate()

    @classmethod
    def preset(cls, size: str, **overrides: Any) -> DFINEConfig:
        """Build a config from a size preset, then apply inline overrides.

        Args:
            size: one of ``"n" | "s" | "m" | "l" | "x"``.
            **overrides: any field name to override the preset value.
        """
        key = size.lower()
        if key not in SIZE_PRESETS:
            raise ValueError(f"Unknown size {size!r}; expected one of {SIZES}.")
        merged: dict[str, Any] = {"size": key, **SIZE_PRESETS[key], **overrides}
        return cls(**merged)

    def override(self, **changes: Any) -> DFINEConfig:
        """Return a copy with ``changes`` applied (re-runs validation)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view of all fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DFINEConfig:
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------- validation --
    def _validate(self) -> None:
        if self.size is not None and self.size not in SIZE_PRESETS:
            raise ValueError(f"size must be one of {SIZES} or None, got {self.size!r}.")
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}.")
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but num_classes="
                f"{self.num_classes}."
            )
        if not self.backbone.startswith("hgnetv2_b") or self.backbone[-1] not in "012345":
            raise ValueError(f"backbone must be hgnetv2_b0..b5, got {self.backbone!r}.")

        n = self.num_levels
        for name in ("in_channels", "feat_strides", "feat_channels", "num_points", "return_idx"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"len({name})={got} must equal num_levels={n}.")
        if any(i >= n for i in self.use_encoder_idx):
            raise ValueError(f"use_encoder_idx {self.use_encoder_idx} out of range for {n} levels.")

        for name in ("hidden_dim", "num_queries", "reg_max", "imgsz", "batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}.")
        if self.reg_scale <= 0:
            raise ValueError(f"reg_scale must be > 0, got {self.reg_scale}.")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf must be in [0, 1], got {self.conf}.")
        if not -self.decoder_layers <= self.eval_idx < self.decoder_layers:
            raise ValueError(
                f"eval_idx={self.eval_idx} out of range for decoder_layers={self.decoder_layers}."
            )


def list_presets() -> tuple[str, ...]:
    """Return the available preset size names."""
    return SIZES


# -------------------------------------------------------------------------------
# Size presets — verified against D-FINE/configs/dfine/dfine_hgnetv2_{n,s,m,l,x}_coco.yml
# plus the shared include/dfine_hgnetv2.yml + include/optimizer.yml. Only fields that
# differ from the dataclass defaults (the "L-ish" base) are listed per size.
# -------------------------------------------------------------------------------
SIZE_PRESETS: dict[str, dict[str, Any]] = {
    # Nano: 2-level pyramid, hidden_dim 128 — structurally different from the rest.
    "n": {
        "backbone": "hgnetv2_b0",
        "use_lab": True,
        "return_idx": [2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "in_channels": [512, 1024],
        "feat_strides": [16, 32],
        "hidden_dim": 128,
        "use_encoder_idx": [1],
        "encoder_dim_feedforward": 512,
        "decoder_dim_feedforward": 512,
        "encoder_expansion": 0.34,
        "depth_mult": 0.5,
        "feat_channels": [128, 128],
        "num_levels": 2,
        "num_points": [6, 6],
        "decoder_layers": 3,
        "lr": 8e-4,
        "lr_backbone": 4e-4,
        "weight_decay": 1e-4,
        "epochs": 160,
        "no_aug_epoch": 12,
        "batch": 128,
        "base_size_repeat": None,
    },
    "s": {
        "backbone": "hgnetv2_b0",
        "use_lab": True,
        "return_idx": [1, 2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "in_channels": [256, 512, 1024],
        "hidden_dim": 256,
        "depth_mult": 0.34,
        "encoder_expansion": 0.5,
        "decoder_layers": 3,
        "lr": 2e-4,
        "lr_backbone": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 132,
        "no_aug_epoch": 12,
        "base_size_repeat": 20,
    },
    "m": {
        "backbone": "hgnetv2_b2",
        "use_lab": True,
        "return_idx": [1, 2, 3],
        "freeze_at": -1,
        "freeze_norm": False,
        "in_channels": [384, 768, 1536],
        "hidden_dim": 256,
        "depth_mult": 0.67,
        "decoder_layers": 4,
        "lr": 2e-4,
        "lr_backbone": 2e-5,
        "weight_decay": 1e-4,
        "epochs": 132,
        "no_aug_epoch": 12,
        "base_size_repeat": 6,
    },
    "l": {
        "backbone": "hgnetv2_b4",
        "use_lab": False,
        "return_idx": [1, 2, 3],
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "in_channels": [512, 1024, 2048],
        "hidden_dim": 256,
        "depth_mult": 1.0,
        "encoder_expansion": 1.0,
        "decoder_layers": 6,
        "lr": 2.5e-4,
        "lr_backbone": 1.25e-5,
        "weight_decay": 1.25e-4,
        "epochs": 80,
        "no_aug_epoch": 8,
        "base_size_repeat": 4,
    },
    "x": {
        "backbone": "hgnetv2_b5",
        "use_lab": False,
        "return_idx": [1, 2, 3],
        "freeze_stem_only": True,
        "freeze_at": 0,
        "freeze_norm": True,
        "in_channels": [512, 1024, 2048],
        "hidden_dim": 384,
        "decoder_hidden_dim": 256,  # decoder stays 256 while the encoder runs at 384
        "encoder_dim_feedforward": 2048,
        "feat_channels": [384, 384, 384],
        "reg_scale": 8.0,
        "decoder_layers": 6,
        "lr": 2.5e-4,
        "lr_backbone": 2.5e-6,
        "weight_decay": 1.25e-4,
        "epochs": 80,
        "no_aug_epoch": 8,
        "base_size_repeat": 3,
    },
}
