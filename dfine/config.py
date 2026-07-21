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
from pathlib import Path
from typing import Any

SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class DFINEConfig:
    """All D-FINE model + training options as one typed, frozen dataclass.

    Build with a preset via :meth:`preset` (fills size-dependent fields) or construct
    directly for a fully custom architecture. Every field is overridable inline.
    """

    size: str | None = None
    task: str = "detect"
    num_classes: int = 80
    class_names: list[str] | None = None
    imgsz: int = 640
    device: str = "cpu"
    remap_mscoco_category: bool = False
    mask_dim: int = 256

    backbone: str = "hgnetv2_b4"
    backbone_pretrained: bool = True
    return_idx: list[int] = field(default_factory=lambda: [1, 2, 3])
    freeze_at: int = -1
    freeze_stem_only: bool = False
    freeze_norm: bool = False
    use_lab: bool = False
    backbone_local_dir: str | None = None

    hidden_dim: int = 256
    in_channels: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    feat_strides: list[int] = field(default_factory=lambda: [8, 16, 32])
    use_encoder_idx: list[int] = field(default_factory=lambda: [2])
    encoder_layers: int = 1
    nhead: int = 8
    encoder_dim_feedforward: int = 1024
    encoder_dropout: float = 0.0
    enc_act: str = "gelu"
    encoder_expansion: float = 1.0
    depth_mult: float = 1.0
    encoder_act: str = "silu"

    decoder_hidden_dim: int | None = None
    num_queries: int = 300
    decoder_dim_feedforward: int = 1024
    decoder_layers: int = 6
    eval_idx: int = -1
    num_levels: int = 3
    feat_channels: list[int] = field(default_factory=lambda: [256, 256, 256])
    num_points: list[int] = field(default_factory=lambda: [3, 6, 3])
    decoder_nhead: int = 8
    decoder_offset_scale: float = 0.5
    decoder_method: str = "default"
    query_select_method: str = "default"
    layer_scale: float = 1.0

    reg_max: int = 32
    reg_scale: float = 4.0

    lqe_hidden_dim: int = 64
    lqe_layers: int = 2

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0

    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0

    loss_vfl_weight: float = 1.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    loss_fgl_weight: float = 0.15
    loss_ddf_weight: float = 1.5
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    aux_loss: bool = True

    num_top_queries: int = 300
    conf: float = 0.4

    epochs: int = 72
    batch: int = 32
    lr: float = 2.5e-4
    lr_backbone: float = 1.25e-5
    weight_decay: float = 1.25e-4
    zero_wd_encdec_bias: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    clip_max_norm: float = 0.1
    warmup_iters: int = 500
    lr_milestones: list[int] | None = None
    lr_gamma: float = 0.1
    scheduler: str = "flatcosine"
    ema_decay: float = 0.9999
    ema_warmups: int = 1000
    use_amp: bool = True
    no_aug_epoch: int = 2
    seed: int = 0
    workers: int = 4
    checkpoint_freq: int = 1
    sync_bn: bool = True
    find_unused_parameters: bool = False

    aug_photometric: bool = True
    aug_zoom_out: bool = True
    aug_iou_crop: bool = True
    aug_hflip: bool = True
    multiscale: bool = True
    base_size: int = 640
    base_size_repeat: int | None = 3

    TASKS = ("detect", "segment", "sem_seg")

    def __post_init__(self) -> None:
        if self.decoder_hidden_dim is None:
            object.__setattr__(self, "decoder_hidden_dim", self.hidden_dim)
        self._validate()

    @property
    def enable_mask_head(self) -> bool:
        """Whether the decoder runs the instance-mask branch (task="segment")."""
        return self.task == "segment"

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
        kwargs = {k: v for k, v in data.items() if k in known}
        if isinstance(kwargs.get("betas"), list):
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)

    @staticmethod
    def _require_yaml():
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - trivial guard
            raise ImportError(
                "YAML interop needs PyYAML — install with `pip install pyyaml` or "
                "`pip install pydfine[train]`."
            ) from exc
        return yaml

    def to_yaml(self, path: str | Path | None = None) -> str | Path:
        """Serialize the config to YAML: return the string, or write it to ``path``.

        Round-trips through :meth:`from_yaml`. Tuples (e.g. ``betas``) are written as
        lists so the output is plain, safe YAML.
        """
        yaml = self._require_yaml()
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.to_dict().items()}
        text = yaml.safe_dump(data, sort_keys=False)
        if path is None:
            return text
        out = Path(path)
        out.write_text(text)
        return out

    @classmethod
    def from_yaml(cls, source: str | Path) -> DFINEConfig:
        """Build a config from a YAML file path or a YAML string (unknown keys ignored).

        ``source`` is a :class:`~pathlib.Path`, a path string ending in ``.yaml``/``.yml``,
        or the YAML text itself. Validation runs on construction.
        """
        yaml = cls._require_yaml()
        if isinstance(source, Path):
            text = source.read_text()
        elif (
            isinstance(source, str)
            and "\n" not in source
            and source.rstrip().endswith((".yaml", ".yml"))
        ):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"YAML config not found: {source!r}")
            text = p.read_text()
        else:
            text = source
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("from_yaml expected a YAML mapping (file path or YAML string).")
        return cls.from_dict(data)

    def _validate(self) -> None:
        if self.size is not None and self.size not in SIZE_PRESETS:
            raise ValueError(f"size must be one of {SIZES} or None, got {self.size!r}.")
        if self.task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}, got {self.task!r}.")
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


SIZE_PRESETS: dict[str, dict[str, Any]] = {
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
        "mask_dim": 128,
        "lr": 8e-4,
        "lr_backbone": 4e-4,
        "weight_decay": 1e-4,
        "zero_wd_encdec_bias": True,
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
        "zero_wd_encdec_bias": True,
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
        "zero_wd_encdec_bias": True,
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
        "decoder_hidden_dim": 256,
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
