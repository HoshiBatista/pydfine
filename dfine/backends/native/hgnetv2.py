"""HGNetV2 backbone — native D-FINE port.

Ported from ``D-FINE/src/nn/backbone/hgnetv2.py`` (Apache-2.0, © 2024 The D-FINE
Authors), which in turn adapts PaddleDetection's PP-HGNetV2. Changes from upstream:

- Dropped the ``@register()`` decorator and the ``src.core`` registry import.
- Replaced ``print`` calls and ``exit()`` on download failure with ``logging``
  (non-fatal: falls back to random init so a model can still be built offline).
- Added :meth:`HGNetv2.from_config` and public-name normalisation
  (``"hgnetv2_b4"`` -> ``"B4"``), plus ``out_channels``/``out_strides`` helpers.

Layer and parameter names are kept identical to upstream so the released
``PPHGNetV2_*_stage1.pth`` checkpoints load without a remap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .common import FrozenBatchNorm2d

if TYPE_CHECKING:
    from ...config import DFINEConfig

logger = logging.getLogger(__name__)

__all__ = ["HGNetv2"]


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: str = "",
        use_act: bool = True,
        use_lab: bool = False,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.ReLU() if self.use_act else nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lab(self.act(self.bn(self.conv(x))))


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        groups: int = 1,
        use_lab: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_chs: int, mid_chs: int, out_chs: int, use_lab: bool = False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs: int):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_lab: bool = False,
        agg: str = "ese",
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(
                total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        block_num: int,
        layer_num: int,
        downsample: bool = True,
        light_block: bool = False,
        kernel_size: int = 3,
        use_lab: bool = False,
        agg: str = "se",
        drop_path: float = 0.0,
    ):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetv2(nn.Module):
    """PP-HGNetV2 conv backbone (variants B0–B6).

    Args:
        name: variant, accepts ``"B0".."B6"`` or the public ``"hgnetv2_b0".."b6"``.
        use_lab: use ``LearnableAffineBlock`` (True for the light N/S/M presets).
        return_idx: which stages (0–3) to emit as the feature pyramid.
        freeze_stem_only: when freezing, freeze only the stem, not whole stages.
        freeze_at: freeze stages up to this index (-1 = freeze nothing).
        freeze_norm: replace BatchNorm with :class:`FrozenBatchNorm2d`.
        pretrained: load the ImageNet-pretrained stage-1 checkpoint.
        local_model_dir: cache dir for the download (None = torch hub default).
    """

    # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block,
    # kernel_size, layer_num
    arch_configs: dict[str, dict] = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth",
        },
        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth",
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
        },
        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth",
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth",
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth",
        },
        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth",
        },
    }

    def __init__(
        self,
        name: str,
        use_lab: bool = False,
        return_idx: list[int] | None = None,
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = True,
        local_model_dir: str | None = None,
    ):
        super().__init__()
        name = self._normalize_name(name)
        self.name = name
        self.use_lab = use_lab
        self.return_idx = [1, 2, 3] if return_idx is None else list(return_idx)

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]
        download_url = self.arch_configs[name]["url"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
        )

        self.stages = nn.ModuleList()
        for k in stage_config:
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                )
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            self._load_pretrained(download_url, local_model_dir)

    @classmethod
    def from_config(cls, cfg: DFINEConfig, *, return_idx: list[int] | None = None) -> HGNetv2:
        """Build the backbone from a :class:`DFINEConfig`.

        ``return_idx`` overrides ``cfg.return_idx`` (used by the segmentation path to
        emit an extra stride-8 level for the mask decoder) without mutating the config.
        """
        return cls(
            name=cfg.backbone,
            use_lab=cfg.use_lab,
            return_idx=cfg.return_idx if return_idx is None else return_idx,
            freeze_stem_only=cfg.freeze_stem_only,
            freeze_at=cfg.freeze_at,
            freeze_norm=cfg.freeze_norm,
            pretrained=cfg.backbone_pretrained,
            local_model_dir=cfg.backbone_local_dir,
        )

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        key = name.upper().replace("HGNETV2_", "")
        if key not in cls.arch_configs:
            raise ValueError(
                f"Unknown HGNetV2 variant {name!r}; expected one of {list(cls.arch_configs)}."
            )
        return key

    @property
    def out_channels(self) -> list[int]:
        """Channels of the emitted pyramid levels (in ``return_idx`` order)."""
        return [self._out_channels[i] for i in self.return_idx]

    @property
    def out_strides(self) -> list[int]:
        """Strides of the emitted pyramid levels (in ``return_idx`` order)."""
        return [self._out_strides[i] for i in self.return_idx]

    def _freeze_norm(self, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.BatchNorm2d):
            return FrozenBatchNorm2d(m.num_features)
        for name, child in m.named_children():
            new_child = self._freeze_norm(child)
            if new_child is not child:
                setattr(m, name, new_child)
        return m

    def _freeze_parameters(self, m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False

    def _load_pretrained(self, url: str, local_model_dir: str | None) -> None:
        kwargs = {"map_location": "cpu"}
        if local_model_dir:
            kwargs["model_dir"] = local_model_dir
        try:
            state = torch.hub.load_state_dict_from_url(url, **kwargs)
            self.load_state_dict(state)
            logger.info("Loaded pretrained HGNetV2 %s backbone.", self.name)
        except Exception as exc:  # network/offline — non-fatal for a library
            logger.warning(
                "Could not load pretrained HGNetV2 %s (%s). Continuing with random "
                "init; download it manually from %s if you need the weights.",
                self.name,
                exc,
                url,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
