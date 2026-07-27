"""Build a fully custom D-FINE — no preset — straight from typed params.

    python custom_architecture.py

The whole architecture (backbone, encoder, decoder, FDR head, denoising) is defined by
Python kwargs on one class. This mirrors the README's "fully custom" example and is the
config-first design in a nutshell: no YAML, no registry. See docs/CONFIG_REFERENCE.md
for every parameter and its default.

Needs: pip install pydfine[torch]
"""

from __future__ import annotations

from dfine import DFINE, DFINEConfig


def build_model() -> DFINE:
    # Option A — kwargs straight on DFINE (each one is a typed DFINEConfig field).
    model = DFINE(
        num_classes=3,
        class_names=["cat", "dog", "bird"],
        backbone="hgnetv2_b0",
        use_lab=True,
        freeze_at=-1,
        hidden_dim=256,
        encoder_dim_feedforward=1024,
        encoder_layers=1,
        nhead=8,
        decoder_layers=4,
        eval_idx=-1,
        num_levels=3,
        num_points=[3, 6, 3],
        reg_max=32,
        reg_scale=4.0,
        lqe_layers=2,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        imgsz=640,
    )
    return model


def build_from_preset() -> DFINE:
    # Option B — start from a preset and override just what you need. `preset` fills the
    # verified upstream defaults for that size; kwargs win over them.
    cfg = DFINEConfig.preset("s", num_classes=3, class_names=["cat", "dog", "bird"])
    return DFINE(config=cfg)


def main() -> None:
    model = build_model()
    print(repr(model))
    model.info(verbose=True)  # layers / params / GFLOPs + per-module breakdown

    preset_model = build_from_preset()
    print(repr(preset_model))


if __name__ == "__main__":
    main()
