"""Save/load a model's architecture as YAML — reproducible configs, no code duplication.

    python config_as_yaml.py --size l --num-classes 3 --out model.yaml
    python config_as_yaml.py --from-yaml model.yaml           # rebuild from the file

pydfine is config-first: the whole architecture is a typed `DFINEConfig`. You can freeze
one to YAML for versioning/sharing, then rebuild the exact model with `DFINE(config=...)`.
This is interop only — the *user* path never needs YAML; presets + kwargs are enough.

Needs: pip install pydfine[torch]   (YAML I/O needs pyyaml, in pydfine[train])
"""

from __future__ import annotations

import argparse

from dfine import DFINE, DFINEConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-yaml", default=None, help="rebuild a model from this YAML and exit")
    ap.add_argument("--size", default="l", help="preset to start from (n|s|m|l|x)")
    ap.add_argument("--num-classes", type=int, default=80, help="class count")
    ap.add_argument("--imgsz", type=int, default=640, help="build-time resolution")
    ap.add_argument("--out", default="model.yaml", help="where to write the config YAML")
    args = ap.parse_args()

    if args.from_yaml:
        cfg = DFINEConfig.from_yaml(args.from_yaml)  # path, or a YAML string
        model = DFINE(config=cfg)  # exact architecture, no kwargs to remember
        print(f"rebuilt {model!r} from {args.from_yaml}")
        return

    # Build a config from a preset + overrides, then serialize it.
    cfg = DFINEConfig.preset(args.size, num_classes=args.num_classes, imgsz=args.imgsz)
    cfg.to_yaml(args.out)  # round-trips through from_yaml()
    print(f"wrote {args.out}")
    print(cfg.to_yaml()[:400], "...")  # preview the top of the file

    # Sanity check: the round-trip reproduces the same config.
    assert DFINEConfig.from_yaml(args.out).to_dict() == cfg.to_dict()
    print("round-trip OK — DFINE(config=DFINEConfig.from_yaml(...)) rebuilds this model")


if __name__ == "__main__":
    main()
