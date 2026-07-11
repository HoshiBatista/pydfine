"""``dfine`` command-line entrypoint.

Only ``dfine models`` is functional in Phase 0/1 (it inspects presets/checkpoints and
needs no torch). ``predict``/``train``/``val``/``export`` are declared so the surface
is visible, and report that they arrive with the inference/training backends.
"""

from __future__ import annotations

import argparse
import sys

from .config import DFINEConfig, list_presets
from .registry import CHECKPOINTS, list_checkpoints

_NOT_READY = {
    "predict": "Phase 2 (backend + inference)",
    "train": "Phase 4 (training)",
    "val": "Phase 4 (validation)",
    "export": "Phase 3 (export)",
}


def _cmd_models(_: argparse.Namespace) -> int:
    print("Size presets:")
    for size in list_presets():
        c = DFINEConfig.preset(size)
        print(
            f"  {size:<2} backbone={c.backbone:<11} hidden_dim={c.hidden_dim:<4} "
            f"levels={c.num_levels} decoder_layers={c.decoder_layers}"
        )
    print("\nCheckpoints:")
    for name in list_checkpoints():
        url, size = CHECKPOINTS[name]
        print(f"  {name:<18} (size={size})  {url}")
    return 0


def _cmd_stub(args: argparse.Namespace) -> int:
    where = _NOT_READY[args.command]
    print(f"`dfine {args.command}` is not implemented yet — arriving in {where}.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dfine", description="D-FINE object detection CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list size presets and known checkpoints")
    for name in _NOT_READY:
        sub.add_parser(name, help=f"(coming soon) {name}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        return _cmd_models(args)
    return _cmd_stub(args)


if __name__ == "__main__":
    raise SystemExit(main())
