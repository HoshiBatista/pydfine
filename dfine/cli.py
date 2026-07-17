"""``dfine`` command-line entrypoint.

``dfine models`` (inspect presets/checkpoints) and ``dfine convert`` (YAML-free, torch-
free) run in a base install. ``dfine export`` needs the export extra. ``predict``/
``train``/``val`` are declared so the surface is visible and report where they live.
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
        spec = CHECKPOINTS[name]
        print(f"  {name:<18} (size={spec.size} {spec.num_classes}cls)  {spec.url}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .convert import yolo_to_coco

    written = yolo_to_coco(
        args.yolo_root,
        args.output_dir,
        class_names=args.names,
        copy_images=not args.symlink,
    )
    for split, path in written.items():
        print(f"  {split:<10} -> {path}")
    print(f"COCO dataset written to {args.output_dir}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .model import DFINE

    if args.model.lower() in CHECKPOINTS:
        model = DFINE.from_pretrained(args.model.lower())
    else:
        # Build at the requested imgsz so the encoder's precomputed positional
        # embeddings (sized to cfg.imgsz) match the export resolution.
        overrides = {"imgsz": args.imgsz} if args.imgsz else {}
        model = DFINE(size=args.model, **overrides)
        if args.weights:
            model.load(args.weights)
    path = model.export(
        format="onnx",
        file=args.file,
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=not args.no_dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )
    print(f"exported ONNX -> {path}")
    return 0


def _cmd_stub(args: argparse.Namespace) -> int:
    where = _NOT_READY[args.command]
    print(f"`dfine {args.command}` is not implemented yet — arriving in {where}.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dfine", description="D-FINE object detection CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list size presets and known checkpoints")

    conv = sub.add_parser("convert", help="convert a YOLO dataset to the COCO layout")
    conv.add_argument("yolo_root", help="YOLO dataset root (images/<split> + labels/<split>)")
    conv.add_argument("output_dir", help="destination for the COCO train2017/…/annotations")
    conv.add_argument(
        "--names", nargs="+", default=None, help="class names (else read data.yaml / infer)"
    )
    conv.add_argument(
        "--symlink", action="store_true", help="symlink images instead of copying them"
    )

    exp = sub.add_parser("export", help="export a model to ONNX")
    exp.add_argument("model", help="a checkpoint name (e.g. dfine-s) or a size (n/s/m/l/x)")
    exp.add_argument("--weights", default=None, help="local .pth to load (for a bare size)")
    exp.add_argument("--file", default=None, help="output .onnx path (default dfine-<size>.onnx)")
    exp.add_argument("--imgsz", type=int, default=None, help="input resolution (default cfg.imgsz)")
    exp.add_argument("--batch", type=int, default=1, help="dummy trace batch size")
    exp.add_argument("--no-dynamic", action="store_true", help="fix the batch dim (no dynamic N)")
    exp.add_argument("--simplify", action="store_true", help="run onnxsim on the graph")
    exp.add_argument("--opset", type=int, default=16, help="ONNX opset version")

    for name in _NOT_READY:
        sub.add_parser(name, help=f"(coming soon) {name}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "convert":
        return _cmd_convert(args)
    return _cmd_stub(args)


if __name__ == "__main__":
    raise SystemExit(main())
