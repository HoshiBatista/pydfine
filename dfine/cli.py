"""``dfine`` command-line entrypoint.

``dfine models`` (inspect presets/checkpoints) and ``dfine convert`` (YAML-free, torch-
free) run in a base install. ``dfine predict``/``train``/``val``/``export`` build a model
and need the inference deps (``pydfine[torch]``; ``export`` also needs ``pydfine[export]``).

The ``model`` argument on the model commands is either a checkpoint name (``dfine-s``)
— resolved + downloaded via :meth:`DFINE.from_pretrained` — or a bare size (``n``..``x``),
optionally with local ``--weights``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DFINEConfig, list_presets
from .registry import CHECKPOINTS, list_checkpoints


def _build_model(model_arg: str, weights: str | None = None, **overrides):
    """Build a :class:`DFINE` from a checkpoint name or a bare size (+ optional weights)."""
    from .model import DFINE

    if model_arg.lower() in CHECKPOINTS:
        return DFINE.from_pretrained(model_arg.lower(), **overrides)
    model = DFINE(size=model_arg, **overrides)
    if weights:
        model.load(weights)
    return model


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


def _cmd_predict(args: argparse.Namespace) -> int:
    model = _build_model(args.model, args.weights)
    results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, res in zip(args.source, results):
        dst = out_dir / f"{Path(src).stem}_pred.jpg"
        res.save(dst)
        print(f"  {Path(src).name}: {len(res)} detections -> {dst}")
    return 0


def _cmd_val(args: argparse.Namespace) -> int:
    model = _build_model(args.model, args.weights, remap_mscoco_category=args.remap)
    metrics = model.val(data=args.data)
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.4f}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    model = _build_model(args.model, args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        devices=args.devices,
    )
    print(f"training done -> {args.output_dir}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .model import DFINE

    if args.model.lower() in CHECKPOINTS:
        model = DFINE.from_pretrained(args.model.lower())
    else:
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


def _add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="a checkpoint name (e.g. dfine-s) or a size (n/s/m/l/x)")
    p.add_argument("--weights", default=None, help="local .pth to load (for a bare size)")


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

    pred = sub.add_parser("predict", help="detect objects in image(s) and save annotated output")
    _add_model_arg(pred)
    pred.add_argument("source", nargs="+", help="image path(s) to run detection on")
    pred.add_argument("--conf", type=float, default=0.25, help="score threshold")
    pred.add_argument("--imgsz", type=int, default=None, help="inference resolution")
    pred.add_argument("--output", default="runs/predict", help="output directory")

    val = sub.add_parser("val", help="evaluate COCO metrics on a dataset")
    _add_model_arg(val)
    val.add_argument(
        "--data", required=True, help="COCO dataset root (with val2017/ + annotations)"
    )
    val.add_argument(
        "--remap", action="store_true", help="remap to MS-COCO ids (for stock 80-class COCO GT)"
    )

    tr = sub.add_parser("train", help="fine-tune on a COCO dataset")
    _add_model_arg(tr)
    tr.add_argument("--data", required=True, help="COCO dataset root")
    tr.add_argument("--epochs", type=int, default=None, help="override the preset's epoch count")
    tr.add_argument("--batch-size", type=int, default=4, help="per-step batch size")
    tr.add_argument("--output-dir", default="runs/train", help="checkpoints/logs directory")
    tr.add_argument("--devices", type=int, default=None, help="number of GPUs (multi-GPU DDP)")

    exp = sub.add_parser("export", help="export a model to ONNX")
    _add_model_arg(exp)
    exp.add_argument("--file", default=None, help="output .onnx path (default dfine-<size>.onnx)")
    exp.add_argument("--imgsz", type=int, default=None, help="input resolution (default cfg.imgsz)")
    exp.add_argument("--batch", type=int, default=1, help="dummy trace batch size")
    exp.add_argument("--no-dynamic", action="store_true", help="fix the batch dim (no dynamic N)")
    exp.add_argument("--simplify", action="store_true", help="run onnxsim on the graph")
    exp.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    return parser


_COMMANDS = {
    "models": _cmd_models,
    "convert": _cmd_convert,
    "predict": _cmd_predict,
    "val": _cmd_val,
    "train": _cmd_train,
    "export": _cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
