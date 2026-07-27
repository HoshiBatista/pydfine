"""Export a model to ONNX (or TorchScript) for deployment.

    python export_onnx.py --model dfine-s --format onnx --simplify

The exported graph takes (images, orig_target_sizes) and returns (labels, boxes,
scores) already scaled to the original image — the postprocessor is fused in, so a
runtime only needs to feed pixels and read detections.

Needs: pip install pydfine[export]   (TorchScript needs only torch)
"""

from __future__ import annotations

import argparse

from dfine import DFINE
from dfine.export import tensorrt_command


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dfine-s", help="checkpoint name or a size preset")
    ap.add_argument("--weights", default=None, help="local .pth for a bare size")
    ap.add_argument("--format", default="onnx", choices=["onnx", "torchscript"])
    ap.add_argument("--file", default=None, help="output path (default dfine-<size>.<ext>)")
    ap.add_argument("--batch", type=int, default=1, help="trace batch size")
    ap.add_argument("--no-dynamic", action="store_true", help="fix the batch dim (ONNX)")
    ap.add_argument("--simplify", action="store_true", help="run onnxsim on the graph (ONNX)")
    ap.add_argument("--opset", type=int, default=16, help="ONNX opset")
    args = ap.parse_args()

    if args.weights:
        model = DFINE(size=args.model).load(args.weights)
    else:
        model = DFINE.from_pretrained(args.model)

    path = model.export(
        format=args.format,
        file=args.file,
        batch=args.batch,
        dynamic=not args.no_dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )
    print(f"exported {args.format} -> {path}")

    if args.format == "onnx":
        # Print the trtexec line to build a TensorRT engine from this ONNX (FP16).
        print("\nBuild a TensorRT engine with:")
        print("  " + tensorrt_command(path, fp16=True))


if __name__ == "__main__":
    main()
