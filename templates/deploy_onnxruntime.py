"""Run an exported pydfine ONNX graph with onnxruntime — no torch, no pydfine at serve time.

    python export_onnx.py --model dfine-s --file dfine-s.onnx      # 1) export once
    python deploy_onnxruntime.py --onnx dfine-s.onnx --image street.jpg --imgsz 640

The detection graph fuses the postprocessor: it takes (images, orig_target_sizes) and
returns (labels, boxes, scores) already scaled to the original image, so a runtime only
feeds pixels + the original (W, H) and reads detections back. Preprocessing here mirrors
DFINE.predict exactly (resize-to-square, /255, CHW) so results match the torch path.

Needs: pip install onnxruntime pillow numpy   (the graph itself was built with pydfine[export])
"""

from __future__ import annotations

import argparse


def preprocess(image_path: str, imgsz: int):
    """PIL image -> (NCHW float32 in [0,1], orig_target_sizes int64 [[W, H]])."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.width, img.height
    resized = img.resize((imgsz, imgsz), Image.BILINEAR)  # matches T.Resize(BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC [0,1]
    chw = arr.transpose(2, 0, 1)[None]  # 1CHW
    sizes = np.array([[w0, h0]], dtype=np.int64)  # graph was traced with int64 (W, H)
    return chw, sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="exported .onnx graph (detect task)")
    ap.add_argument("--image", required=True, help="image to run detection on")
    ap.add_argument("--imgsz", type=int, default=640, help="MUST match the model's build imgsz")
    ap.add_argument("--conf", type=float, default=0.4, help="score threshold")
    ap.add_argument("--names", nargs="+", default=None, help="optional class names for printing")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install the runtime first: pip install onnxruntime") from e

    images, sizes = preprocess(args.image, args.imgsz)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    labels, boxes, scores = sess.run(None, {"images": images, "orig_target_sizes": sizes})

    # Each output is [N_images, num_top_queries]; take image 0 and threshold on score.
    labels, boxes, scores = labels[0], boxes[0], scores[0]
    keep = scores >= args.conf
    print(f"{int(keep.sum())} detections >= {args.conf}")
    for cls_id, box, score in zip(labels[keep], boxes[keep], scores[keep]):
        name = (
            args.names[int(cls_id)] if args.names and int(cls_id) < len(args.names) else int(cls_id)
        )
        x1, y1, x2, y2 = (round(float(v), 1) for v in box)
        print(f"  {name:>12}  {float(score):.2f}  [{x1}, {y1}, {x2}, {y2}]")


if __name__ == "__main__":
    main()
