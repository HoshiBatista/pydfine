"""Convert a YOLO detection dataset to COCO, then fine-tune on it — end to end.

    python train_from_yolo.py --yolo path/to/yolo --coco path/to/coco --model s

yolo_to_coco reads the standard YOLO layout (images/<split> + labels/<split>, plus an
optional data.yaml for class names and split paths — including Roboflow's valid/ split)
and writes the COCO layout DFINE.train consumes, with 0-indexed categories that already
line up with the model's contiguous labels.

Needs: pip install pydfine[train]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfine import DFINE, yolo_to_coco


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yolo", required=True, help="YOLO dataset root")
    ap.add_argument("--coco", required=True, help="output COCO root (created)")
    ap.add_argument("--model", default="s", help="size preset n|s|m|l|x")
    ap.add_argument("--epochs", type=int, default=None, help="override the preset epoch count")
    ap.add_argument("--batch-size", type=int, default=4, help="per-step batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="training resolution")
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    args = ap.parse_args()

    # 1) Convert once. Returns {split: annotation_json_path}. class_names come from data.yaml
    #    (or pass class_names=[...] here); ids stay 0-indexed to match the model's labels.
    written = yolo_to_coco(args.yolo, args.coco, copy_images=not args.symlink)
    print("converted splits:", written)

    # Read the class count straight from the annotations so the head matches the data.
    train_json = json.loads(Path(written["train"]).read_text())
    num_classes = len(train_json["categories"])
    print(f"{num_classes} classes")

    # 2) Train on the freshly written COCO root.
    model = DFINE(size=args.model, num_classes=num_classes, imgsz=args.imgsz)
    model.train(data=args.coco, epochs=args.epochs, batch_size=args.batch_size)

    if "val" in written:
        metrics = model.val(data=args.coco)
        print(f"AP={metrics['AP']:.4f}")


if __name__ == "__main__":
    main()
