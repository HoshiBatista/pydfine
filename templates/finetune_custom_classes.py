"""Train a detector for your own set of classes, starting from a pretrained backbone.

    python finetune_custom_classes.py --data path/to/coco --names cat dog bird

Build with `num_classes=N` (and matching `class_names=[...]` for readable labels). By
default the backbone is initialized from ImageNet weights (`backbone_pretrained=True`),
so you are fine-tuning strong features onto your classes, not training from scratch —
this is the recommended path for a custom class set.

(Released detector checkpoints like `dfine-s` can only be loaded whole, into a model
with the *same* class count — `model.load()` is strict. To reuse the full COCO detector,
keep `num_classes=80`; to train your own classes, use this ImageNet-backbone path.)

Needs: pip install pydfine[train]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="COCO dataset root (contiguous 0..N-1 ids)")
    ap.add_argument("--names", nargs="+", required=True, help="class names, in id order")
    ap.add_argument("--size", default="s", help="size preset n|s|m|l|x")
    ap.add_argument("--epochs", type=int, default=48, help="fine-tune epochs")
    ap.add_argument("--batch-size", type=int, default=8, help="per-step batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="training resolution")
    ap.add_argument("--predict", default=None, help="optional image to run + save after training")
    args = ap.parse_args()

    # num_classes defines the head; class_names must match it and gives readable labels.
    # backbone_pretrained defaults to True, so features start from ImageNet.
    model = DFINE(
        size=args.size,
        num_classes=len(args.names),
        class_names=args.names,
        imgsz=args.imgsz,
    )
    print(f"{model!r} — {len(args.names)} classes: {', '.join(args.names)}")

    model.train(data=args.data, epochs=args.epochs, batch_size=args.batch_size, val_plots=True)

    metrics = model.val(data=args.data, plots=True)
    print(f"AP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}")

    # Trained weights are live on `model`; best.pth/last.pth are under runs/train.
    if args.predict:  # optional: run the trained model on an image and save the annotated copy
        model.predict(args.predict, conf=0.4, save=True)
        print(f"prediction saved under runs/detect/ for {args.predict}")


if __name__ == "__main__":
    main()
