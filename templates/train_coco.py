"""Fine-tune a D-FINE preset on a COCO-format dataset root.

    python train_coco.py --data path/to/coco --model l --epochs 72

Expected layout (what DFINE.train(data=...) consumes):

    coco/
      train/                          # images
      val/                            # images (optional but recommended)
      annotations/
        instances_train.json
        instances_val.json

Category ids must be contiguous 0..N-1 (as `dfine convert` writes them). For *stock*
80-class MS-COCO with its sparse ids, add --remap. Multi-GPU is one flag: --devices N
spawns one DDP worker per GPU (no torchrun needed).

Needs: pip install pydfine[train]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="COCO dataset root")
    ap.add_argument("--model", default="l", help="size preset n|s|m|l|x, or a checkpoint name")
    ap.add_argument("--num-classes", type=int, default=80, help="number of classes in the dataset")
    ap.add_argument("--epochs", type=int, default=None, help="override the preset epoch count")
    ap.add_argument("--batch-size", type=int, default=4, help="per-step batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="training resolution (build-time)")
    ap.add_argument("--devices", type=int, default=None, help="number of GPUs for DDP")
    ap.add_argument("--remap", action="store_true", help="remap sparse MS-COCO ids (stock COCO)")
    ap.add_argument("--output-dir", default="runs/train", help="checkpoints + logs")
    ap.add_argument("--resume", action="store_true", help="resume from output-dir/last.pth")
    ap.add_argument("--val-plots", action="store_true", help="render analytics every epoch")
    args = ap.parse_args()

    # A bare size builds an ImageNet-backbone model; a checkpoint name starts from released
    # COCO weights. Set imgsz/num_classes at build time — they define the architecture.
    if args.model in ("n", "s", "m", "l", "x"):
        model = DFINE(size=args.model, num_classes=args.num_classes, imgsz=args.imgsz)
    else:
        model = DFINE.from_pretrained(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        devices=args.devices,
        remap_mscoco_category=args.remap,
        output_dir=args.output_dir,
        resume=args.resume,
        val_plots=args.val_plots,
    )
    # best.pth / last.pth are under output_dir; the trained (EMA) weights are live on `model`.
    print(f"training done -> {args.output_dir}")

    if args.data:  # quick sanity score on the val split, if present
        metrics = model.val(data=args.data, remap_mscoco_category=args.remap)
        print(f"AP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}")


if __name__ == "__main__":
    main()
