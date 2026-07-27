"""Train instance (`segment`) or semantic (`sem_seg`) segmentation from a YOLO-style root.

    # instance segmentation, fine-tuned from released weights
    python train_segmentation.py --task segment --data dataset/ --model dfine-seg-l

    # semantic segmentation from scratch (e.g. 19-class Cityscapes)
    python train_segmentation.py --task sem_seg --data dataset/ --model l --num-classes 19

Dataset layout (labels resolved next to images by swapping /images/ -> /labels/):

    dataset/
      images/  a.jpg  b.jpg  …          # or images/{train,val}/ for an explicit split
      labels/  a.txt  b.txt  …          # .txt polygons for segment · .png maps for sem_seg

A flat root is split deterministically (--val-split, default 0.2); images/{train,val}
subdirs are used verbatim. Each epoch the val split is scored with mask AP (segment) or
mIoU (sem_seg), and the best checkpoint is saved to output-dir/best.pth.

Needs: pip install pydfine[train]   (add pydfine[hf] to fine-tune dfine-seg-* weights)
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["segment", "sem_seg"], help="which head")
    ap.add_argument("--data", required=True, help="YOLO-style dataset root (images/ + labels/)")
    ap.add_argument(
        "--model",
        default="l",
        help="size preset n|s|m|l|x (from scratch), or a dfine-seg-* checkpoint (segment only)",
    )
    ap.add_argument("--num-classes", type=int, default=80, help="class count (ignored for a ckpt)")
    ap.add_argument("--epochs", type=int, default=None, help="override the preset epoch count")
    ap.add_argument("--batch-size", type=int, default=8, help="per-step batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="training resolution (build-time)")
    ap.add_argument("--val-split", type=float, default=0.2, help="held-out fraction (flat root)")
    ap.add_argument("--devices", type=int, default=None, help="number of GPUs for DDP")
    ap.add_argument("--output-dir", default="runs/seg", help="checkpoints + logs")
    args = ap.parse_args()

    # A checkpoint name starts from released weights (mask head already trained — segment only);
    # a bare size builds fresh at your class count. task/num_classes/imgsz are build-time.
    if args.model in ("n", "s", "m", "l", "x"):
        model = DFINE(
            size=args.model, task=args.task, num_classes=args.num_classes, imgsz=args.imgsz
        )
    else:
        model = DFINE.from_pretrained(args.model)  # e.g. dfine-seg-l

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        devices=args.devices,
        output_dir=args.output_dir,
    )
    metric = "mask AP" if args.task == "segment" else "mIoU"
    print(f"training done -> {args.output_dir}  (best.pth = best {metric})")


if __name__ == "__main__":
    main()
