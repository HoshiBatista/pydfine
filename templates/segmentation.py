"""Instance and semantic segmentation inference with the same one-class façade.

    python segmentation.py instance street.jpg
    python segmentation.py semantic street.jpg

Predictions come back at the *original* image scale, ready to plot or export. Instance
weights ship from D-FINE-seg (auto-downloaded from Hugging Face).

Needs: pip install pydfine[hf]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def instance_seg(image: str, conf: float) -> None:
    # dfine-seg-{n,s,m,l,x}: masks + boxes, aligned 1:1.
    model = DFINE.from_pretrained("dfine-seg-l")
    r = model.predict(image, conf=conf)[0]
    print(r.verbose())
    print("boxes:", tuple(r.boxes.xyxy.shape))  # (N, 4) original-scale xyxy
    print("masks:", tuple(r.masks.data.shape))  # (N, H, W) bool, one per box
    r.save("instance_seg.jpg")  # boxes + per-instance mask overlays


def semantic_seg(image: str) -> None:
    # sem_seg: dense per-pixel label map, no boxes. num_classes matches your label set
    # (Cityscapes-style 19 here); the classifier/neck are trained on your own data.
    model = DFINE(size="l", task="sem_seg", num_classes=19)
    r = model.predict(image)[0]
    print("label map:", tuple(r.sem_seg.data.shape))  # (H, W) uint8 class ids (255 = void)
    r.save("semantic_seg.jpg")  # per-class color overlay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["instance", "semantic"])
    ap.add_argument("image", help="path to an image")
    ap.add_argument("--conf", type=float, default=0.4, help="score threshold (instance only)")
    args = ap.parse_args()

    if args.task == "instance":
        instance_seg(args.image, args.conf)
    else:
        semantic_seg(args.image)


if __name__ == "__main__":
    main()
