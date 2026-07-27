"""Detect objects in a single image, inspect the results, and save an annotated copy.

    python predict_image.py street.jpg --model dfine-s --conf 0.4

Needs: pip install pydfine[torch]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="path to an image")
    ap.add_argument("--model", default="dfine-s", help="checkpoint name (see `dfine models`)")
    ap.add_argument("--conf", type=float, default=0.25, help="score threshold")
    ap.add_argument("--out", default="out.jpg", help="where to write the annotated image")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    # from_pretrained resolves the size + num_classes from the checkpoint, downloads the
    # weights (cached), and strict-loads them — one line to a ready detector.
    model = DFINE.from_pretrained(args.model, device=args.device)

    # predict returns one Results per image; we passed a single path, so take [0].
    result = model.predict(args.image, conf=args.conf)[0]

    # Boxes are in original-image pixels. .cls are integer class ids; map via result.names.
    print(result.verbose())  # e.g. "3 persons, 1 car"
    for xyxy, conf, cls in result.boxes:
        x1, y1, x2, y2 = (round(v) for v in xyxy.tolist())
        print(f"  {result.names[int(cls)]:<15} {float(conf):.2f}  [{x1}, {y1}, {x2}, {y2}]")

    saved = result.save(args.out)  # draws boxes + labels and writes the file
    print(f"annotated image -> {saved}")


if __name__ == "__main__":
    main()
