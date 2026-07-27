"""Run detection over a folder (or glob) and save annotated images, labels, and crops.

    python predict_folder.py ./images --model dfine-l --conf 0.3 --save-txt --save-crop

A directory source runs over every image in it (sorted); a glob like "imgs/*.jpg" runs
over the matches. Outputs land in a fresh, auto-incremented run dir (predict, predict2…).

Needs: pip install pydfine[torch]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="a folder, a glob ('imgs/*.jpg'), or a single image")
    ap.add_argument("--model", default="dfine-s", help="checkpoint name (see `dfine models`)")
    ap.add_argument("--conf", type=float, default=0.25, help="score threshold")
    ap.add_argument("--project", default="runs/detect", help="run directory root")
    ap.add_argument("--name", default="predict", help="run name (auto-incremented)")
    ap.add_argument("--save-txt", action="store_true", help="also write YOLO-format labels")
    ap.add_argument("--save-crop", action="store_true", help="also write per-detection crops")
    ap.add_argument("--save-conf", action="store_true", help="append confidence to --save-txt")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    model = DFINE.from_pretrained(args.model, device=args.device)

    # save*=True flags write into project/name/. predict still returns the Results list,
    # so you can keep working with the detections in-process afterwards.
    results = model.predict(
        args.source,
        conf=args.conf,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
    )

    total = sum(len(r.boxes) for r in results)
    print(f"{len(results)} image(s), {total} detection(s) total")


if __name__ == "__main__":
    main()
