"""Turn predictions into other formats — DataFrame, COCO JSON, supervision, plain dicts.

    python results_interop.py --model dfine-s --image street.jpg --conf 0.4

`Results` is the single object every predict call returns; it exports to the ecosystems
you already use so pydfine drops into an existing pipeline without glue code.

Needs: pip install pydfine[torch]
  optional: pip install pydfine[interop]   (pandas + supervision)
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dfine-s", help="checkpoint name or size preset")
    ap.add_argument("--image", required=True, help="image to run detection on")
    ap.add_argument("--conf", type=float, default=0.4, help="score threshold")
    args = ap.parse_args()

    model = DFINE.from_pretrained(args.model)
    r = model.predict(args.image, conf=args.conf)[0]
    print(r.verbose())  # e.g. "3 persons, 1 car"

    # 1) Raw tensors (original-image pixel scale).
    print("boxes xyxy:", r.boxes.xyxy.tolist())
    print("scores    :", r.boxes.conf.tolist())
    print("class ids :", r.boxes.cls.tolist())

    # 2) JSON-serializable list of dicts — great for an API response.
    print("\nsummary():", r.summary(decimals=2))
    r.tojson()  # the same, as a JSON string

    # 3) COCO-format result dicts (loadRes layout) for pycocotools / faster-coco-eval.
    print("\nto_coco():", r.to_coco(image_id=1))

    # 4) pandas DataFrame (ultralytics-style .pandas().xyxy[0]) — needs pydfine[interop].
    try:
        df = r.to_pandas()
        print("\nto_pandas():\n", df.to_string(index=False))
    except ImportError:
        print("\n(pandas not installed — pip install pydfine[interop] for to_pandas())")

    # 5) supervision.Detections — plug into supervision's annotators / zones / trackers.
    try:
        dets = r.to_supervision()
        print("\nto_supervision():", dets)
    except ImportError:
        print("(supervision not installed — pip install pydfine[interop] for to_supervision())")


if __name__ == "__main__":
    main()
