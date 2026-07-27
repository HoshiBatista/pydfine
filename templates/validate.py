"""Score COCO metrics on a validation set and render the analytics plot bundle.

    python validate.py --data path/to/coco --model dfine-l --plots

Returns the 12 standard COCO metrics (AP is the primary mAP@[.50:.95]). With --plots it
also writes a confusion matrix, P/R/F1-vs-confidence curves, a per-class AP table, and a
worst-predictions gallery under --output-dir.

Needs: pip install pydfine[train]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="COCO root with val/ + instances_val.json")
    ap.add_argument("--model", default="dfine-l", help="checkpoint name or a local model")
    ap.add_argument("--weights", default=None, help="local .pth to load onto a bare size")
    ap.add_argument("--remap", action="store_true", help="remap sparse MS-COCO ids (stock COCO)")
    ap.add_argument("--plots", action="store_true", help="also render the analytics bundle")
    ap.add_argument("--output-dir", default="runs/val", help="where --plots artifacts go")
    args = ap.parse_args()

    # Build the model matching the ground truth. For stock 80-class MS-COCO, build with
    # remap so predicted labels align with the annotations' sparse ids.
    if args.weights:
        # a bare size + your own weights (e.g. from training)
        model = DFINE(size=args.model, remap_mscoco_category=args.remap).load(args.weights)
    else:
        model = DFINE.from_pretrained(args.model, remap_mscoco_category=args.remap)

    metrics = model.val(
        data=args.data,
        remap_mscoco_category=args.remap,
        plots=args.plots,
        output_dir=args.output_dir,
    )

    # Keys are the COCO_STAT_NAMES the evaluator returns (note the underscores).
    for key in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
        print(f"  {key:<10} {metrics[key]:.4f}")
    if args.plots:
        print(f"analytics -> {args.output_dir}/ (confusion_matrix, pr_curve, f1_curve, worst/)")


if __name__ == "__main__":
    main()
