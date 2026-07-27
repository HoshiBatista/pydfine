"""Count objects crossing a line in a video — streaming ByteTrack with persistent IDs.

    python track_and_count.py --video traffic.mp4 --model dfine-s --line 0.5 --class-id 2

Streams per-frame Results (nothing is written), keeps each track's previous position,
and tallies a crossing when a track's center moves across a horizontal line at
`--line` (fraction of frame height). A stable ByteTrack ID means each object is counted
once. This is the skeleton for people-counting, traffic flow, etc.

Needs: pip install pydfine[track]   (torch + opencv + scipy)
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--model", default="dfine-s", help="checkpoint name or size preset")
    ap.add_argument("--conf", type=float, default=0.4, help="score threshold")
    ap.add_argument("--line", type=float, default=0.5, help="line height as a fraction of frame")
    ap.add_argument(
        "--class-id", type=int, default=None, help="only count this class id (else all)"
    )
    args = ap.parse_args()

    model = DFINE.from_pretrained(args.model)

    prev_y: dict[int, float] = {}  # track id -> last center-y (pixels)
    down = up = 0

    for result in model.predict_video(args.video, stream=True, track=True, conf=args.conf):
        line_y = args.line * result.orig_shape[0]  # orig_shape is (H, W)
        ids = result.boxes.id
        if ids is None:
            continue
        for i, (xyxy, _conf, cls) in enumerate(result.boxes):
            if args.class_id is not None and int(cls) != args.class_id:
                continue
            tid = int(ids[i])
            cy = float((xyxy[1] + xyxy[3]) / 2)  # box center-y
            was = prev_y.get(tid)
            if was is not None:
                if was < line_y <= cy:  # crossed downward
                    down += 1
                elif was > line_y >= cy:  # crossed upward
                    up += 1
            prev_y[tid] = cy

    print(f"crossings — down: {down}   up: {up}   net: {down - up}")


if __name__ == "__main__":
    main()
