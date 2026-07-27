"""Annotate a video file frame-by-frame, optionally with persistent ByteTrack IDs.

    python predict_video.py input.mp4 --out annotated.mp4 --track

With --stream the script pulls per-frame Results in a Python loop instead of writing a
file (handy for custom logic — counting, cropping, sending frames elsewhere).

Needs: pip install pydfine[video]   (and pydfine[track] for --track)
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--model", default="dfine-s", help="checkpoint name (see `dfine models`)")
    ap.add_argument("--out", default="output.mp4", help="annotated output path")
    ap.add_argument("--conf", type=float, default=0.25, help="score threshold")
    ap.add_argument("--track", action="store_true", help="assign persistent IDs (ByteTrack)")
    ap.add_argument(
        "--stream", action="store_true", help="iterate frames instead of writing a file"
    )
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    model = DFINE.from_pretrained(args.model, device=args.device)

    if args.stream:
        # Generator of one Results per frame; nothing is written to disk.
        for i, result in enumerate(
            model.predict_video(args.video, conf=args.conf, stream=True, track=args.track)
        ):
            ids = result.boxes.id
            tag = "" if ids is None else f"  ids={ids.tolist()}"
            print(f"frame {i:05d}: {len(result.boxes)} object(s){tag}")
        return

    # Writes an annotated mp4 at the source resolution/fps; with track=True boxes carry a
    # stable #id across frames, colored per track.
    out = model.predict_video(args.video, output=args.out, conf=args.conf, track=args.track)
    print(f"annotated video -> {out}")


if __name__ == "__main__":
    main()
