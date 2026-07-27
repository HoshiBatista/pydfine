"""Measure forward-pass latency/FPS and print a model summary.

    python benchmark_and_info.py --model dfine-s --batch 1 --runs 100

benchmark() times the compute-bound model forward (postprocessor excluded) on random
input at the model's resolution; info() reports layer/param/gradient counts (and GFLOPs
if `thop` is installed).

Needs: pip install pydfine[torch]
"""

from __future__ import annotations

import argparse

from dfine import DFINE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dfine-s", help="checkpoint name or a size preset")
    ap.add_argument("--batch", type=int, default=1, help="batch size")
    ap.add_argument("--runs", type=int, default=100, help="timed forward passes")
    ap.add_argument("--warmup", type=int, default=10, help="untimed warmup passes")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    if args.model in ("n", "s", "m", "l", "x"):
        model = DFINE(size=args.model, device=args.device)
    else:
        model = DFINE.from_pretrained(args.model, device=args.device)

    summary = model.info(verbose=True)
    stats = model.benchmark(runs=args.runs, warmup=args.warmup, batch=args.batch)

    print(f"\n{model!r}")
    print(f"  params : {summary['parameters']:,}")
    print(f"  GFLOPs : {summary['gflops']}")
    print(f"  speed  : {stats['ms_per_image']:.2f} ms/image  ({stats['fps']:.1f} FPS)")
    print(f"  device : {stats['device']}")


if __name__ == "__main__":
    main()
