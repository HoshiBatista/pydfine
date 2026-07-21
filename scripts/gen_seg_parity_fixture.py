"""Generate the seg-parity fixture consumed by ``tests/test_seg_parity.py``.

Runs the *genuine* upstream D-FINE-seg (``D-FINE-seg/src`` via ``build_model``) for
a size on a deterministic seeded input, and saves its raw model outputs — detection
logits/boxes over all queries plus a small slice of instance-mask maps — to
``tests/data/seg_parity_<size>.pt``. The test then builds our native port from the
same checkpoint, feeds the same input, and asserts it reproduces these numbers. That
is the end-to-end proof that our mask branch computes what D-FINE-seg computes, not
merely that the seg weights load.

Like ``gen_parity_fixture.py`` this is a *developer* script, not part of the test run:
D-FINE-seg needs its own stack (loguru/hydra/…), which pydfine deliberately doesn't
depend on. Regenerate only when the port or the reference checkpoint legitimately
changes.

Usage (from repo root, with D-FINE-seg importable and the seg .pt cached):

    python scripts/gen_seg_parity_fixture.py n /path/to/dfine_seg_n_coco.pt
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "D-FINE-seg")

SEED = 0
IMGSZ = 640
ORIG_SIZE = [[640, 640]]
NUM_CLASSES = 80  # coco
MASK_K = 8  # store masks for the first K queries only — keeps the fixture ~0.4 MB


def main(size: str, ckpt_path: str) -> None:
    sys.path.insert(0, UPSTREAM)
    from src.d_fine.dfine import build_model

    # Build D-FINE-seg's own instance-seg model (its build_model does the nano
    # low-level-feat wiring), then strict-load the released checkpoint ourselves.
    model = build_model(
        model_name=size,
        num_classes=NUM_CLASSES,
        enable_mask_head=True,
        device="cpu",
        img_size=[IMGSZ, IMGSZ],
        task="segment",
    ).eval()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    torch.manual_seed(SEED)
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        out = model(x)

    masks = out["pred_masks"][0, :MASK_K]  # [K, H/4, W/4], sigmoid at eval
    fixture = {
        "size": size,
        "seed": SEED,
        "imgsz": IMGSZ,
        "orig_size": ORIG_SIZE,
        "mask_k": MASK_K,
        "pred_logits": out["pred_logits"].clone(),  # [1, Q, 80]
        "pred_boxes": out["pred_boxes"].clone(),  # [1, Q, 4], normalized cxcywh
        "pred_masks": masks.half().clone(),  # fp16 to keep the blob small
    }
    dst = os.path.join(REPO, "tests", "data", f"seg_parity_{size}.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(fixture, dst)
    print(
        f"wrote {dst}  (logits {tuple(out['pred_logits'].shape)}, "
        f"masks {tuple(out['pred_masks'].shape)} → stored {tuple(masks.shape)})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/gen_seg_parity_fixture.py <size> <checkpoint.pt>")
    main(sys.argv[1], sys.argv[2])
