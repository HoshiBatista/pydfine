"""Generate the upstream-parity fixture consumed by ``tests/test_parity.py``.

Runs the *genuine* upstream D-FINE (``D-FINE/src`` via ``YAMLConfig``) for a size
on a deterministic seeded input, and saves its raw model outputs + final
postprocessed detections to ``tests/data/parity_<size>.pt``. The test then builds
our native port from the same checkpoint and asserts it reproduces these numbers.

This is a *developer* script, not part of the test run: upstream needs its full
training stack (tensorboard/transformers/…), which we deliberately don't depend on.
Regenerate only when the port or the reference checkpoint legitimately changes.

Usage (from repo root, with the upstream deps installed and the COCO .pth cached):

    python scripts/gen_parity_fixture.py s /path/to/dfine_s_coco.pth
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "D-FINE")

# Config file per preset size.
_CONFIGS = {
    "n": "configs/dfine/dfine_hgnetv2_n_coco.yml",
    "s": "configs/dfine/dfine_hgnetv2_s_coco.yml",
    "m": "configs/dfine/dfine_hgnetv2_m_coco.yml",
    "l": "configs/dfine/dfine_hgnetv2_l_coco.yml",
    "x": "configs/dfine/dfine_hgnetv2_x_coco.yml",
}

SEED = 0
IMGSZ = 640
ORIG_SIZE = [[640, 640]]  # square: postprocessor boxes come back in the 640 frame


def main(size: str, ckpt_path: str) -> None:
    sys.path.insert(0, UPSTREAM)
    from src.core import YAMLConfig

    cfg = YAMLConfig(os.path.join(UPSTREAM, _CONFIGS[size]))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    model = cfg.model.eval()
    pp = cfg.postprocessor.eval()
    # Our library standardizes on contiguous 0..79 labels (COCO names are mapped
    # separately, on the public path). The COCO YAML turns on remap_mscoco_category,
    # which would emit 1..90 category ids instead; force it off so both sides speak
    # the same label space and the comparison is apples-to-apples.
    pp.remap_mscoco_category = False

    torch.manual_seed(SEED)
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        out = model(x)
        det = pp(out, torch.tensor(ORIG_SIZE))[0]

    # Store raw pred_boxes (all 300 queries, normalized cxcywh) + the final
    # postprocessed detections. Raw pred_logits (1,300,80 ≈ 96 KB) is intentionally
    # dropped: the final labels/scores are argmax/sigmoid+topk over exactly those
    # logits, so comparing them tightly already pins the logit path — no need to
    # commit a 100 KB blob per size.
    fixture = {
        "size": size,
        "seed": SEED,
        "imgsz": IMGSZ,
        "orig_size": ORIG_SIZE,
        "pred_boxes": out["pred_boxes"].clone(),
        "labels": det["labels"].clone(),
        "boxes": det["boxes"].clone(),
        "scores": det["scores"].clone(),
    }
    dst = os.path.join(REPO, "tests", "data", f"parity_{size}.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(fixture, dst)
    n = int((fixture["scores"] > 0.3).sum())
    print(f"wrote {dst}  (raw {tuple(out['pred_logits'].shape)}; {n} detections >0.3)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/gen_parity_fixture.py <size> <checkpoint.pth>")
    main(sys.argv[1], sys.argv[2])
