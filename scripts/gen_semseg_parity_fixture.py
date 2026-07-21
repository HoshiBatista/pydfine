"""Generate the sem_seg-parity fixture consumed by ``tests/test_semseg_parity.py``.

Runs the *genuine* upstream D-FINE-seg ``SemSegDecoder`` (``D-FINE-seg/src`` ) on small
synthetic feature pyramids and stores, for two wiring cases (nano low-level-feat and a
plain stride-8 pyramid), the decoder weights, the input features, and the resulting
``sem_seg_logits``. The test then builds our native port, loads the *same* weights, feeds
the *same* inputs, and asserts it reproduces the logits bit-exactly.

The sem_seg forward is dimension-independent (the fuse/upsample steps are size/scale based),
so tiny channels/spatials exercise every path while keeping the fixture a few KB and fully
self-contained (no checkpoint, no seed-based regen). There are no released *trained* sem_seg
weights (``dfine_seg_*_coco.pt`` is instance-seg); the real trained fuser transfer is pinned
separately by the SS1/SS2 strict-load tests.

Like the other gen_* scripts this is a *developer* tool, not part of the test run.

Usage (from repo root, with D-FINE-seg importable):

    python scripts/gen_semseg_parity_fixture.py
"""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "D-FINE-seg")

# (name, num_classes, feat_channels, mask_dim, mask_low_level_ch, neck_dim, low_level_ch)
# mask_dim/neck_dim must be multiples of 32 (the fuser/neck use GroupNorm(32, ...)).
_CASES = [
    ("nano", 4, [6, 6], 32, 10, 32, 10),  # nano: extra stride-8 low-level feat prepended
    ("stride8", 5, [6, 6, 6], 32, None, 32, None),  # native 3-level pyramid, no low-level feat
]


def _feats(feat_channels, low_level_ch):
    """Finest-first synthetic pyramid (+ optional low-level feat), each level half-size."""
    base = 8
    low = None if low_level_ch is None else torch.randn(1, low_level_ch, base, base)
    off = 1 if low_level_ch is not None else 0
    feats = [
        torch.randn(1, c, base // (2 ** (i + off)), base // (2 ** (i + off)))
        for i, c in enumerate(feat_channels)
    ]
    return low, feats


def main() -> None:
    sys.path.insert(0, UPSTREAM)
    from src.d_fine.arch.dfine_decoder import SemSegDecoder

    torch.manual_seed(0)
    cases = []
    for name, ncls, fch, mdim, mllc, neck, low_ch in _CASES:
        dec = SemSegDecoder(
            num_classes=ncls,
            feat_channels=fch,
            mask_dim=mdim,
            mask_low_level_ch=mllc,
            neck_dim=neck,
        ).eval()
        low, feats = _feats(fch, low_ch)
        with torch.no_grad():
            out = dec(feats, low_level_feat=low)
        cases.append(
            {
                "name": name,
                "num_classes": ncls,
                "feat_channels": fch,
                "mask_dim": mdim,
                "mask_low_level_ch": mllc,
                "neck_dim": neck,
                "state": {k: v.clone() for k, v in dec.state_dict().items()},
                "low_level_feat": None if low is None else low.clone(),
                "feats": [f.clone() for f in feats],
                "sem_seg_logits": out["sem_seg_logits"].clone(),
            }
        )
        print(
            f"{name}: logits {tuple(out['sem_seg_logits'].shape)}, {len(cases[-1]['state'])} keys"
        )

    dst = os.path.join(REPO, "tests", "data", "semseg_parity.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save({"cases": cases}, dst)
    print(f"wrote {dst}  ({os.path.getsize(dst) // 1024} KB)")


if __name__ == "__main__":
    main()
