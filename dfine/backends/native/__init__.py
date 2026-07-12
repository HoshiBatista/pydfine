"""Native D-FINE modules (Path A port).

Each module is copied from upstream ``D-FINE/src`` with the registry/YAML layer
removed and a ``from_config`` constructor added. Layer/parameter names match
upstream so released checkpoints load without a remap.
"""

from __future__ import annotations

from .dfine import DFINE
from .dfine_decoder import DFINETransformer
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .loader import extract_state_dict, load_checkpoint
from .postprocessor import DFINEPostProcessor

__all__ = [
    "DFINE",
    "DFINEPostProcessor",
    "DFINETransformer",
    "HGNetv2",
    "HybridEncoder",
    "extract_state_dict",
    "load_checkpoint",
]
