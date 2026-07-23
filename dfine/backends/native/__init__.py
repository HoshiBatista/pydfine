"""Native D-FINE modules (Path A port).

Each module is copied from upstream ``D-FINE/src`` with the registry/YAML layer
removed and a ``from_config`` constructor added. Layer/parameter names match
upstream so released checkpoints load without a remap.
"""

from __future__ import annotations

from .criterion import DFINECriterion
from .dfine import DFINE
from .dfine_decoder import DFINETransformer
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .loader import extract_state_dict, load_checkpoint
from .mask_decoder import MaskDecoder
from .matcher import HungarianMatcher
from .postprocessor import DFINEPostProcessor
from .sem_seg_criterion import SemSegCriterion
from .sem_seg_decoder import SemSegDecoder
from .sem_seg_postprocessor import SemSegPostProcessor

__all__ = [
    "DFINE",
    "DFINECriterion",
    "DFINEPostProcessor",
    "DFINETransformer",
    "HGNetv2",
    "HungarianMatcher",
    "HybridEncoder",
    "MaskDecoder",
    "SemSegCriterion",
    "SemSegDecoder",
    "SemSegPostProcessor",
    "extract_state_dict",
    "load_checkpoint",
]
