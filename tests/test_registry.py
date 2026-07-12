"""Offline tests for the checkpoint catalogue + 'which model to use' logic.

No network/weights — just the metadata: availability per size/dataset, the
num_classes wiring, and the config built for each checkpoint.
"""

from __future__ import annotations

import pytest

from dfine import DFINEConfig
from dfine.config import SIZES
from dfine.registry import (
    CHECKPOINTS,
    DATASET_NUM_CLASSES,
    available_datasets,
    config_for,
    list_checkpoints,
    resolve,
    resolve_weights,
)

# What upstream actually released (README weights tables): every size has COCO;
# only s/m/l/x add the Objects365 variants. N is COCO-only.
_EXPECTED = {
    "n": ["coco"],
    "s": ["coco", "obj2coco", "obj365"],
    "m": ["coco", "obj2coco", "obj365"],
    "l": ["coco", "obj2coco", "obj365"],
    "x": ["coco", "obj2coco", "obj365"],
}


@pytest.mark.parametrize("size", SIZES)
def test_available_datasets_matches_upstream(size):
    assert sorted(available_datasets(size)) == sorted(_EXPECTED[size])


def test_n_is_coco_only():
    assert available_datasets("n") == ["coco"]
    for dataset in ("obj2coco", "obj365"):
        with pytest.raises(ValueError, match="only"):
            resolve_weights("n", dataset)


def test_num_classes_by_dataset():
    assert DATASET_NUM_CLASSES == {"coco": 80, "obj2coco": 80, "obj365": 366}
    assert resolve_weights("s", "coco").num_classes == 80
    assert resolve_weights("s", "obj2coco").num_classes == 80
    assert resolve_weights("s", "obj365").num_classes == 366


def test_resolve_default_dataset_is_coco():
    assert resolve_weights("m").dataset == "coco"
    assert resolve_weights("m").name == "dfine-m"


def test_names_and_urls_well_formed():
    for name, spec in CHECKPOINTS.items():
        assert spec.name == name
        assert spec.size in SIZES
        assert spec.url.endswith(".pth")
        assert spec.url.endswith(spec.filename)
        assert spec.filename.startswith(f"dfine_{spec.size}_")


def test_l_obj2coco_uses_e25_asset():
    assert resolve_weights("l", "obj2coco").filename == "dfine_l_obj2coco_e25.pth"


def test_resolve_unknown_name_raises():
    with pytest.raises(KeyError):
        resolve("dfine-n-obj365")  # never released
    assert "dfine-n-obj365" not in list_checkpoints()


@pytest.mark.parametrize("name", ["dfine-s", "dfine-l-obj2coco", "dfine-x-obj365"])
def test_config_for_matches_checkpoint(name):
    spec = resolve(name)
    cfg = config_for(spec)
    assert isinstance(cfg, DFINEConfig)
    assert cfg.size == spec.size
    assert cfg.num_classes == spec.num_classes
    assert cfg.imgsz == 640  # matches released anchor buffers


def test_config_for_accepts_name_and_overrides():
    cfg = config_for("dfine-s", backbone_pretrained=False)
    assert cfg.num_classes == 80 and cfg.backbone_pretrained is False
