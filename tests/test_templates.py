"""Run every script in templates/ as a smoke test so the docs' copy-paste code can't rot.

The templates are the code users start from, so a broken one (a wrong metric key, a
hardcoded path, a renamed kwarg) is a real regression. Each is executed exactly as a user
would — via ``runpy`` with a real ``argv`` and tiny on-disk assets — but made offline and
fast by a ``tiny`` fixture that swaps ``DFINE`` for a nano, ImageNet-free, small-``imgsz``
build. No network, no downloads; the whole file adds a handful of seconds.

Needs the ``[dev]`` extra (torch + cv2 + onnx(runtime) + faster-coco-eval + torchmetrics +
scipy + pandas + supervision + pyyaml + matplotlib); each case ``importorskip``s what it uses.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


# --- the offline/fast model swap -------------------------------------------------------


@pytest.fixture
def tiny(monkeypatch):
    """Replace ``dfine.model.DFINE`` with a nano, offline, ``imgsz=320`` build.

    Keeps the template's task / num_classes / class_names (and custom arch) but forces a
    small, ImageNet-free model so every script runs on CPU in seconds with no download.
    """
    import dfine.model as dm
    from dfine.backends.native import hgnetv2

    real = dm.DFINE
    # never fetch pretrained backbone weights (random init is fine for a smoke test)
    monkeypatch.setattr(hgnetv2.HGNetv2, "_load_pretrained", lambda self, *a, **k: None)

    class TinyDFINE(real):
        def __init__(self, size=None, *, config=None, weights=None, device=None, **params):
            if config is not None:
                config = config.override(imgsz=320, backbone_pretrained=False, num_denoising=0)
                super().__init__(config=config, device=device)
            else:
                params.update(imgsz=320, backbone_pretrained=False, num_denoising=0)
                super().__init__(size, device=device, **params)

        @classmethod
        def from_pretrained(cls, name, device=None, **overrides):
            task = "segment" if "seg" in str(name) else "detect"
            nc = overrides.pop("num_classes", 2)
            remap = overrides.pop("remap_mscoco_category", False)
            return cls(task=task, num_classes=nc, remap_mscoco_category=remap, device=device)

    monkeypatch.setattr(dm, "DFINE", TinyDFINE)
    return TinyDFINE


def _run(name, argv, tmp_path, monkeypatch):
    """Execute templates/<name> as ``__main__`` with the given argv, in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [name, *argv])
    runpy.run_path(str(TEMPLATES / name), run_name="__main__")


# --- tiny on-disk assets (built only where a case needs them) --------------------------


def _rand_img(path, size=(128, 128)):
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(size[1], size[0], 3) * 255).astype("uint8")).save(path)


def make_image(root):
    _rand_img(root / "street.jpg")


def make_folder(root):
    _rand_img(root / "imgs" / "a.jpg")
    _rand_img(root / "imgs" / "b.png")


def make_video(root):
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(root / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128))
    for _ in range(8):
        vw.write((np.random.rand(128, 128, 3) * 255).astype("uint8"))
    vw.release()


def make_coco(root):
    base = root / "coco"
    for split in ("train", "val"):
        (base / "annotations").mkdir(parents=True, exist_ok=True)
        imgs, anns, aid = [], [], 1
        for i in range(1, 3):  # 2 imgs/split — one batch, minimal augmentation cost
            _rand_img(base / split / f"img{i}.jpg")
            imgs.append({"id": i, "file_name": f"img{i}.jpg", "width": 128, "height": 128})
            for c in (0, 1):
                anns.append(
                    {
                        "id": aid,
                        "image_id": i,
                        "category_id": c,
                        "bbox": [20, 20, 40, 40],
                        "area": 1600,
                        "iscrowd": 0,
                    }
                )
                aid += 1
        (base / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(
                {
                    "images": imgs,
                    "annotations": anns,
                    "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}],
                }
            )
        )


def make_yolo(root):
    base = root / "yolo"
    for split in ("train", "val"):
        for i in range(1, 3):
            _rand_img(base / "images" / split / f"i{i}.jpg")
            lbl = base / "labels" / split / f"i{i}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n")


def make_seg_inst(root):
    base = root / "seg_inst"
    for i in range(1, 5):
        _rand_img(base / "images" / f"i{i}.jpg")
        lbl = base / "labels" / f"i{i}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")


def make_seg_sem(root):
    import numpy as np

    base = root / "seg_sem"
    for i in range(1, 5):
        _rand_img(base / "images" / f"i{i}.jpg")
        m = np.zeros((128, 128), "uint8")
        m[:, 64:] = 1
        m[:64, :] = 2
        (base / "labels").mkdir(parents=True, exist_ok=True)
        Image.fromarray(m).save(base / "labels" / f"i{i}.png")


# --- the cases -------------------------------------------------------------------------

# (id, script, argv, needs-importorskip, asset-builders)
CASES = [
    ("predict_image", "predict_image.py", ["street.jpg", "--model", "dfine-n"], [], [make_image]),
    (
        "predict_folder",
        "predict_folder.py",
        ["imgs/", "--model", "dfine-n", "--save-txt", "--save-crop"],
        [],
        [make_folder],
    ),
    (
        "predict_video",
        "predict_video.py",
        ["clip.mp4", "--model", "dfine-n", "--out", "out.mp4"],
        ["cv2"],
        [make_video],
    ),
    (
        "results_interop",
        "results_interop.py",
        ["--model", "dfine-n", "--image", "street.jpg"],
        [],
        [make_image],
    ),
    (
        "benchmark_and_info",
        "benchmark_and_info.py",
        ["--model", "dfine-n", "--runs", "2", "--warmup", "1"],
        [],
        [],
    ),
    (
        "track_and_count",
        "track_and_count.py",
        ["--model", "dfine-n", "--video", "clip.mp4"],
        ["cv2", "scipy"],
        [make_video],
    ),
    ("custom_architecture", "custom_architecture.py", [], [], []),
    (
        "config_as_yaml",
        "config_as_yaml.py",
        ["--size", "n", "--num-classes", "2", "--out", "model.yaml"],
        ["yaml"],
        [],
    ),
    ("segmentation_instance", "segmentation.py", ["instance", "street.jpg"], [], [make_image]),
    ("segmentation_semantic", "segmentation.py", ["semantic", "street.jpg"], [], [make_image]),
    (
        "train_coco",
        "train_coco.py",
        [
            "--data",
            "coco",
            "--model",
            "n",
            "--num-classes",
            "2",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch-size",
            "2",
        ],
        ["faster_coco_eval", "scipy"],
        [make_coco],
    ),
    (
        "train_from_yolo",
        "train_from_yolo.py",
        [
            "--yolo",
            "yolo",
            "--coco",
            "cocoout",
            "--model",
            "n",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch-size",
            "2",
        ],
        ["faster_coco_eval", "scipy"],
        [make_yolo],
    ),
    (
        "finetune",
        "finetune_custom_classes.py",
        [
            "--data",
            "coco",
            "--names",
            "cat",
            "dog",
            "--size",
            "n",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch-size",
            "2",
        ],
        ["faster_coco_eval", "scipy", "matplotlib"],
        [make_coco],
    ),
    (
        "validate",
        "validate.py",
        ["--model", "dfine-n", "--data", "coco"],
        ["faster_coco_eval", "scipy"],
        [make_coco],
    ),
    (
        "train_seg_instance",
        "train_segmentation.py",
        [
            "--task",
            "segment",
            "--data",
            "seg_inst",
            "--model",
            "n",
            "--num-classes",
            "1",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch-size",
            "2",
        ],
        ["scipy", "torchmetrics", "cv2"],
        [make_seg_inst],
    ),
    (
        "train_seg_semantic",
        "train_segmentation.py",
        [
            "--task",
            "sem_seg",
            "--data",
            "seg_sem",
            "--model",
            "n",
            "--num-classes",
            "3",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch-size",
            "2",
        ],
        ["scipy"],
        [make_seg_sem],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_template_runs(case, tiny, tmp_path, monkeypatch):
    _id, script, argv, needs, assets = case
    for mod in needs:
        pytest.importorskip(mod)
    for build in assets:
        build(tmp_path)
    _run(script, argv, tmp_path, monkeypatch)  # must not raise


def test_template_export_then_deploy(tiny, tmp_path, monkeypatch):
    """The export -> onnxruntime chain: deploy consumes the graph export just wrote."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    make_image(tmp_path)
    _run("export_onnx.py", ["--model", "dfine-n", "--file", "m.onnx"], tmp_path, monkeypatch)
    assert (tmp_path / "m.onnx").exists()
    # the tiny model is built at imgsz 320, so deploy must preprocess at 320 to match.
    _run(
        "deploy_onnxruntime.py",
        ["--onnx", "m.onnx", "--image", "street.jpg", "--imgsz", "320"],
        tmp_path,
        monkeypatch,
    )


def test_config_as_yaml_roundtrip(tiny, tmp_path, monkeypatch):
    """Also exercise config_as_yaml's rebuild path (DFINE(config=...) from the file)."""
    pytest.importorskip("yaml")
    _run(
        "config_as_yaml.py",
        ["--size", "n", "--num-classes", "2", "--out", "m.yaml"],
        tmp_path,
        monkeypatch,
    )
    _run("config_as_yaml.py", ["--from-yaml", "m.yaml"], tmp_path, monkeypatch)


def test_all_templates_parse():
    """Every templates/*.py compiles — guards even scripts not in the run matrix."""
    import ast

    scripts = sorted(TEMPLATES.glob("*.py"))
    assert scripts, "no templates found"
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
