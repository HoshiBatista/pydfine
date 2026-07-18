"""Tests for the public DFINE class (predict + config-first construction).

Random-init weights (no network); asserts the pipeline wiring, not accuracy.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from dfine import DFINE  # noqa: E402
from dfine.results import Results  # noqa: E402

IMGSZ = 320


def _model(**kw):
    kw.setdefault("num_classes", 80)
    return DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False, **kw)


def _image(w=640, h=480):
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"))


def test_construct_from_preset_and_overrides():
    m = _model(num_classes=3)
    assert m.config.size == "n"
    assert m.config.num_classes == 3
    assert m.model.decoder.enc_score_head.out_features == 3


def test_construct_custom_no_preset():
    # size=None -> pure config from params (still valid for N-like 2-level).
    m = DFINE(backbone_pretrained=False, imgsz=IMGSZ)
    assert m.config.size is None
    assert isinstance(m, DFINE)


def test_predict_single_returns_results():
    m = _model()
    out = m.predict(_image(), conf=0.0, imgsz=IMGSZ)
    assert isinstance(out, list) and len(out) == 1
    r = out[0]
    assert isinstance(r, Results)
    assert r.orig_shape == (480, 640)
    # conf=0 keeps every top-k query; boxes in original pixel scale.
    assert len(r) == m.config.num_top_queries
    assert r.boxes.xyxy.shape == (m.config.num_top_queries, 4)


def test_predict_imgsz_must_match_model():
    # The encoder's positional embeddings are precomputed for cfg.imgsz, so predicting at
    # a different resolution must raise clearly rather than crash deep in the encoder.
    m = _model()
    with pytest.raises(ValueError, match="must equal the model's imgsz"):
        m.predict(_image(), imgsz=IMGSZ // 2)


def test_predict_batch_and_call_alias():
    m = _model()
    imgs = [_image(640, 480), _image(320, 320)]
    out = m(imgs, conf=0.0, imgsz=IMGSZ)  # __call__ == predict
    assert len(out) == 2
    assert out[0].orig_shape == (480, 640)
    assert out[1].orig_shape == (320, 320)


def test_conf_filter_reduces_detections():
    m = _model()
    img = _image()
    all_dets = m.predict(img, conf=0.0, imgsz=IMGSZ)[0]
    high = m.predict(img, conf=0.99, imgsz=IMGSZ)[0]
    assert len(high) <= len(all_dets)


def test_predict_accepts_ndarray():
    m = _model()
    arr = (np.random.rand(200, 300, 3) * 255).astype("uint8")
    r = m.predict(arr, conf=0.0, imgsz=IMGSZ)[0]
    assert r.orig_shape == (200, 300)


def test_names_default_to_coco_for_80_classes():
    m = _model()
    assert m.names[0] == "person"
    assert len(m.names) == 80


def test_names_from_class_names():
    m = _model(num_classes=3, class_names=["cat", "dog", "bird"])
    assert m.names == {0: "cat", 1: "dog", 2: "bird"}


def test_load_rejects_unknown_source():
    m = _model()
    with pytest.raises(FileNotFoundError):
        m.load("/no/such/file.pth")


def test_export_imgsz_must_match_model():
    m = _model()
    # predict/train/val/export are all implemented now; export guards a mismatched
    # imgsz (the encoder's positional embeddings are sized to cfg.imgsz).
    with pytest.raises(ValueError, match="must match"):
        m.export(imgsz=m.config.imgsz + 32)


def test_val_requires_data_or_loader():
    m = _model()
    with pytest.raises(ValueError, match="data=|val_loader="):
        m.val()


def test_val_rejects_both_data_and_loader():
    m = _model()
    with pytest.raises(ValueError, match="not both"):
        m.val(data="somewhere", val_loader=object())


def test_val_from_data_path(tmp_path):
    pytest.importorskip("faster_coco_eval")
    from dfine.train.evaluator import COCO_STAT_NAMES
    from tests.test_dataset import _write_split

    _write_split(
        tmp_path / "val2017",
        tmp_path / "annotations" / "instances_val2017.json",
        ((200, 150),),
    )
    m = DFINE(size="n", imgsz=IMGSZ, backbone_pretrained=False)
    metrics = m.val(data=str(tmp_path), batch_size=1, num_workers=0)
    assert set(metrics) == set(COCO_STAT_NAMES)
    assert all(isinstance(v, float) for v in metrics.values())


def test_train_requires_data_or_loader():
    m = _model()
    with pytest.raises(ValueError, match="data=|train_loader="):
        m.train()


def test_train_rejects_both_data_and_loader():
    m = _model()
    with pytest.raises(ValueError, match="not both"):
        m.train(train_loader=object(), data="somewhere")


def test_train_from_data_path(tmp_path):
    pytest.importorskip("faster_coco_eval")
    pytest.importorskip("scipy")
    from tests.test_dataset import _make_coco_root

    # with_val=True exercises the auto COCO val_fn wired inside train().
    root = _make_coco_root(tmp_path, with_val=True)
    m = DFINE(
        size="n",
        imgsz=IMGSZ,
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=0,
    )
    out = m.train(
        data=root,
        epochs=1,
        batch_size=2,
        num_workers=0,
        remap_mscoco_category=True,
        output_dir=str(tmp_path / "runs"),
        visualize=False,
    )
    assert out is m
    assert (tmp_path / "runs" / "last.pth").exists()
