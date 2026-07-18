"""Smoke tests for the package import and the `dfine` CLI."""

from __future__ import annotations

import pytest

import dfine
from dfine.cli import main


def test_package_imports_without_torch():
    assert dfine.__version__
    assert "l" in dfine.list_presets()


def test_unknown_symbol_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = dfine.NoSuchThing


def test_public_symbols_lazy_resolve():
    pytest.importorskip("torch")
    from dfine.model import DFINE as _ModelDFINE

    assert dfine.DFINE is _ModelDFINE


def test_cli_models_runs(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "Size presets:" in out
    assert "dfine-l" in out


def test_cli_predict_requires_source():
    # predict/val/train are wired now; argparse enforces their required args.
    from dfine.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "dfine-s"])  # missing source
    with pytest.raises(SystemExit):
        build_parser().parse_args(["val", "dfine-s"])  # missing --data


def test_cli_predict_runs_and_saves(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from PIL import Image

    import dfine.cli as cli
    from dfine.model import DFINE

    # Offline model (no ImageNet-backbone download), built at imgsz 320 so predict(320) fits.
    monkeypatch.setattr(
        cli,
        "_build_model",
        lambda *a, **k: DFINE(size="n", backbone_pretrained=False, num_classes=80, imgsz=320),
    )
    img = tmp_path / "img.png"
    Image.fromarray((np.random.rand(48, 64, 3) * 255).astype("uint8")).save(img)
    out = tmp_path / "out"
    rc = main(["predict", "n", str(img), "--conf", "0.99", "--imgsz", "320", "--output", str(out)])
    assert rc == 0
    assert (out / "img_pred.jpg").exists()
