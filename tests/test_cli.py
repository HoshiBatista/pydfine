"""Smoke tests for the package import and the `dfine` CLI."""

from __future__ import annotations

import pytest

import dfine
from dfine.cli import main


def test_package_imports_without_torch():
    assert dfine.__version__
    assert "l" in dfine.list_presets()


def test_pending_symbol_raises_clear_error():
    with pytest.raises(AttributeError, match="not implemented yet"):
        _ = dfine.DFINE


def test_cli_models_runs(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "Size presets:" in out
    assert "dfine-l" in out


def test_cli_stub_reports_phase(capsys):
    assert main(["predict"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
