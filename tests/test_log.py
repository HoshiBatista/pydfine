"""The zero-dependency colored console logger (torch-free)."""

from __future__ import annotations

import logging

from dfine import log
from dfine.log import LOGGER, banner, colorstr, fmt_num, metrics_line, rule


def _with_color(monkeypatch, on: bool) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1") if on else monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("NO_COLOR" if on else "FORCE_COLOR", raising=False)
    log._color_enabled.cache_clear()


def test_colorstr_plain_when_color_disabled(monkeypatch):
    _with_color(monkeypatch, False)
    assert colorstr("green", "bold", "hello") == "hello"
    assert "\033[" not in banner("title", {"k": "v"})


def test_colorstr_wraps_ansi_when_enabled(monkeypatch):
    _with_color(monkeypatch, True)
    out = colorstr("green", "bold", "hello")
    assert out.startswith("\033[") and out.endswith("\033[0m") and "hello" in out
    assert colorstr("just text").endswith("\033[0m")  # default blue+bold


def test_fmt_num_rounds_floats():
    assert fmt_num(0.412345) == "0.4123"
    assert fmt_num(72) == "72"
    assert fmt_num("on") == "on"


def test_banner_and_metrics_line_content(monkeypatch):
    _with_color(monkeypatch, False)  # compare on plain text
    b = banner("D-FINE n", {"epochs": 72, "device": "cpu"})
    assert "D-FINE n" in b and "epochs" in b and "72" in b and "device" in b and "cpu" in b
    line = metrics_line({"AP": 0.412, "AP50": 0.583})
    assert "AP" in line and "0.412" in line and "AP50" in line and "0.583" in line
    assert "eval" in rule("eval")


def test_logger_emits_message(monkeypatch, caplog):
    _with_color(monkeypatch, False)
    with caplog.at_level(logging.INFO, logger="dfine"):
        LOGGER.info("training started")
    assert "training started" in caplog.text
