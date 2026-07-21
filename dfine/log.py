"""Colorful, zero-dependency console logging for training and evaluation.

Provides an ultralytics-style :func:`colorstr` for ANSI styling and a configured
stdlib :data:`LOGGER`, plus small helpers (:func:`banner`, :func:`rule`,
:func:`metrics_line`) that the trainer and evaluator use to print readable, colored
progress. Colors auto-disable when stdout is not a TTY, when ``NO_COLOR`` is set, or
on a ``dumb`` terminal, so piped logs and files stay plain.

Deliberately import-light — no torch — so ``import dfine`` (and this module) stays
usable without the inference/training extras.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

__all__ = ["LOGGER", "colorstr", "banner", "rule", "metrics_line", "fmt_num"]

_CODES = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gray": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
    "end": "\033[0m",
}


@lru_cache(maxsize=1)
def _color_enabled() -> bool:
    """Whether ANSI codes should be emitted, honoring ``NO_COLOR``/``FORCE_COLOR``."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def colorstr(*args: object) -> str:
    """Wrap the last argument in the ANSI styles named by the preceding ones.

    ``colorstr("text")`` defaults to bold blue; ``colorstr("green", "bold", x)`` applies
    those styles. Returns the plain string unchanged when color is disabled.
    """
    if len(args) == 1:
        styles: tuple[object, ...] = ("blue", "bold")
        text = args[0]
    else:
        *styles, text = args
    text = str(text)
    if not _color_enabled():
        return text
    prefix = "".join(_CODES[s] for s in styles if isinstance(s, str) and s in _CODES)
    return f"{prefix}{text}{_CODES['end']}" if prefix else text


def fmt_num(v: object) -> str:
    """Format a metric value compactly: floats to 4 significant figures."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def banner(title: str, info: dict[str, object]) -> str:
    """A titled key/value block: bold-cyan title, right-aligned dim keys, bright values."""
    lines = [colorstr("cyan", "bold", title)]
    width = max((len(k) for k in info), default=0)
    for key, value in info.items():
        lines.append(f"  {colorstr('gray', key.rjust(width))}  {colorstr('white', value)}")
    return "\n".join(lines)


def rule(label: str, color: str = "green") -> str:
    """A section header like ``── train ──`` in the given color."""
    return colorstr(color, "bold", f"── {label} ──")


def metrics_line(
    metrics: dict[str, object],
    key_color: str = "gray",
    val_color: str = "bright_cyan",
) -> str:
    """Render ``{name: value}`` as a spaced, colored ``name value`` sequence."""
    return "  ".join(
        f"{colorstr(key_color, k)} {colorstr(val_color, 'bold', fmt_num(v))}"
        for k, v in metrics.items()
    )


class _ColorFormatter(logging.Formatter):
    """Prefix warnings/errors with a colored level tag; leave info messages clean."""

    _TAGS = {
        logging.WARNING: ("yellow", "bold", "WARNING"),
        logging.ERROR: ("red", "bold", "ERROR"),
        logging.CRITICAL: ("red", "bold", "underline", "CRITICAL"),
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        tag = self._TAGS.get(record.levelno)
        if tag is None:
            return msg
        return f"{colorstr(*tag)} {msg}"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("dfine")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = _build_logger()
