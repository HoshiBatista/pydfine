"""Download + cache released checkpoints.

Fetches a checkpoint URL into a local cache (``~/.cache/dfine`` by default,
overridable via ``$DFINE_CACHE_DIR`` or the ``cache_dir`` arg) and returns the
local path. A file already present is reused — downloads are content-addressed by
filename, and released assets are immutable.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["cache_dir", "download", "download_weights"]


def cache_dir(override: str | os.PathLike | None = None) -> Path:
    """Resolve the weights cache directory (creating it if needed)."""
    root = override or os.environ.get("DFINE_CACHE_DIR") or (Path.home() / ".cache" / "dfine")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, filename: str | None = None, cache_dir_override=None, progress: bool = True):
    """Download ``url`` into the cache and return the local :class:`~pathlib.Path`.

    Skips the network if the target file already exists. ``filename`` defaults to
    the basename of the URL.
    """
    name = filename or url.rsplit("/", 1)[-1]
    dst = cache_dir(cache_dir_override) / name
    if dst.exists():
        return dst

    import torch.hub

    # Download to a temp name, then atomically rename so a killed download never
    # leaves a truncated file that later looks "cached".
    tmp = dst.with_suffix(dst.suffix + ".part")
    torch.hub.download_url_to_file(url, str(tmp), progress=progress)
    tmp.replace(dst)
    return dst


def download_weights(spec, cache_dir_override=None, progress: bool = True):
    """Download the checkpoint for a :class:`~dfine.registry.CheckpointSpec` (or name)."""
    from .registry import resolve

    if isinstance(spec, str):
        spec = resolve(spec)
    return download(spec.url, spec.filename, cache_dir_override, progress)
