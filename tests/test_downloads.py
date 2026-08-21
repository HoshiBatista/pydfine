"""Tests for atomic, process-safe checkpoint downloads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

torch = pytest.importorskip("torch")

from dfine.downloads import download  # noqa: E402


def test_failed_download_removes_temporary_file(monkeypatch, tmp_path):
    def fail(_url, path, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("network failed")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fail)
    with pytest.raises(OSError, match="network failed"):
        download("https://example.test/model.pth", cache_dir_override=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_concurrent_downloads_use_distinct_temporary_files(monkeypatch, tmp_path):
    barrier = Barrier(2)
    temporary_paths = []

    def fake_download(_url, path, **_kwargs):
        temporary_paths.append(path)
        barrier.wait()
        Path(path).write_bytes(b"complete checkpoint")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    url = "https://example.test/model.pth"
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: download(url, cache_dir_override=tmp_path), range(2)))

    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == b"complete checkpoint"
    assert len(set(temporary_paths)) == 2
    assert not list(tmp_path.glob("*.part"))
