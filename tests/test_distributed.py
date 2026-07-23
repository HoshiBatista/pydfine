"""Multi-GPU / DDP helper tests (Phase 4).

The launch machinery in ``dfine.train.distributed`` is exercised without real GPUs:
the no-group defaults, and a genuine 1-process ``gloo`` group (CPU) for the DDP /
DistributedSampler wrapping. A full 2-process spawn end-to-end is gated behind
``DFINE_TEST_MULTIGPU=1`` (CPU + gloo) since it is slow and needs >=2 cores.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from dfine.train import distributed as D  # noqa: E402


def test_defaults_without_process_group():
    assert not D.is_dist_available_and_initialized()
    assert D.get_rank() == 0
    assert D.get_world_size() == 1
    assert D.is_main_process()
    D.barrier()  # no-op, must not raise


def test_de_parallel_passthrough_when_unwrapped():
    m = nn.Linear(2, 2)
    assert D.de_parallel(m) is m


def test_wrap_helpers_are_noops_without_group():
    m = nn.Linear(2, 2)
    assert D.wrap_model_ddp(m) is m
    loader = DataLoader(TensorDataset(torch.zeros(4, 2)), batch_size=2)
    assert D.wrap_loader_distributed(loader, shuffle=True) is loader


def test_launched_via_torchrun_reads_env(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert not D.launched_via_torchrun()
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    assert D.launched_via_torchrun()
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not D.launched_via_torchrun()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gloo_group():
    """A real single-process gloo group so the DDP/sampler paths run on CPU."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_get_rank_world_size_in_group(gloo_group):
    assert D.is_dist_available_and_initialized()
    assert D.get_rank() == 0
    assert D.get_world_size() == 1


def test_wrap_model_ddp_and_de_parallel(gloo_group):
    m = nn.Linear(3, 3)
    wrapped = D.wrap_model_ddp(m, sync_bn=True)
    assert isinstance(wrapped, DDP)
    assert D.de_parallel(wrapped) is m


def test_wrap_loader_distributed_shards_and_set_epoch(gloo_group):
    ds = TensorDataset(torch.arange(8).float().reshape(8, 1))
    loader = DataLoader(ds, batch_size=2, drop_last=False)
    dloader = D.wrap_loader_distributed(loader, shuffle=False)

    from torch.utils.data import DistributedSampler

    assert isinstance(dloader.sampler, DistributedSampler)
    # world_size == 1 -> the single rank still sees every sample.
    seen = torch.cat([b[0] for b in dloader]).flatten()
    assert sorted(seen.tolist()) == list(range(8))
    dloader.set_epoch(3)  # forwarded to the sampler, must not raise


@pytest.mark.skipif(
    os.environ.get("DFINE_TEST_MULTIGPU") != "1",
    reason="set DFINE_TEST_MULTIGPU=1 to run the 2-process CPU/gloo spawn end-to-end",
)
def test_multigpu_spawn_end_to_end(tmp_path):
    pytest.importorskip("faster_coco_eval")
    pytest.importorskip("scipy")
    from dfine import DFINE
    from tests.test_dataset import _write_split

    # Enough images that each of the 2 ranks still gets a full batch after sharding.
    _write_split(
        tmp_path / "train",
        tmp_path / "annotations" / "instances_train.json",
        ((200, 150), (120, 90), (160, 128), (96, 96)),
    )
    root = str(tmp_path)
    m = DFINE(
        size="n",
        imgsz=320,
        backbone_pretrained=False,
        freeze_norm=False,
        freeze_at=-1,
        num_denoising=0,
    )
    out = m.train(
        data=root,
        devices=2,
        epochs=1,
        batch_size=1,
        num_workers=0,
        remap_mscoco_category=True,
        output_dir=str(tmp_path / "runs"),
        visualize=False,
    )
    assert out is m
    assert (tmp_path / "runs" / "last.pth").exists()


def test_multigpu_requires_data_path():
    from dfine import DFINE

    m = DFINE(size="n", imgsz=320, backbone_pretrained=False)
    with pytest.raises(ValueError, match="needs `data=`"):
        m.train(train_loader=[], devices=2)
