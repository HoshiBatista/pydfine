"""Multi-GPU launch + DDP helpers for training (Phase 4).

Ports the pieces of upstream ``D-FINE/src/misc/dist_utils.py`` the single-node training
path needs, torchrun-free: rank/world-size queries, process-group setup/teardown, model
(DDP + optional SyncBN) and dataloader (``DistributedSampler``) wrapping, and a
``spawn`` that launches N worker processes — the "wrap torchrun" entry point behind
``DFINE.train(devices=N)``.

Two ways in, both funnel through the same DDP-aware ``Trainer``:

* ``DFINE.train(data=..., devices=N)`` — this process is the launcher; it ``spawn``s N
  workers (one per GPU) that each rebuild the model/loaders and train under DDP.
* ``torchrun --nproc_per_node=N your_script.py`` where the script calls
  ``DFINE.train(...)`` — each torchrun worker has ``WORLD_SIZE>1`` in its env, so
  ``train`` detects it and joins the existing group instead of spawning again.

Everything degrades cleanly to ``world_size == 1`` (rank 0) when no group is running,
and works on CPU with the ``gloo`` backend (used by the tests).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

__all__ = [
    "is_dist_available_and_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "launched_via_torchrun",
    "setup_distributed",
    "cleanup_distributed",
    "de_parallel",
    "wrap_model_ddp",
    "wrap_loader_distributed",
    "spawn",
]


def is_dist_available_and_initialized() -> bool:
    """True once a process group is running (else everything acts single-process)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of this process (0 when not distributed)."""
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    """Number of processes in the group (1 when not distributed)."""
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process() -> bool:
    """True on the rank-0 process (the only one that saves/logs)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks (no-op when not distributed)."""
    if is_dist_available_and_initialized():
        dist.barrier()


def launched_via_torchrun() -> bool:
    """True when torchrun/elastic set a multi-process env we should join (not spawn)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 and "RANK" in os.environ


def _default_backend() -> str:
    """``nccl`` for CUDA, ``gloo`` otherwise (CPU / tests)."""
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup_distributed(backend: str | None = None) -> bool:
    """Initialize the process group from env vars (``RANK``/``WORLD_SIZE``/…).

    Returns True if a group is now running. Idempotent and a no-op for a 1-process
    world. Binds the current CUDA device to ``LOCAL_RANK`` when on GPU.
    """
    if is_dist_available_and_initialized():
        return True
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False

    dist.init_process_group(backend=backend or _default_backend(), init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.barrier()
    return True


def cleanup_distributed() -> None:
    """Tear down the process group (safe to call when none is running)."""
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def de_parallel(model: nn.Module) -> nn.Module:
    """Unwrap a ``DDP``/``DataParallel`` model back to the underlying module."""
    return model.module if isinstance(model, (DP, DDP)) else model


def wrap_model_ddp(
    model: nn.Module,
    *,
    device: torch.device | None = None,
    sync_bn: bool = False,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap ``model`` in ``DistributedDataParallel`` (+ optional SyncBN).

    Returns the model unchanged when no process group is running, so callers can wrap
    unconditionally. On CUDA the wrapper is pinned to the local-rank device.
    """
    if not is_dist_available_and_initialized():
        return model
    if sync_bn and device is not None and device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device is not None and device.type == "cuda":
        return DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused_parameters,
        )
    return DDP(model, find_unused_parameters=find_unused_parameters)


def wrap_loader_distributed(loader: DataLoader, *, shuffle: bool) -> DataLoader:
    """Rebuild ``loader`` with a ``DistributedSampler`` so each rank sees a shard.

    Preserves batch size / collate / workers / drop-last and forwards ``set_epoch`` to
    the sampler (and, if present, to the original loader's dataset/collate). Returns the
    loader unchanged when not distributed.
    """
    if not is_dist_available_and_initialized():
        return loader

    sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
    new = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )

    inner = loader

    def set_epoch(epoch: int) -> None:
        sampler.set_epoch(epoch)
        if hasattr(inner, "set_epoch"):
            inner.set_epoch(epoch)

    new.set_epoch = set_epoch  # type: ignore[attr-defined]
    return new


def spawn(worker: Callable, world_size: int, args: tuple = ()) -> None:
    """Launch ``world_size`` worker processes (one per GPU) — the torchrun wrapper.

    Sets ``MASTER_ADDR``/``MASTER_PORT`` and hands each worker ``(rank, world_size,
    *args)``; the worker is responsible for calling :func:`setup_distributed`. Uses
    ``torch.multiprocessing.spawn`` so no external launcher/script is needed.
    """
    import torch.multiprocessing as mp

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    mp.spawn(worker, args=(world_size, *args), nprocs=world_size, join=True)


def _free_port() -> int:
    """Pick a currently-free localhost TCP port for the rendezvous."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
