"""TorchScript export — a self-contained ``.torchscript`` deploy graph (Phase 3).

Traces the same deploy-mode ``model → postprocessor`` module used for ONNX (see
:mod:`dfine.export.onnx`) and serializes it with ``torch.jit``. The result loads with
``torch.jit.load`` and runs without Python/pydfine — handy for LibTorch / mobile / a
dependency-light server. No extra install beyond torch (unlike the ONNX toolchain).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .onnx import _build_deploy

__all__ = ["export_torchscript"]


def export_torchscript(
    model: nn.Module,
    postprocessor: nn.Module,
    file: str | Path,
    *,
    task: str = "detect",
    imgsz: int = 640,
    batch: int = 1,
    device: torch.device | str = "cpu",
) -> Path:
    """Trace ``model`` (+ ``postprocessor``) into a TorchScript ``.torchscript`` at ``file``.

    Same per-task I/O contract as :func:`dfine.export.onnx.export_onnx`: ``detect`` →
    ``(images, orig_target_sizes)`` → ``(labels, boxes, scores)``; ``segment`` adds
    ``masks``; ``sem_seg`` takes ``images`` only → a ``[N, H, W]`` uint8 label map. The
    graph is traced at a **fixed** ``batch``/``imgsz`` (TorchScript keeps the traced input
    shapes), so trace at the batch/resolution you'll serve. Both submodules are deep-copied
    to deploy mode inside the deploy wrapper, so the live model stays usable.

    Args:
        task: ``"detect"`` / ``"segment"`` / ``"sem_seg"`` — output contract.
        imgsz: square input resolution (match the model's ``imgsz``).
        batch: fixed batch size baked into the trace.
        device: device to trace on.

    Returns:
        The written ``.torchscript`` :class:`~pathlib.Path`.
    """
    device = torch.device(device)
    file = Path(file)
    deploy, input_names, _ = _build_deploy(model, postprocessor, task)
    deploy = deploy.to(device).eval()

    images = torch.rand(batch, 3, imgsz, imgsz, device=device)
    sizes = torch.tensor([[imgsz, imgsz]] * batch, device=device)
    args = (images, sizes) if "orig_target_sizes" in input_names else (images,)

    with torch.no_grad():
        scripted = torch.jit.trace(deploy, args, check_trace=False)
    file.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(file))
    return file
