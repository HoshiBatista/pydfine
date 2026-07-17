"""ONNX export for D-FINE — ports ``tools/deployment/export_onnx.py``.

Wraps the deploy-mode model + postprocessor into a single graph with the upstream
two-input signature ``(images, orig_target_sizes)`` and three outputs
``(labels, boxes, scores)``, so the exported graph matches the torch path and the same
ONNX runs on onnxruntime / TensorRT (`trtexec --fp16`) / OpenVINO downstream.

The batch dimension is dynamic by default (``N``), so one export serves any batch size.
``onnx`` + ``onnxruntime`` (and optional ``onnxsim``) come from ``pip install
dfine[export]`` and are imported lazily — building a model never requires them.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["DeployModel", "export_onnx", "tensorrt_command"]


class DeployModel(nn.Module):
    """Deploy-mode ``model → postprocessor`` as one module (single ONNX graph).

    ``forward(images, orig_target_sizes)`` returns ``(labels, boxes, scores)`` — boxes
    are ``xyxy`` in the original image scale given by ``orig_target_sizes`` (``[W, H]``
    per image). Both submodules are deep-copied and switched to deploy mode, so the
    caller's model stays trainable/usable after export.
    """

    def __init__(self, model: nn.Module, postprocessor: nn.Module):
        super().__init__()
        self.model = deepcopy(model).deploy()
        self.postprocessor = deepcopy(postprocessor).deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        return self.postprocessor(self.model(images), orig_target_sizes)


def export_onnx(
    model: nn.Module,
    postprocessor: nn.Module,
    file: str | Path,
    *,
    imgsz: int = 640,
    batch: int = 1,
    opset: int = 16,
    dynamic: bool = True,
    simplify: bool = False,
    check: bool = True,
    device: torch.device | str = "cpu",
) -> Path:
    """Export ``model`` + ``postprocessor`` to an ONNX graph at ``file``.

    Args:
        imgsz: square input resolution of the dummy input (match the model's ``imgsz``).
        batch: dummy batch size (a real value even when ``dynamic`` — used for tracing).
        opset: ONNX opset (upstream uses 16).
        dynamic: mark the batch dim ``N`` dynamic on inputs and outputs.
        simplify: run ``onnxsim`` on the graph (needs ``onnxsim``).
        check: run ``onnx.checker`` on the exported graph.
        device: device to trace on.

    Returns:
        The written ``.onnx`` :class:`~pathlib.Path`.
    """
    device = torch.device(device)
    file = Path(file)
    deploy = DeployModel(model, postprocessor).to(device).eval()

    # The decoder takes a batched code path only when batch > 1; trace with >= 2 for a
    # dynamic export so the graph generalizes to any N (it still serves batch 1). Upstream
    # traces with 32 for the same reason.
    trace_batch = max(batch, 2) if dynamic else batch
    images = torch.rand(trace_batch, 3, imgsz, imgsz, device=device)
    orig_target_sizes = torch.tensor([[imgsz, imgsz]] * trace_batch, device=device)
    deploy(images, orig_target_sizes)  # sanity forward before export

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {0: "N"} for name in ("images", "orig_target_sizes")}
        dynamic_axes.update({name: {0: "N"} for name in ("labels", "boxes", "scores")})

    # Force the legacy TorchScript exporter (what upstream validated at opset 16); newer
    # torch defaults to the dynamo exporter, which needs the extra ``onnxscript`` dep.
    export_kwargs = dict(
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    import inspect

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(deploy, (images, orig_target_sizes), str(file), **export_kwargs)

    if check:
        import onnx

        onnx.checker.check_model(onnx.load(str(file)))

    if simplify:
        import onnx
        import onnxsim

        shapes = {
            "images": tuple(images.shape),
            "orig_target_sizes": tuple(orig_target_sizes.shape),
        }
        simplified, ok = onnxsim.simplify(str(file), test_input_shapes=shapes)
        if not ok:
            raise RuntimeError(f"onnxsim failed to validate the simplified graph for {file}.")
        onnx.save(simplified, str(file))

    return file


def tensorrt_command(onnx_file: str | Path, *, fp16: bool = True, engine: str | None = None) -> str:
    """Return the ``trtexec`` command to build a TensorRT engine from ``onnx_file``.

    The graph's batch dim is dynamic, so TensorRT needs an optimization profile; this
    provides sensible min/opt/max shapes. Run the returned command where ``trtexec`` (and
    OpenVINO's ``ovc <onnx_file>`` for OpenVINO) is installed — those toolchains are not
    Python deps here.
    """
    onnx_file = Path(onnx_file)
    engine = engine or str(onnx_file.with_suffix(".engine"))
    parts = [
        "trtexec",
        f"--onnx={onnx_file}",
        f"--saveEngine={engine}",
        "--minShapes=images:1x3x640x640,orig_target_sizes:1x2",
        "--optShapes=images:1x3x640x640,orig_target_sizes:1x2",
        "--maxShapes=images:32x3x640x640,orig_target_sizes:32x2",
    ]
    if fp16:
        parts.append("--fp16")
    return " ".join(parts)
