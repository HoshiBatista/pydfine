"""ONNX export for D-FINE — ports ``tools/deployment/export_onnx.py``.

Wraps the deploy-mode model (+ postprocessor) into a single graph. The output contract
depends on the model's ``task``:

- ``detect``  — inputs ``(images, orig_target_sizes)`` → ``(labels, boxes, scores)``.
- ``segment`` — same inputs → ``(labels, boxes, scores, masks)``, where ``masks`` are the
  top-k queries' **sigmoid** mask probabilities at the decoder's 1/4 resolution
  ``[N, K, H/4, W/4]``; threshold, resize to the original size, and clip to each box on
  the host (exactly what :meth:`dfine.DFINE.predict` does).
- ``sem_seg`` — input ``images`` only → ``sem_seg`` ``[N, H, W]`` uint8 label map, the
  per-pixel **argmax fused into the graph** at the network resolution; resize (nearest) to
  each original image size on the host, like :class:`SemSegPostProcessor`.

So the exported graph matches the torch path and the same ONNX runs on onnxruntime /
TensorRT (`trtexec --fp16`) / OpenVINO downstream. The batch dimension is dynamic by
default (``N``), so one export serves any batch size. ``onnx`` + ``onnxruntime`` (and
optional ``onnxsim``) come from ``pip install pydfine[export]`` and are imported lazily —
building a model never requires them.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

__all__ = [
    "DeployModel",
    "SegInstanceDeployModel",
    "SemSegDeployModel",
    "export_onnx",
    "tensorrt_command",
]


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


class SegInstanceDeployModel(DeployModel):
    """Instance-seg deploy graph: adds ``masks`` to the detection outputs.

    ``forward(images, orig_target_sizes)`` → ``(labels, boxes, scores, masks)``. The
    postprocessor's deploy path gathers the top-k queries' mask maps, so ``masks`` is
    ``[N, K, H/4, W/4]`` sigmoid probabilities; finish (threshold + resize + box-clip) on
    the host as in :meth:`dfine.DFINE.predict`. Identical wiring to :class:`DeployModel`;
    the extra output falls out of the segmentation model emitting ``pred_masks``.
    """


class SemSegDeployModel(nn.Module):
    """Semantic-seg deploy graph: ``images`` → ``sem_seg`` ``[N, H, W]`` uint8 label map.

    The per-pixel argmax is fused into the graph; the label map is at the network input
    resolution — resize it (nearest) to each original image size on the host, like
    :class:`SemSegPostProcessor`. No ``orig_target_sizes`` input (host-side resize).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = deepcopy(model).deploy()

    def forward(self, images: torch.Tensor):
        logits = self.model(images)["sem_seg_logits"]
        return logits.argmax(1).to(torch.uint8)


def _build_deploy(model, postprocessor, task):
    """Return ``(deploy_module, input_names, output_names)`` for the model's ``task``."""
    if task == "sem_seg":
        return SemSegDeployModel(model), ["images"], ["sem_seg"]
    if task == "segment":
        names = ["images", "orig_target_sizes"]
        return (
            SegInstanceDeployModel(model, postprocessor),
            names,
            [
                "labels",
                "boxes",
                "scores",
                "masks",
            ],
        )
    names = ["images", "orig_target_sizes"]
    return DeployModel(model, postprocessor), names, ["labels", "boxes", "scores"]


def export_onnx(
    model: nn.Module,
    postprocessor: nn.Module,
    file: str | Path,
    *,
    task: str = "detect",
    imgsz: int = 640,
    batch: int = 1,
    opset: int = 16,
    dynamic: bool = True,
    simplify: bool = False,
    check: bool = True,
    device: torch.device | str = "cpu",
) -> Path:
    """Export ``model`` (+ ``postprocessor``) to an ONNX graph at ``file``.

    Args:
        task: ``"detect"`` / ``"segment"`` / ``"sem_seg"`` — picks the output contract
            (see the module docstring). ``postprocessor`` is unused for ``"sem_seg"``.
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
    deploy, input_names, output_names = _build_deploy(model, postprocessor, task)
    deploy = deploy.to(device).eval()

    trace_batch = max(batch, 2) if dynamic else batch
    images = torch.rand(trace_batch, 3, imgsz, imgsz, device=device)
    sizes = torch.tensor([[imgsz, imgsz]] * trace_batch, device=device)
    args = (images, sizes) if "orig_target_sizes" in input_names else (images,)
    deploy(*args)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {0: "N"} for name in input_names + output_names}

    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    import inspect

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(deploy, args, str(file), **export_kwargs)

    if check:
        import onnx

        onnx.checker.check_model(onnx.load(str(file)))

    if simplify:
        import onnx
        import onnxsim

        shapes = {"images": tuple(images.shape)}
        if "orig_target_sizes" in input_names:
            shapes["orig_target_sizes"] = tuple(sizes.shape)
        simplified, ok = onnxsim.simplify(str(file), test_input_shapes=shapes)
        if not ok:
            raise RuntimeError(f"onnxsim failed to validate the simplified graph for {file}.")
        onnx.save(simplified, str(file))

    return file


def tensorrt_command(
    onnx_file: str | Path,
    *,
    task: str = "detect",
    imgsz: int = 640,
    fp16: bool = True,
    engine: str | None = None,
    max_batch: int = 32,
) -> str:
    """Return the ``trtexec`` command to build a TensorRT engine from ``onnx_file``.

    The graph's batch dim is dynamic (H/W are fixed to the export resolution), so
    TensorRT needs an optimization profile; this provides min/opt/max shapes at
    ``imgsz`` (pass the same value you exported with) with batch 1..``max_batch``. The
    ``sem_seg`` graph has a single ``images`` input; ``detect``/``segment`` also take
    ``orig_target_sizes`` (pass the matching ``task``). Run the returned command where
    ``trtexec`` (and OpenVINO's ``ovc <onnx_file>`` for OpenVINO) is installed — those
    toolchains are not Python deps here.
    """
    onnx_file = Path(onnx_file)
    engine = engine or str(onnx_file.with_suffix(".engine"))
    hw = f"3x{imgsz}x{imgsz}"
    sizes = "" if task == "sem_seg" else ",orig_target_sizes:{b}x2"
    parts = [
        "trtexec",
        f"--onnx={onnx_file}",
        f"--saveEngine={engine}",
        f"--minShapes=images:1x{hw}{sizes.format(b=1)}",
        f"--optShapes=images:1x{hw}{sizes.format(b=1)}",
        f"--maxShapes=images:{max_batch}x{hw}{sizes.format(b=max_batch)}",
    ]
    if fp16:
        parts.append("--fp16")
    return " ".join(parts)
