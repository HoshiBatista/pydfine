"""Export backends (Phase 3). ONNX + TorchScript; TensorRT/OpenVINO via downstream helpers.

Everything here imports torch; the ONNX toolchain (``onnx``/``onnxruntime``/``onnxsim``)
is imported lazily and comes from ``pip install pydfine[export]``, while TorchScript needs
only torch. The public entry point is :meth:`dfine.DFINE.export`; the pieces are importable
directly for custom pipelines.
"""

from __future__ import annotations

from .onnx import DeployModel, export_onnx, tensorrt_command
from .torchscript import export_torchscript

__all__ = ["export_onnx", "export_torchscript", "DeployModel", "tensorrt_command"]
