"""Export backends (Phase 3). ONNX today; TensorRT/OpenVINO via downstream helpers.

Everything here imports torch; the ONNX toolchain (``onnx``/``onnxruntime``/``onnxsim``)
is imported lazily and comes from ``pip install dfine[export]``. The public entry point
is :meth:`dfine.DFINE.export`; the pieces are importable directly for custom pipelines.
"""

from __future__ import annotations

from .onnx import DeployModel, export_onnx, tensorrt_command

__all__ = ["export_onnx", "DeployModel", "tensorrt_command"]
