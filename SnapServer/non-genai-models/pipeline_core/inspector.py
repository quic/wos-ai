# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import onnxruntime as ort


_ORT_DTYPE_MAP: Dict[str, str] = {
    "tensor(float)":   "float32",
    "tensor(double)":  "float64",
    "tensor(float16)": "float16",
    "tensor(int32)":   "int32",
    "tensor(int64)":   "int64",
    "tensor(uint8)":   "uint8",
    "tensor(bool)":    "bool",
    "tensor(string)":  "string",
}


@dataclass
class TensorInfo:
    name: str
    shape: List[Optional[int]]
    dtype: str


@dataclass
class OnnxMetadata:
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    metadata_props: Dict[str, str] = field(default_factory=dict)


class OnnxModelInspector:
    def __init__(self, model_path: str, use_qnn: bool = False) -> None:
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        from .session import create_session
        self._session, _ = create_session(model_path, use_qnn=use_qnn)

    @property
    def inputs(self) -> List[TensorInfo]:
        return [
            TensorInfo(
                name=n.name,
                shape=list(n.shape),
                dtype=_ORT_DTYPE_MAP.get(n.type, n.type),
            )
            for n in self._session.get_inputs()
        ]

    @property
    def outputs(self) -> List[TensorInfo]:
        return [
            TensorInfo(
                name=n.name,
                shape=list(n.shape),
                dtype=_ORT_DTYPE_MAP.get(n.type, n.type),
            )
            for n in self._session.get_outputs()
        ]

    @property
    def input_layout(self) -> str:
        inputs = self.inputs
        if not inputs:
            return "NCHW"
        shape = inputs[0].shape
        if len(shape) == 4:
            # [B, C, H, W] → NCHW if dim[1] ∈ {1,3,4} and < dim[2] and < dim[3]
            try:
                b, c, h, w = shape
                if isinstance(c, int) and isinstance(h, int) and isinstance(w, int):
                    if c in (1, 3, 4) and c < h and c < w:
                        return "NCHW"
            except (ValueError, TypeError):
                pass
            # [B, H, W, C] → NHWC if dim[3] ∈ {1,3,4} and < dim[1] and < dim[2]
            try:
                b, h, w, c = shape
                if isinstance(c, int) and isinstance(h, int) and isinstance(w, int):
                    if c in (1, 3, 4) and c < h and c < w:
                        return "NHWC"
            except (ValueError, TypeError):
                pass
        return "NCHW"

    @property
    def metadata_props(self) -> Dict[str, str]:
        meta = self._session.get_modelmeta()
        props: Dict[str, str] = {}
        if meta.custom_metadata_map:
            props.update(meta.custom_metadata_map)
        props["producer_name"] = meta.producer_name or ""
        props["graph_name"]    = meta.graph_name or ""
        props["domain"]        = meta.domain or ""
        props["description"]   = meta.description or ""
        props["version"]       = str(meta.version) if meta.version else ""
        return props

    def summary(self) -> str:
        lines = ["=== OnnxModelInspector ==="]
        lines.append("Inputs:")
        for t in self.inputs:
            lines.append(f"  {t.name}: {t.dtype} {t.shape}")
        lines.append("Outputs:")
        for t in self.outputs:
            lines.append(f"  {t.name}: {t.dtype} {t.shape}")
        lines.append("Layout: " + self.input_layout)
        props = self.metadata_props
        if any(v for v in props.values()):
            lines.append("Metadata:")
            for k, v in props.items():
                if v:
                    lines.append(f"  {k}: {v}")
        return "\n".join(lines)
