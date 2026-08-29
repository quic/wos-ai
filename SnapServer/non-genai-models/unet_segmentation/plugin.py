# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Unet-Segmentation plugin — binary mask segmentation.
output_type: segmentation
"""
from __future__ import annotations

import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict

try:
    from utils.venv_plugin import VenvPlugin
except ModuleNotFoundError:
    class VenvPlugin:   # minimal stand-in when running outside the WoS server
        pass


class UnetSegmentationPlugin(VenvPlugin):
    def load(self, model_id: str, config: Dict) -> None:
        from pipeline_core import ConfigLoader, ModelConfig, OnnxModelInspector, Preprocessor, Postprocessor
        from pipeline_core import create_session

        model_path  = config["model_path"]
        config_path = config["config_path"]

        inspector  = OnnxModelInspector(model_path, use_qnn=config.get("use_qnn", False))
        loader     = ConfigLoader(config_path)
        self._cfg  = ModelConfig(inspector, loader)
        self._pre  = Preprocessor(self._cfg)
        self._post = Postprocessor(self._cfg)
        self._session, self._run_opts = create_session(
            model_path, use_qnn=config.get("use_qnn", False)
        )

    def unload(self) -> None:
        if hasattr(self, "_session"):
            del self._session

    def image_variation(self, image: bytes, n: int = 1, size: str = "1024x1024",
                        response_format: str = "b64_json", **kwargs) -> list:
        return self._run(image, response_format)

    def _run(self, image: bytes, response_format: str = "b64_json") -> list:
        from PIL import Image as _Image

        tensor  = self._pre.process(image)
        outputs = self._session.run(None, {self._cfg.input_name: tensor}, self._run_opts)
        mask    = self._post.process(outputs)         # [1, H, W] uint8 (0/1)
        mask2d  = (mask[0] * 255).astype("uint8")    # [H, W] 0-or-255 grayscale
        buf = io.BytesIO()
        _Image.fromarray(mask2d, mode="L").save(buf, format="PNG")
        png_bytes = buf.getvalue()
        if response_format == "url":
            return [{"image_bytes": png_bytes}]
        return [{"b64_json": base64.b64encode(png_bytes).decode()}]
