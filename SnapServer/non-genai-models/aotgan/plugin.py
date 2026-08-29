# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
AOT-GAN plugin -- image inpainting.
input_type:  inpainting  -> resize 512x512 stretch, RGB /255 -> float32 [0,1] NHWC,
                            mask binarised > 0 -> float32 {0,1} NHWC [1,512,512,1]
output_type: inpainting  -> painted_image [1,512,512,3] float32 [0,1] NHWC -> PNG bytes

Source: qualcomm/ai-hub-models v0.56.0 models/aotgan/model.py + _shared/repaint/
"""
from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict

try:
    try:
        from utils.venv_plugin import VenvPlugin
    except ModuleNotFoundError:
        class VenvPlugin:   # minimal stand-in when running outside the WoS server
            pass
except ModuleNotFoundError:
    VenvPlugin = object  # type: ignore[assignment,misc]


class AOTGANPlugin(VenvPlugin):
    def load(self, model_id: str, config: Dict) -> None:
        from pipeline_core import (
            ConfigLoader, ModelConfig, OnnxModelInspector,
            Preprocessor, Postprocessor, create_session,
        )
        model_path  = config["model_path"]
        config_path = config["config_path"]
        use_qnn     = config.get("use_qnn", False)

        inspector  = OnnxModelInspector(model_path, use_qnn=use_qnn)
        loader     = ConfigLoader(config_path)
        self._cfg  = ModelConfig(inspector, loader)
        self._pre  = Preprocessor(self._cfg)
        self._post = Postprocessor(self._cfg)
        self._session, self._run_opts = create_session(model_path, use_qnn=use_qnn)

        # Read both ONNX input names directly from the graph (avoids hardcoding)
        inputs = self._session.get_inputs()
        self._image_input_name = inputs[0].name   # "image"
        self._mask_input_name  = inputs[1].name   # "mask"

    def unload(self) -> None:
        if hasattr(self, "_session"):
            del self._session

    def image_variation(self, image: bytes, n: int = 1, size: str = "1024x1024",
                        response_format: str = "b64_json", **kwargs) -> list:
        """Inpaint the masked region of image.

        Pass mask bytes via kwargs["mask"].  If omitted, _pipeline_inpainting
        generates a default centre 50%x50% rectangle mask automatically.
        """
        mask_data = kwargs.get("mask")
        image_tensor, mask_tensor = self._pre.process((image, mask_data))
        outputs = self._session.run(
            None,
            {
                self._image_input_name: image_tensor,
                self._mask_input_name:  mask_tensor,
            },
            self._run_opts,
        )
        png_bytes = self._post.process(outputs)
        if response_format == "url":
            return [{"image_bytes": png_bytes}]
        return [{"b64_json": base64.b64encode(png_bytes).decode()}]
