# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Inception-v3 plugin — image classification (1000-class ImageNet).
output_type: classification
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict

try:
    from utils.venv_plugin import VenvPlugin
except ModuleNotFoundError:
    class VenvPlugin:   # minimal stand-in when running outside the WoS server
        pass


class InceptionV3Plugin(VenvPlugin):
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
                        response_format: str = "url", **kwargs) -> list:
        tensor  = self._pre.process(image)
        outputs = self._session.run(None, {self._cfg.input_name: tensor}, self._run_opts)
        result  = self._post.process(outputs)
        return [{"b64_json": json.dumps({
            "labels": result.labels,
            "scores": [float(s) for s in result.scores],
        })}]
