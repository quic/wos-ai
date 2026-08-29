# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
YamNet plugin — audio event classification.
output_type: softmax_top_k   (521-class AudioSet labels)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict

try:
    from utils.venv_plugin import VenvPlugin
except ModuleNotFoundError:
    class VenvPlugin:   # minimal stand-in when running outside the WoS server
        pass


class YamNetPlugin(VenvPlugin):
    def load(self, model_id: str, config: Dict) -> None:
        from pipeline_core import ConfigLoader, ModelConfig, OnnxModelInspector, Postprocessor, Preprocessor
        from pipeline_core import create_session

        model_path  = config["model_path"]
        config_path = config["config_path"]

        inspector    = OnnxModelInspector(model_path, use_qnn=config.get("use_qnn", False))
        loader       = ConfigLoader(config_path)
        self._cfg    = ModelConfig(inspector, loader)
        self._pre    = Preprocessor(self._cfg)
        self._post   = Postprocessor(self._cfg)
        self._session, self._run_opts = create_session(
            model_path, use_qnn=config.get("use_qnn", False)
        )

    def unload(self) -> None:
        if hasattr(self, "_session"):
            del self._session

    def transcribe(self, data: bytes, **kwargs) -> Dict:
        tensor  = self._pre.process(data)
        outputs = self._session.run(None, {self._cfg.input_name: tensor}, self._run_opts)
        results = self._post.process(outputs)
        top = results[0] if results else {"label": "unknown", "score": 0.0}
        return {"text": top["label"], "classifications": results}
