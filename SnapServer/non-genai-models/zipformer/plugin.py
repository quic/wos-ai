# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Zipformer plugin — streaming RNN-T ASR (English / Chinese).
output_type: transducer_greedy  (encoder → decoder → joiner greedy loop)
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


class ZipformerPlugin(VenvPlugin):
    def load(self, model_id: str, config: Dict) -> None:
        from pipeline_core import (
            ConfigLoader, ModelConfig, OnnxModelInspector,
            Preprocessor, Postprocessor, create_session,
            zipformer_transducer_decode,
        )

        encoder_path = config["encoder_path"]
        decoder_path = config["decoder_path"]
        joiner_path  = config["joiner_path"]
        config_path  = config["config_path"]
        use_qnn      = config.get("use_qnn", False)

        inspector = OnnxModelInspector(encoder_path, use_qnn=use_qnn)
        loader    = ConfigLoader(config_path)
        self._cfg = ModelConfig(inspector, loader)

        self._pre  = Preprocessor(self._cfg)
        self._post = Postprocessor(self._cfg)

        self._enc, self._enc_opts = create_session(encoder_path, use_qnn=use_qnn)
        self._dec, self._dec_opts = create_session(decoder_path, use_qnn=use_qnn)
        self._joi, self._joi_opts = create_session(joiner_path,  use_qnn=use_qnn)

        cfg = self._cfg
        dec, joi = self._dec, self._joi
        dec_opts, joi_opts = self._dec_opts, self._joi_opts

        def _decode(encoder_out, _cfg):
            return zipformer_transducer_decode(
                encoder_out, dec, joi, dec_opts, joi_opts,
                self._post._vocab, _cfg.blank_id, _cfg.context_size,
            )

        self._post.register("transducer_greedy", _decode)

    def unload(self) -> None:
        for attr in ("_enc", "_dec", "_joi"):
            if hasattr(self, attr):
                delattr(self, attr)

    def transcribe(self, data: bytes, **kwargs) -> Dict:
        import numpy as np

        feat = self._pre.process(data)   # [1, T, n_mels]

        # The encoder processes 71-frame windows (segment = decode_chunk_size*2 + 7)
        # but advances by 64 frames per step (offset = decode_chunk_size*2 = 32*2).
        # The 7-frame overlap provides left-context for the attention layers.
        segment = 71
        offset  = 64
        T = feat.shape[1]
        enc_outs = []
        enc_inputs = self._enc.get_inputs()
        out_names  = [o.name for o in self._enc.get_outputs()]

        states = {
            inp.name: np.zeros(inp.shape, dtype=np.float32
                               if "float" in inp.type else np.int32)
            for inp in enc_inputs
            if inp.name != "x"
        }

        for start in range(0, T, offset):
            frame = feat[:, start:start + segment, :]          # [1, <=71, 80]
            if frame.shape[1] < segment:
                frame = np.pad(frame, ((0, 0), (0, segment - frame.shape[1]), (0, 0)))
            feed = {"x": frame}
            feed.update(states)
            outs = self._enc.run(None, feed, self._enc_opts)
            out_dict = dict(zip(out_names, outs))
            enc_outs.append(out_dict["encoder_out"])          # [1, 16, 512]
            for name in states:
                new_name = name.replace("cached_", "new_cached_")
                if new_name in out_dict:
                    states[name] = out_dict[new_name]

        encoder_out = np.concatenate(enc_outs, axis=1)        # [1, T', 512]
        text = self._post.process(encoder_out)
        return {"text": text}
