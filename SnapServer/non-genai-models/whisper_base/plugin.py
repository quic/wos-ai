# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Whisper-Base plugin — multilingual ASR via encoder+decoder ONNX.
output_type: attention_greedy

In-process plugin for the WoS server. VenvPlugin injects source_dir and
venv site-packages onto sys.path before load() runs, so all pipeline_core
imports happen inside load() after the paths are set up.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.venv_plugin import VenvPlugin
except ModuleNotFoundError:
    class VenvPlugin:   # minimal stand-in when running outside the WoS server
        pass


class WhisperBasePlugin(VenvPlugin):

    def load(self, model_id: str, config: Dict) -> None:
        import numpy as np
        from pipeline_core import (
            ConfigLoader, ModelConfig, OnnxModelInspector,
            Preprocessor, Postprocessor, create_session,
            load_vocab, _vocab_id_to_token, whisper_attention_decode,
        )

        encoder_path   = config["encoder_path"]
        decoder_path   = config["decoder_path"]
        config_path    = config["config_path"]
        use_qnn        = config.get("use_qnn", False)
        max_decode_len = config.get("max_decode_len", 224)

        enc_inspector = OnnxModelInspector(encoder_path, use_qnn=use_qnn)
        loader        = ConfigLoader(config_path)
        cfg           = ModelConfig(enc_inspector, loader)

        self._pre = Preprocessor(cfg)
        self._encoder, self._run_opts = create_session(encoder_path, use_qnn=use_qnn)
        decoder, _                    = create_session(decoder_path, use_qnn=use_qnn)

        vocab    = load_vocab(cfg.vocab_file)
        id2tok   = _vocab_id_to_token(vocab)
        eot_id   = vocab.get("<|endoftext|>", 50257)
        suppress = np.array(
            [i for i in range(50257, max(vocab.values()) + 1) if i != eot_id],
            dtype=np.int64,
        )

        self._post = Postprocessor(cfg)
        self._post.register(
            "attention_greedy",
            lambda cross_kv, c: whisper_attention_decode(
                decoder=decoder,
                run_opts=self._run_opts,
                cross_kv=cross_kv,
                suppress_ids=suppress,
                sot_id=vocab.get("<|startoftranscript|>", 50258),
                eot_id=eot_id,
                lang_id=vocab.get("<|en|>", 50259),
                transcribe_id=vocab.get("<|transcribe|>", 50359),
                notimestamps_id=vocab.get("<|notimestamps|>", 50363),
                max_decode_len=max_decode_len,
                id_to_token=id2tok,
            ),
        )

    def unload(self) -> None:
        for attr in ("_encoder", "_pre", "_post", "_run_opts"):
            if hasattr(self, attr):
                delattr(self, attr)

    def transcribe(self, audio_bytes: bytes,
                   language: Optional[str] = None, **kwargs) -> Dict:
        mel      = self._pre.process(audio_bytes)
        enc_outs = self._encoder.run(
            None,
            {self._encoder.get_inputs()[0].name: mel},
            self._run_opts,
        )
        cross_kv = {o.name: enc_outs[i] for i, o in enumerate(self._encoder.get_outputs())}
        text = self._post.process(cross_kv)
        return {"text": text}
