"""
Example: ONNX Runtime QNN EP Plugin
=====================================
Shows how to wrap your existing ONNX Runtime + QNN EP code as a plugin.

    - id: my-onnx-llm
      backend: plugin
      plugin_module: examples/example_onnx_qnn_plugin.py
      plugin_class: OnnxQnnPlugin
      model_path: /path/to/model.onnx
      tokenizer_path: /path/to/tokenizer
      use_qnn: true
      qnn_backend: QnnHtp.dll
      max_tokens: 512
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, Iterator, List, Optional
from utils.inference_plugin import InferencePlugin

try:
    import onnxruntime as ort
    import numpy as np
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class OnnxQnnPlugin(InferencePlugin):
    """Plug your existing ONNX Runtime + QNN EP code here."""

    def load(self, model_id: str, model_config: Dict) -> None:
        """Load the ONNX session.  Paste your existing session creation code here."""
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")

        model_path  = model_config["model_path"]
        use_qnn     = model_config.get("use_qnn", True)
        qnn_backend = model_config.get("qnn_backend", "QnnHtp.dll")
        device_id   = str(model_config.get("qnn_device_id", 0))
        fp16        = model_config.get("qnn_fp16", True)
        tok_path    = model_config.get("tokenizer_path")
        self.max_tokens = int(model_config.get("max_tokens", 512))

        # Your existing ort.InferenceSession creation code
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if use_qnn:
            qnn_opts = {"backend_path": qnn_backend, "device_id": device_id}
            if fp16:
                qnn_opts["enable_htp_fp16_precision"] = "1"
            providers.append(("QNNExecutionProvider", qnn_opts))
        providers.append("CPUExecutionProvider")

        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_opts,
                                                providers=providers)
        except Exception:
            self.session = ort.InferenceSession(model_path, sess_options=sess_opts,
                                                providers=["CPUExecutionProvider"])

        self.tokenizer = None
        if tok_path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Any API's your function is using
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Run inference.  Paste your existing ONNX decode loop here.
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer_path is required for ONNX LLM inference.")

        tok = self.tokenizer
        enc = tok(prompt, return_tensors="np")
        input_ids = enc["input_ids"]
        generated_ids = input_ids.copy()
        eos_id = getattr(tok, "eos_token_id", None)
        input_names = {inp.name for inp in self.session.get_inputs()}


        for _ in range(max_tokens):
            feed = {}
            if "input_ids" in input_names:
                feed["input_ids"] = generated_ids
            if "attention_mask" in input_names:
                feed["attention_mask"] = np.ones_like(generated_ids)
            if "position_ids" in input_names:
                seq_len = generated_ids.shape[1]
                feed["position_ids"] = np.arange(seq_len, dtype=np.int64)[None, :]

            outputs = self.session.run(None, feed)
            logits = outputs[0][0, -1, :]

            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            next_id = int(np.argmax(logits))
            generated_ids = np.concatenate([generated_ids, np.array([[next_id]])], axis=1)

            token_text = tok.decode([next_id], skip_special_tokens=True)
            yield token_text

            if eos_id is not None and next_id == eos_id:
                break
            if stop and any(s in tok.decode(generated_ids[0, input_ids.shape[1]:],
                                            skip_special_tokens=True) for s in stop):
                break


    def unload(self) -> None:
        if hasattr(self, "session"):
            del self.session