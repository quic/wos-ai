# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
ONNX QNN Backend — local inference via ONNX Runtime with QNN Execution Provider.

Supports both CPU (onnxruntime) and NPU/HTP (onnxruntime-qnn) execution.

Example in model.yaml file when this backend is called:
models.yaml:
  backend: onnx_qnn
  model_path: /path/to/model.onnx
  tokenizer_path: /path/to/tokenizer
  use_qnn: true          # false = CPU, true = QNN/NPU
  qnn_backend: htp       # htp | cpu | gpu
  max_tokens: 2048
  lazy_load: true        # default — load on first client request
"""

import os
from typing import AsyncGenerator, Dict, List, Optional

from backends.base_backend import BaseBackend
from utils.logger import get_logger
from utils.prompt_formatter import PromptFormatter

logger = get_logger(__name__)


class _OnnxSession:
    """Holds ONNX Runtime session and tokenizer for one model."""
    def __init__(self, model_id: str, ort_session, tokenizer,
                 formatter: PromptFormatter, system_prompt: str, max_tokens: int):
        self.model_id = model_id
        self.ort_session = ort_session
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def free(self):
        """Release ONNX Runtime session and tokenizer."""
        if self.ort_session is not None:
            try:
                # ORT sessions release GPU/NPU memory when deleted
                del self.ort_session
            except Exception as e:
                logger.warning(f"[OnnxQnn:{self.model_id}] ort_session free error: {e}")
            finally:
                self.ort_session = None

        self.tokenizer = None
        logger.info(f"[OnnxQnn:{self.model_id}] Resources freed")


class OnnxQnnBackend(BaseBackend):

    def __init__(self):
        self._sessions: Dict[str, _OnnxSession] = {}

    # Lifecycle

    async def create_session(self, model_id: str, model_config: Dict) -> None:
        if model_id in self._sessions:
            return

        model_path    = model_config.get("model_path", "")
        tokenizer_path = model_config.get("tokenizer_path", "")
        use_qnn       = model_config.get("use_qnn", False)
        qnn_backend   = model_config.get("qnn_backend", "htp").lower()
        max_tokens    = int(model_config.get("max_tokens", 2048))
        sys_prompt    = model_config.get("system_prompt", "You are a helpful AI assistant.")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[OnnxQnn:{model_id}] model_path not found: '{model_path}'"
            )

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        # Build session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_qnn:
            # QNN Execution Provider (NPU/HTP)
            qnn_options = {
                "backend_path": "QnnHtp.dll" if qnn_backend == "htp" else f"Qnn{qnn_backend.capitalize()}.dll",
                "enable_htp_fp16_precision": "1",
            }
            providers = [("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
            logger.info(f"[OnnxQnn:{model_id}] Using QNN EP ({qnn_backend})")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info(f"[OnnxQnn:{model_id}] Using CPU EP")

        ort_session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers,
        )

        # Load tokenizer
        tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"[OnnxQnn:{model_id}] Tokenizer loaded from {tokenizer_path}")
            except Exception as exc:
                logger.warning(f"[OnnxQnn:{model_id}] Tokenizer load failed: {exc}")

        self._sessions[model_id] = _OnnxSession(
            model_id=model_id,
            ort_session=ort_session,
            tokenizer=tokenizer,
            formatter=PromptFormatter(),
            system_prompt=sys_prompt,
            max_tokens=max_tokens,
        )
        logger.info(f"[OnnxQnn:{model_id}] Ready. use_qnn={use_qnn}, max_tokens={max_tokens}")

    async def destroy_session(self, model_id: str) -> None:
        """Unload model and free ONNX Runtime session."""
        session = self._sessions.pop(model_id, None)
        if session:
            session.free()

    async def is_session_alive(self, model_id: str) -> bool:
        s = self._sessions.get(model_id)
        return s is not None and s.ort_session is not None

    # Sample API calls:
    # For Text generation for LLM's.
    async def generate(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Run ONNX inference. Yields the full response as a single token
        (ONNX models typically don't support streaming natively).
        For streaming, use the plugin backend with a custom generate() iterator.
        """
        session = self._sessions.get(model_id)
        if session is None:
            raise RuntimeError(f"[OnnxQnn] '{model_id}' not loaded")

        if session.tokenizer is None:
            raise RuntimeError(
                f"[OnnxQnn:{model_id}] No tokenizer. Set 'tokenizer_path' in models.yaml."
            )

        prompt = session.formatter.format(
            messages,
            tokenizer=session.tokenizer,
            default_system=session.system_prompt,
            add_generation_prompt=True,
        )

        n_tokens = max_tokens or session.max_tokens

        import asyncio
        loop = asyncio.get_running_loop()

        def _run_inference():
            inputs = session.tokenizer(prompt, return_tensors="np")
            input_ids = inputs["input_ids"]

            # Run ONNX session
            outputs = session.ort_session.run(
                None,
                {"input_ids": input_ids},
            )

            # Decode output tokens
            output_ids = outputs[0]
            if len(output_ids.shape) == 2:
                output_ids = output_ids[0]

            # Trim to max_tokens
            output_ids = output_ids[:n_tokens]

            # Remove input tokens from output (if model echoes input)
            input_len = input_ids.shape[-1]
            if len(output_ids) > input_len:
                output_ids = output_ids[input_len:]

            return session.tokenizer.decode(output_ids, skip_special_tokens=True)

        text = await loop.run_in_executor(None, _run_inference)

        # Apply stop sequences
        if stop:
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]

        yield text

    # Unsupport as this is example backend code. Need to improve this.
    async def embed(self, model_id, input, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support embeddings")

    async def transcribe(self, model_id, audio_bytes, filename, language, prompt,
                         response_format, temperature, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support transcription")

    async def translate(self, model_id, audio_bytes, filename, prompt,
                        response_format, temperature, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support translation")

    async def synthesize(self, model_id, text, voice, response_format, speed, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support TTS")

    async def moderate(self, model_id, input, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support moderation")

    async def image_generate(self, model_id, prompt, n, size, quality,
                              response_format, style, **kwargs):
        raise NotImplementedError(f"OnnxQnnBackend '{model_id}' does not support image generation")