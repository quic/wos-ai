import io
import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.venv_plugin import VenvPlugin


class WhisperASRPlugin(VenvPlugin):

    def load(self, model_id: str, config: dict) -> None:
        """Initialize the Whisper pipeline. Called after venv paths are set up."""
        from onnx_inference import ONNXWhisperPipeline

        source_dir = config.get("source_dir") or config.get("asr_source_dir", "")

        self.pipeline = ONNXWhisperPipeline(
            encoder_path   = config["encoder_path"],
            decoder_path   = config["decoder_path"],
            tokenizer_path = config.get("tokenizer_path", os.path.join(source_dir, "whisper-base")),
            hf_model_id    = config.get("hf_model_id", "openai/whisper-base"),
            use_qnn        = config.get("use_qnn", True),
        )

    def transcribe(self, audio_bytes: bytes, **kwargs) -> dict:
        """Transcribe audio bytes to text."""
        import soundfile as sf

        audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32)

        text = self.pipeline.transcribe(audio_np, audio_sample_rate=sample_rate)
        return {"text": text}

    def unload(self) -> None:
        """Free QNN sessions."""
        if hasattr(self, "pipeline"):
            del self.pipeline.encoder_session
            del self.pipeline.decoder_session
            del self.pipeline


