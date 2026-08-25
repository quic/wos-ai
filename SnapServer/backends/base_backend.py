"""
Abstract interface that every inference backend must implement.

Two-phase lifecycle
-------------------
Phase 1 : create_session(): Load the model into memory once at startup.

Phase 2 : generate() / embed() / transcribe() / translate() / synthesize():
    Run inference per request.  Backends only implement the capabilities they support; unsupported methods raise NotImplementedError.

Supported capability methods
-----------------------------
  generate()       : chat / text completion  (all backends)
  embed()          : text embeddings
  transcribe()     : audio → text (Whisper)
  translate()      : audio → English text (Whisper translation mode)
  synthesize()     : text → audio (TTS)
  moderate()       : content moderation
  image_generate() : text → image
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional


class BaseBackend(ABC):
    """Abstract base class for all inference backends."""

    # Lifecycle methods

    @abstractmethod
    async def create_session(self, model_id: str, model_config: Dict) -> None:
        """Load the model into memory."""

    @abstractmethod
    async def destroy_session(self, model_id: str) -> None:
        """Release all resources held by this model session."""

    @abstractmethod
    async def is_session_alive(self, model_id: str) -> bool:
        """Return True if the session is loaded and ready."""


    # For Text generation 
    @abstractmethod
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
        Generate a response, yielding tokens one by one (streaming).
        Subclasses must use `yield` to make this an async generator.
        """
        raise NotImplementedError
        yield  # noqa: unreachable : makes this an async generator


    # For Embeddings 
    async def embed(
        self,
        model_id: str,
        input: List[str],
        **kwargs,
    ) -> List[List[float]]:
        """Compute text embeddings. Returns list of embedding vectors."""
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support embeddings."
        )


    # For Audio transcription
    async def transcribe(
        self,
        model_id: str,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        """
        Transcribe audio to text (Whisper-style).
        Returns dict with at least {"text": "transcribed text"}.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support transcription."
        )


    # For Audio translation
    async def translate(
        self,
        model_id: str,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        """
        Translate audio to English text (Whisper translation mode).
        Returns dict with at least {"text": "translated text"}.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support audio translation."
        )


    # For Text-to-speech 
    async def synthesize(
        self,
        model_id: str,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Synthesize speech from text (TTS).
        Returns raw audio bytes in the requested format.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support TTS."
        )


    # For Content moderation
    async def moderate(
        self,
        model_id: str,
        input: str,
        **kwargs,
    ) -> Dict:
        """
        Run content moderation on input text.
        Returns OpenAI-compatible moderation result dict.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support moderation."
        )


    # For Image generation 
    async def image_generate(
        self,
        model_id: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
        style: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate images from a text prompt.
        Returns list of dicts with 'url' or 'b64_json' keys.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support image generation."
        )
    
    async def image_edit(
        self,
        model_id: str,
        image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs,
    ) -> List[Dict]:
        """
        Edit an image (inpainting).  →  POST /v1/images/edits
        Returns list of dicts with 'url' or 'b64_json' keys.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support image editing."
        )

    async def image_variation(
        self,
        model_id: str,
        image: bytes,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs,
    ) -> List[Dict]:
        """
        Generate variations of an image.  →  POST /v1/images/variations
        Returns list of dicts with 'url' or 'b64_json' keys.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support image variation."
        )

    # For Convenience purpose: collect full responses 
    async def generate_full(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Collect all streamed tokens into a single string."""
        parts = []
        async for token in self.generate(
            model_id, messages, temperature, max_tokens, stop, **kwargs
        ):
            parts.append(token)
        return "".join(parts)