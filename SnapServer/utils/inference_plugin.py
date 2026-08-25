"""
Inference Plugin Interface
==========================
Base class that users implement to plug their existing local inference code into the server WITHOUT rewriting it.

How it works
------------
1. User creates a Python file (e.g. my_model.py) with a class that inherits from InferencePlugin.
2. They implement load(), unload() and openai endpoint method using their existing code wrapped to below methods.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  OpenAI Endpoint                  │  Plugin method                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  POST /v1/chat/completions        │  generate_with_messages_cb()  ← fast│
  │                                   │  generate_with_messages()     ← alt  │
  │                                   │  generate()                   ← basic│
  │  POST /v1/completions (legacy)    │  generate()                         │
  │  POST /v1/embeddings              │  embed()                            │
  │  POST /v1/audio/transcriptions    │  transcribe()                       │
  │  POST /v1/audio/translations      │  translate()                        │
  │  POST /v1/audio/speech            │  synthesize()                       │
  │  POST /v1/moderations             │  moderate()                         │
  │  POST /v1/images/generations      │  image_generate()                   │
  │  POST /v1/images/edits            │  image_edit()                       │
  │  POST /v1/images/variations       │  image_variation()                  │
  │  POST /v1/models/{id}/reset_dialog│  reset_dialog()                     │
  │  PATCH /v1/models/{id}/system_prompt│ set_system_prompt()               │
  └─────────────────────────────────────────────────────────────────────────

3. They point to it in models.yaml via 'plugin_module' and 'plugin_class'.
4. The server calls load() ONCE at startup, then openai_endpoint() per request.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union


class InferencePlugin(ABC):
    """
    Abstract base class for any inference plugins with openai endpoints.

    Only load() and unload() are mandatory.
    All other methods are optional — implement only what your plugin supports.
    """

    # Required
    @abstractmethod
    def load(self, model_id: str, model_config: Dict) -> None:
        """
        Load the model into memory.  Called ONCE at server startup.

        Args:
            model_id:     The model's ID string from models.yaml.
            model_config: The full config dict for this model from models.yaml.
                          Use it to read paths, parameters, etc.

        Raise any exception on failure — PluginBackend will catch and re-raise.
        """
    @abstractmethod
    def unload(self) -> None:
        """
        Free all resources (sessions, handles, GPU/NPU memory).
        Called when the model is unloaded or the server shuts down.
        Must not raise — log warnings instead.
        """

    # Any Open AI Endpoint inference method 
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Single-turn text generation.  Yield tokens one at a time.
        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate(). "
            "Implement generate(), generate_with_messages(), or "
            "generate_with_messages_cb() to support POST /v1/chat/completions."
        )

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice=None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Multi-turn text generation from a messages list.  Yield tokens one at a time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate_with_messages()."
        )

    def generate_with_messages_cb(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
        on_token=None,
        tools: Optional[List[Dict]] = None,
        tool_choice=None,
        **kwargs,
    ) -> None:
        """
        Fast-path multi-turn generation. 
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate_with_messages_cb()."
        )

    def embed(self, input: Any, **kwargs) -> List[List[float]]:
        """
        Compute text embeddings.

        Args:
            input: str or List[str] — text(s) to embed.

        Returns:
            List of embedding vectors (one per input string).

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement embed(). "
            "Implement embed() to support POST /v1/embeddings."
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        """
        Speech-to-text transcription.  →  POST /v1/audio/transcriptions

        Returns:
            {"text": "transcribed text"}

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement transcribe(). "
            "Implement transcribe() to support POST /v1/audio/transcriptions."
        )

    def translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        """
        Speech-to-English translation.  →  POST /v1/audio/translations

        Returns:
            {"text": "translated text in English"}

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement translate(). "
            "Implement translate() to support POST /v1/audio/translations."
        )

    def synthesize(
        self,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Text-to-speech synthesis.  →  POST /v1/audio/speech

        Returns:
            Audio bytes in the requested format (mp3, wav, opus, aac, flac).

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement synthesize(). "
            "Implement synthesize() to support POST /v1/audio/speech."
        )

    def moderate(self, input: Any, **kwargs) -> Dict:
        """
        Content moderation.  →  POST /v1/moderations

        Args:
            input: str or List[str] — text(s) to moderate.

        Returns:
            OpenAI-compatible moderation response dict.

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement moderate(). "
            "Implement moderate() to support POST /v1/moderations."
        )

    def image_generate(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
        style: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Text-to-image generation.  →  POST /v1/images/generations

        Returns:
            List of image dicts: [{"url": "..."}, ...] or [{"b64_json": "..."}, ...]

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image_generate(). "
            "Implement image_generate() to support POST /v1/images/generations."
        )

    def image_edit(
        self,
        image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs,
    ) -> List[Dict]:
        """
        Image editing (inpainting).  →  POST /v1/images/edits

        Args:
            image:  Original image bytes (PNG, RGBA, < 4 MB).
            prompt: Text description of the desired edit.
            mask:   Optional mask bytes — transparent areas will be edited.

        Returns:
            List of edited image dicts.

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image_edit(). "
            "Implement image_edit() to support POST /v1/images/edits."
        )

    def image_variation(
        self,
        image: bytes,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs,
    ) -> List[Dict]:
        """
        Image variation generation.  →  POST /v1/images/variations

        Args:
            image: Source image bytes (PNG, square, < 4 MB).

        Returns:
            List of variation image dicts.

        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image_variation(). "
            "Implement image_variation() to support POST /v1/images/variations."
        )

    def reset_dialog(self) -> None:
        """
        Reset KV cache / conversation state.
        Called by POST /v1/models/{id}/reset_dialog.

        Default: no-op — stateless plugins do not need KV cache reset.
        """
        pass  # no-op by default

    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt at runtime.
        Called by PATCH /v1/models/{id}/system_prompt.

        Default: no-op — plugins that don't use a system prompt can ignore this.
        """

        pass  # no-op by default
