# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Cloud Backend — proxies requests to OpenAI, Azure, Anthropic, etc.

Example from models.yaml when this file is called:
models.yaml:
  backend: cloud
  provider: openai              # openai | azure | anthropic | google
  model_name: gpt-4o            # model name as the provider knows it
  api_key_env: OPENAI_API_KEY   # optional 
  base_url: https://...         # optional override
  lazy_load: true               # default — no local resources, instant
"""

import json
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from backends.base_backend import BaseBackend
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class CloudBackend(BaseBackend):
    """ Async backend that proxies to an OpenAI-compatible cloud API."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    # Lifecycle

    async def create_session(self, model_id: str, model_config: Dict) -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        backend  = model_config.get("backend", "").lower()
        api_key  = model_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = model_config.get("base_url") or None

        if backend == "azure":
            api_version = model_config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
            if not base_url:
                raise ValueError(f"[CloudBackend] '{model_id}': 'base_url' is required for Azure backend")
            client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
            )
        else:
            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        self._sessions[model_id] = {
            "client":     client,
            "model_name": model_config.get("model_name", model_id),
            "config":     model_config,
        }
        logger.info(f"[CloudBackend] '{model_id}' ready → {model_config.get('model_name', model_id)}")

    async def destroy_session(self, model_id: str) -> None:
        self._sessions.pop(model_id, None)
        logger.info(f"[CloudBackend] '{model_id}' destroyed")

    async def is_session_alive(self, model_id: str) -> bool:
        return model_id in self._sessions

    # Text generation 

    async def generate(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[object] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        s = self._get(model_id)
        params: dict = dict(model=s["model_name"], messages=messages,
                            temperature=temperature, stream=True)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = stop
        if top_p is not None:
            params["top_p"] = top_p
        if presence_penalty:
            params["presence_penalty"] = presence_penalty
        if frequency_penalty:
            params["frequency_penalty"] = frequency_penalty
        if seed is not None:
            params["seed"] = seed
        if response_format:
            params["response_format"] = response_format

        if tools:
            params.pop("stream")
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
            params["tools"] = tools
            resp = await s["client"].chat.completions.create(**params)
            choice = resp.choices[0]
            if choice.finish_reason == "tool_calls":
                yield "\x00TOOL_CALLS\x00" + json.dumps({
                    "content": choice.message.content,
                    "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls],
                })
            else:
                yield choice.message.content or ""
            return

        stream = await s["client"].chat.completions.create(**params)
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    # Embeddings

    async def embed(self, model_id: str, input, **kwargs):
        s = self._get(model_id)
        inputs = [input] if isinstance(input, str) else input
        resp = await s["client"].embeddings.create(model=s["model_name"], input=inputs)
        return [item.embedding for item in resp.data]

    # Audio 

    async def transcribe(self, model_id: str, audio_bytes: bytes,
                         filename: str = "audio.wav", language: str = None,
                         prompt: str = None, response_format: str = "json",
                         temperature: float = 0.0, **kwargs):
        import io
        s = self._get(model_id)
        params = dict(model=s["model_name"],
                      file=(filename, io.BytesIO(audio_bytes), "audio/wav"),
                      response_format=response_format, temperature=temperature)
        if language: params["language"] = language
        if prompt:   params["prompt"]   = prompt
        result = await s["client"].audio.transcriptions.create(**params)
        return {"text": result.text}

    async def translate(self, model_id: str, audio_bytes: bytes,
                        filename: str = "audio.wav", prompt: str = None,
                        response_format: str = "json", temperature: float = 0.0,
                        **kwargs):
        import io
        s = self._get(model_id)
        params = dict(model=s["model_name"],
                      file=(filename, io.BytesIO(audio_bytes), "audio/wav"),
                      response_format=response_format, temperature=temperature)
        if prompt: params["prompt"] = prompt
        result = await s["client"].audio.translations.create(**params)
        return {"text": result.text}

    async def synthesize(self, model_id: str, text: str, voice: str = "alloy",
                         response_format: str = "mp3", speed: float = 1.0, **kwargs):
        s = self._get(model_id)
        resp = await s["client"].audio.speech.create(
            model=s["model_name"], input=text, voice=voice,
            response_format=response_format, speed=speed)
        return resp.content

    # Moderation

    async def moderate(self, model_id: str, input, **kwargs):
        s = self._get(model_id)
        resp = await s["client"].moderations.create(model=s["model_name"], input=input)
        return {"results": [r.model_dump() for r in resp.results]}

    # Image generation 

    async def image_generate(self, model_id: str, prompt: str, n: int = 1,
                              size: str = "1024x1024", quality: str = "standard",
                              response_format: str = "url", style: str = None,
                              **kwargs):
        s = self._get(model_id)
        params = dict(model=s["model_name"], prompt=prompt, n=n, size=size,
                      quality=quality, response_format=response_format)
        if style: params["style"] = style
        resp = await s["client"].images.generate(**params)
        return [{"url": img.url} if response_format == "url"
                else {"b64_json": img.b64_json} for img in resp.data]

    async def image_edit(self, model_id: str, image: bytes, prompt: str,
                         mask: bytes = None, n: int = 1, size: str = "1024x1024",
                         response_format: str = "url", **kwargs):
        import io as _io
        s = self._get(model_id)
        params = dict(
            model=s["model_name"],
            image=("image.png", _io.BytesIO(image), "image/png"),
            prompt=prompt, n=n, size=size, response_format=response_format,
        )
        if mask:
            params["mask"] = ("mask.png", _io.BytesIO(mask), "image/png")
        resp = await s["client"].images.edit(**params)
        return [{"url": img.url} if response_format == "url"
                else {"b64_json": img.b64_json} for img in resp.data]

    async def image_variation(self, model_id: str, image: bytes, n: int = 1,
                               size: str = "1024x1024", response_format: str = "url", **kwargs):
        import io as _io
        s = self._get(model_id)
        params = dict(
            model=s["model_name"],
            image=("image.png", _io.BytesIO(image), "image/png"),
            n=n, size=size, response_format=response_format,
        )
        resp = await s["client"].images.create_variation(**params)
        return [{"url": img.url} if response_format == "url"
                else {"b64_json": img.b64_json} for img in resp.data]

    # Helper 

    def _get(self, model_id: str) -> dict:
        s = self._sessions.get(model_id)
        if s is None:
            raise RuntimeError(f"[CloudBackend] '{model_id}' not loaded")
        return s

    @property
    def _client(self):
        """Return the first available AsyncOpenAI client (used by session_manager)."""
        for s in self._sessions.values():
            return s["client"]
        return None

    def _client_for(self, model_id: str):
        """Return (client, model_name, backend_type) for a registered cloud model."""
        s = self._get(model_id)
        backend_type = s["config"].get("backend", "cloud")
        return s["client"], s["model_name"], backend_type

    def get_any_client(self):
        """Return (client, model_name, backend_type) for any registered cloud model."""
        for model_id, s in self._sessions.items():
            backend_type = s["config"].get("backend", "cloud")
            return s["client"], s["model_name"], backend_type
        raise RuntimeError("No cloud model is currently loaded.")