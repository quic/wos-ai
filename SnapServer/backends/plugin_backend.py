# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Plugin Backend — loads user-defined inference plugins at runtime.

Plugin contract (implement these methods):
  load(model_id: str, config: dict) -> None
  Any OpenAI API, example: generate(prompt: str, max_tokens: int, temperature: float, stop: list) -> Iterator[str]
  unload() -> None                    # free all resources (dialog, model, session)
  reset_dialog() -> None              # optional — reset multi-turn state
"""

import importlib.util
import json
import os
import sys
import asyncio
from typing import AsyncGenerator, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from backends.base_backend import BaseBackend
from utils.logger import get_logger
from utils.prompt_formatter import PromptFormatter

logger = get_logger(__name__)

# Dedicated inference executor
_GENIE_EXECUTOR = ThreadPoolExecutor(max_workers=1,thread_name_prefix="genie-infer")
_ORT_EXECUTOR = ThreadPoolExecutor(max_workers=1,thread_name_prefix="ort-infer")

# Pre-warm both executors at import time — avoids the first-request thread
_GENIE_EXECUTOR.submit(lambda: None)
_ORT_EXECUTOR.submit(lambda: None)

_GENIE_CLASS_MARKERS = ("genie", "Genie", "GENIE", "GeniePlugin", "genieplugin")


# self._executor_for(model_id) = ThreadPoolExecutor(max_workers=1, thread_name_prefix="genie-infer") # max_workers=1: GenieDialog is NOT thread-safe — only one query at a time.
# self._executor_for(model_id).submit(lambda: None)   # pre-warm at import to avoid thread-creation cost later

class PluginBackend(BaseBackend):

    def __init__(self):
        self._plugins: Dict[str, object] = {}
        self._configs: Dict[str, dict] = {}
        self._formatter = PromptFormatter()

    def _get_plugin(self, model_id: str):
        """Return the loaded plugin or raise RuntimeError."""
        plugin = self._plugins.get(model_id)
        if plugin is None:
            raise RuntimeError(f"[PluginBackend] '{model_id}' not loaded")
        return plugin
    
    def _executor_for(self, model_id: str):
        """Return the correct ThreadPoolExecutor for the model. Routing is based on the plugin class name"""

        plugin = self._plugins.get(model_id)
        if plugin is not None:
            cls_name = type(plugin).__name__
            if any(marker in cls_name for marker in _GENIE_CLASS_MARKERS):
                return _GENIE_EXECUTOR

        return _ORT_EXECUTOR

    def _call(self, model_id: str, method: str, *args, required: bool = True, **kwargs):
        """Check which plugin.method is called"""
        plugin = self._get_plugin(model_id)
        fn = getattr(plugin, method, None)
        if not callable(fn):
            if required:
                raise NotImplementedError(
                    f"Plugin '{model_id}' does not implement '{method}'. "
                    f"Add  def {method}(self, ...)  to your plugin class "
                    f"(see sample_plugins/example_template_plugin.py)."
                )
            return None
        return fn(*args, **kwargs)

    async def create_session(self, model_id: str, config: dict) -> None:
        plugin_path = config.get("plugin_path", config.get("plugin_module", ""))
        plugin_class = config.get("plugin_class", "")

        if not plugin_path or not plugin_class:
            raise ValueError(
                f"[PluginBackend] '{model_id}' requires 'plugin_path' and 'plugin_class' in config"
            )

        logger.info(f"[PluginBackend] Loading '{model_id}' via {plugin_path}::{plugin_class}")
        
        try:
            plugin = self._load_plugin_class(plugin_path, plugin_class)()
            plugin.load(model_id, config)
            
            self._plugins[model_id] = plugin
            self._configs[model_id] = config
            logger.info(f"[PluginBackend] '{model_id}' ready")
        except Exception as e:
            # Re-raise with more context but preserve the original error
            logger.error(f"[PluginBackend] Failed to load '{model_id}': {e}", exc_info=True)
            raise

    async def destroy_session(self, model_id: str) -> None:
        plugin = self._plugins.pop(model_id, None)
        self._configs.pop(model_id, None)
        if plugin:
            fn = getattr(plugin, "unload", None)
            if callable(fn):
                try:
                    fn()
                    logger.info(f"[PluginBackend] '{model_id}' unloaded")
                except Exception as e:
                    logger.warning(f"[PluginBackend] unload error for '{model_id}': {e}")
       

    async def is_session_alive(self, model_id: str) -> bool:
        return model_id in self._plugins

    async def generate(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        tools: Optional[List] = None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        plugin = self._get_plugin(model_id)
        config = self._configs.get(model_id, {})
        n_tokens = max_tokens or config.get("max_tokens", 512)

        # Priority: generate_with_messages_cb > generate_with_messages > generate
        has_cb   = callable(getattr(plugin, "generate_with_messages_cb",  None))
        has_msgs = callable(getattr(plugin, "generate_with_messages",     None))

        prompt = None   # defined below for single-turn path; None for multi-turn
        if has_msgs or has_cb:
            prompt_desc = f"{len(messages)} messages (multi-turn)"
        else:
            tokenizer = getattr(plugin, "tokenizer", None)
            prompt = self._formatter.format(
                messages,
                tokenizer=tokenizer,
                template=config.get("chat_template"),
                default_system=config.get("system_prompt"),
                tools=tools,
            )
            prompt_desc = f"{len(prompt)} chars (single-turn)"

        import asyncio
        loop       = asyncio.get_running_loop()
        token_q: asyncio.Queue = asyncio.Queue()

        logger.info(f"[PluginBackend:{model_id}] generate() called, prompt={prompt_desc}")


        def _run():
            path = "GenieAPI's path" if has_cb else ("multi-turn" if has_msgs else "single-turn")
            logger.info(
                f"[PluginBackend:{model_id}] _run() thread started, path={path}"
            )
            try:
                if has_cb:
                    # dialog.query() in THIS thread
                    def _on_token(token: str):
                        loop.call_soon_threadsafe(token_q.put_nowait, token)

                    plugin.generate_with_messages_cb(
                        messages=messages, max_tokens=n_tokens,
                        temperature=temperature, stop=stop or [],
                        on_token=_on_token,
                        tools=tools, tool_choice=tool_choice,
                    )

                elif has_msgs:
                    # Generator path: multi-turn
                    for token in plugin.generate_with_messages(
                        messages=messages, max_tokens=n_tokens,
                        temperature=temperature, stop=stop or [],
                        tools=tools, tool_choice=tool_choice,
                    ):
                        loop.call_soon_threadsafe(token_q.put_nowait, token)

                else:
                    # Generator path: single-turn
                    for token in plugin.generate(
                        prompt=prompt, max_tokens=n_tokens,
                        temperature=temperature, stop=stop or [],
                    ):
                        loop.call_soon_threadsafe(token_q.put_nowait, token)

            except Exception as exc:
                loop.call_soon_threadsafe(token_q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(token_q.put_nowait, None)  # sentinel

        loop.run_in_executor(self._executor_for(model_id), _run)

        # When tools are provided
        if tools:
            collected: List[str] = []
            while True:
                item = await token_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error(
                        f"[PluginBackend:{model_id}] Inference error (tool path): {item!r}"
                    )
                    break
                collected.append(item)
            full_text = "".join(collected)
            tool_calls = _parse_tool_calls(full_text)
            if tool_calls:
                logger.info(
                    f"[PluginBackend:{model_id}] Tool call detected — {len(tool_calls)} call(s)"
                )
                yield "\x00TOOL_CALLS\x00" + json.dumps({"content": None, "tool_calls": tool_calls})
            else:
                yield full_text
            logger.info(f"[PluginBackend:{model_id}] Generation COMPLETE (tool path)")
            return

        token_count   = 0
        while True:
            item = await token_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                # DO NOT re-raise — WinError 10054 issue.
                logger.error(
                    f"[PluginBackend:{model_id}] Inference error (streaming stopped): {item!r}"
                )
                break
            token_count += 1
            yield item

        logger.info(
            f"[PluginBackend:{model_id}] Generation COMPLETE — {token_count} tokens"
        )

    async def embed(self, model_id: str, input, **kwargs):
        """POST /v1/embeddings — plugin must implement embed(input, **kwargs)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id),lambda: self._call(model_id, "embed", input, **kwargs))

    async def transcribe(self, model_id: str, audio_bytes: bytes, filename: str = "audio.wav", language: str = None,
                         prompt: str = None, response_format: str = "json", temperature: float = 0.0, **kwargs):
        """POST /v1/audio/transcriptions — plugin must implement transcribe(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id),
            lambda: self._call(model_id, "transcribe", audio_bytes,
                               filename=filename, language=language, prompt=prompt,
                               response_format=response_format,
                               temperature=temperature, **kwargs))

    async def translate(self, model_id: str, audio_bytes: bytes, filename: str = "audio.wav", prompt: str = None,
                        response_format: str = "json", temperature: float = 0.0, **kwargs):
        """POST /v1/audio/translations — plugin must implement translate(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id),
            lambda: self._call(model_id, "translate", audio_bytes,
                               filename=filename, prompt=prompt,
                               response_format=response_format,
                               temperature=temperature, **kwargs))

    async def synthesize(self, model_id: str, text: str, voice: str = "alloy", response_format: str = "mp3", speed: float = 1.0, **kwargs):
        """POST /v1/audio/speech — plugin must implement synthesize(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id),
            lambda: self._call(model_id, "synthesize", text,
                               voice=voice, response_format=response_format,
                               speed=speed, **kwargs))

    async def moderate(self, model_id: str, input, **kwargs):
        """POST /v1/moderations — plugin must implement moderate(input, **kwargs)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id), lambda: self._call(model_id, "moderate", input, **kwargs))

    async def image_generate(self, model_id: str, prompt: str, n: int = 1, size: str = "1024x1024", quality: str = "standard",
                              response_format: str = "url", style: str = None, **kwargs):
        """POST /v1/images/generations — plugin must implement image_generate(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id), 
                                          lambda: self._call(model_id, "image_generate", prompt, n=n, size=size, quality=quality, 
                                                             response_format=response_format, style=style, **kwargs))

    async def image_edit(self, model_id: str, image: bytes, prompt: str, mask: bytes = None, n: int = 1, size: str = "1024x1024",
                         response_format: str = "url", **kwargs):
        """POST /v1/images/edits — plugin must implement image_edit(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id), lambda: self._call(model_id, "image_edit", image, prompt,
                                                                                           mask=mask, n=n, size=size, 
                                                                                           response_format=response_format, **kwargs))

    async def image_variation(self, model_id: str, image: bytes, n: int = 1, size: str = "1024x1024", response_format: str = "url",**kwargs):
        """POST /v1/images/variations — plugin must implement image_variation(...)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_for(model_id), 
                                          lambda: self._call(model_id, "image_variation", image, n=n, size=size, 
                                                             response_format=response_format,**kwargs))

    async def reset_dialog(self, model_id: str) -> None:
        """POST /v1/models/{id}/reset_dialog"""
        plugin = self._get_plugin(model_id)
        if callable(getattr(plugin, "reset_dialog", None)):
            plugin.reset_dialog()
            logger.info(f"[PluginBackend:{model_id}] reset_dialog() called on plugin")
        else:
            logger.info(f"[PluginBackend:{model_id}] plugin has no reset_dialog — no-op")

    def get_prompt_diagnostics(self, model_id: str) -> Optional[Dict]:
        """GET /status"""
        plugin = self._plugins.get(model_id)
        if plugin is None:
            return None
        get_fn = getattr(plugin, "get_prompt_diagnostics", None)
        return get_fn() if callable(get_fn) else None

    def set_system_prompt(self, model_id: str, prompt: str) -> None:
        """ PATCH /v1/models/{id}/system_prompt
        Sync method — called by session_manager.set_system_prompt().
        Updates both the stored config dict (so future load() calls pick it up) and the live plugin instance (via plugin.set_system_prompt() or direct attribute assignment).
        """
        config = self._configs.get(model_id, {})
        config["system_prompt"] = prompt

        plugin = self._plugins.get(model_id)
        if plugin:
            if callable(getattr(plugin, "set_system_prompt", None)):
                plugin.set_system_prompt(prompt)
            elif hasattr(plugin, "system_prompt"):
                # Fallback: direct attribute assignment
                plugin.system_prompt = prompt
        logger.info(f"[PluginBackend:{model_id}] system_prompt updated")

    # Helpers 
    @staticmethod
    def _load_plugin_class(plugin_path: str, class_name: str):
        
        # Dotted module path (e.g. "sample_plugins.genie_plugin") 
        if not plugin_path.endswith(".py") and os.sep not in plugin_path and "/" not in plugin_path:
            import importlib as _importlib
            try:
                module = _importlib.import_module(plugin_path)
            except ImportError as e:
                raise ImportError( f"Could not import plugin module '{plugin_path}': {e}\n"
                    f"Ensure the module is on sys.path or use a file path instead.") from e
            if not hasattr(module, class_name):
                raise AttributeError( f"Class '{class_name}' not found in module '{plugin_path}'. "
                    f"Check 'plugin_class' in models.yaml." )
            return getattr(module, class_name)

        # File path (e.g. "sample_plugins/genie_plugin.py")
        abs_path = os.path.abspath(plugin_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Plugin file not found: {abs_path}")

        module_key = "plugin_" + abs_path.replace("\\", "_").replace("/", "_").replace(":", "").replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_key, abs_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {abs_path}")
        return getattr(module, class_name)

# Tool-call output parser
def _scan_json_values(text: str) -> List[object]:
    """Find every top-level JSON object/array in text, in order."""
    decoder = json.JSONDecoder()
    values: List[object] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in "{[":
            try:
                obj, end = decoder.raw_decode(text, i)
                values.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return values


def _parse_tool_calls(text: str) -> Optional[List[Dict]]:
    """Try to extract tool call(s) from raw model output."""
    import uuid

    def _to_openai(obj) -> Optional[Dict]:
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        if not name and isinstance(obj.get("function"), dict):
            name = obj["function"].get("name")
        if not name:
            return None
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters", {})
        args_str = args if isinstance(args, str) else json.dumps(args)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        }

    calls: List[Dict] = []
    for obj in _scan_json_values(text):
        candidates = obj if isinstance(obj, list) else [obj]
        for candidate in candidates:
            tc = _to_openai(candidate)
            if tc:
                calls.append(tc)

    return calls or None