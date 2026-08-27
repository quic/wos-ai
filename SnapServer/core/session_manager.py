# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Session Manager
===============
Routes all API calls to the correct backend.

SYSTEM PROMPT
-------------
Priority (highest → lowest):
  1. system message in the request messages[] array
       {"role": "system", "content": "You are a coding assistant."}
  2. system_prompt field in models.yaml
       system_prompt: "You are a helpful AI assistant."
  3. Built-in default: "You are a helpful AI assistant."
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from backends.base_backend import BaseBackend
from backends.cloud_backend import CloudBackend
from backends.onnx_qnn_backend import OnnxQnnBackend
from backends.plugin_backend import PluginBackend
from core.metrics import MODEL_INFERENCE_TOTAL, MODEL_LOAD_EVENTS_TOTAL, safe_inc
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelState(str, Enum):
    NOT_LOADED    = "not_loaded"     # registered, lazy, never requested
    LOADING       = "loading"        # _load_model() in flight
    LOADED        = "loaded"         # create_session() succeeded
    LOAD_FAILED   = "load_failed"    # last _load_model() attempt raised
    UNLOADED      = "unloaded"       # explicitly unloaded via /v1/models/{id}/unload
    UNLOADED_IDLE = "unloaded_idle"  # auto-unloaded by the idle checker

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken (accurate) or word-split (fallback)."""
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text.split()))


def _message_text(msg: Dict) -> str:
    """Extract countable text from a message, tolerating None/list content"""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)]
        return " ".join(parts)
    return ""


def _count_messages_tokens(messages: List[Dict], model: str = "gpt-4o") -> int:
    """Count tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += _count_tokens(_message_text(msg), model)
        total += 4  # per-message overhead
    total += 2  # reply priming
    return total


class SessionManager:
    """Owns all backend instances and routes requests to the correct one."""

    def __init__(self):
        self._backends: Dict[str, BaseBackend] = {
            "plugin":   PluginBackend(),
            "onnx_qnn": OnnxQnnBackend(),
            "cloud":    CloudBackend(),
        }
        self._model_backend: Dict[str, str]  = {}   # model_id → backend key
        self._model_configs: Dict[str, Dict] = {}   # model_id → config (mutable)
        self._lazy_pending:  set             = set() # model_ids not yet loaded

        # Per-model locks — concurrent requests to different models run in parallel;
        # requests to the same model are serialised (Genie dialog is not thread-safe)
        self._init_locks: Dict[str, asyncio.Lock] = {}
        self._init_locks_lock = asyncio.Lock()

        # Separate per-model locks for inference serialisation.
        # _infer_locks are held for the entire duration of each inference call.
        self._infer_locks: Dict[str, asyncio.Lock] = {}
        self._infer_locks_lock = asyncio.Lock()

        # Global NPU load lock — prevents two different plugin models from calling create_session() simultaneously. 
        self._npu_load_lock = asyncio.Lock()

        # Idle timeout tracking
        # _last_used[model_id] = monotonic timestamp of last inference call
        # _idle_timeout[model_id] = seconds before auto-unload (None = never)
        self._last_used: Dict[str, float] = {}
        self._idle_timeout: Dict[str, Optional[float]] = {}
        self._idle_checker_task: Optional[asyncio.Task] = None

        # Lifecycle state tracking (surfaced via GET /status)
        self._model_state: Dict[str, ModelState] = {}
        self._load_error:  Dict[str, Dict] = {}   # model_id → {"message", "type", "timestamp"}
        self._state_since: Dict[str, float] = {}  # model_id → monotonic() of last transition

    # Session lifecycle 
    async def initialize_model(self, model_id: str, model_config: Dict) -> None:
        """
        Register and optionally load a model.

        If lazy_load: true in model_config, the model is registered but NOT loaded until the first inference request arrives.

        idle_timeout_minutes (optional, models.yaml):
          How many minutes of inactivity before the model is automatically unloaded to free NPU/GPU memory.  
          Set to 0 or omit to disable. The model will lazy-reload on the next request.

          Example:
            idle_timeout_minutes: 30   # unload after 30 min idle
        """
        # Validate required keys
        backend = model_config.get("backend", "").lower()
        if not backend:
            raise ValueError(f"Model '{model_id}': 'backend' is required in config")
        
        # Backend-specific validation
        if backend == "plugin":
            if not model_config.get("plugin_path") and not model_config.get("plugin_module"):
                raise ValueError(
                    f"Model '{model_id}': 'plugin_path' or 'plugin_module' is required for plugin backend"
                )
            if not model_config.get("plugin_class"):
                raise ValueError(f"Model '{model_id}': 'plugin_class' is required for plugin backend")
        
        elif backend == "genie":
            if not model_config.get("genie_config"):
                raise ValueError(f"Model '{model_id}': 'genie_config' is required for genie backend")
        
        elif backend == "onnx_qnn":
            if not model_config.get("model_path"):
                raise ValueError(f"Model '{model_id}': 'model_path' is required for onnx_qnn backend")
        
        elif backend in ("cloud", "openai", "azure", "vllm"):
            if not model_config.get("model_name"):
                raise ValueError(f"Model '{model_id}': 'model_name' is required for cloud backend")

        # Store config (mutable — system_prompt can be updated at runtime)
        self._model_configs[model_id] = dict(model_config)

        # Create per-model lock
        async with self._init_locks_lock:
            if model_id not in self._init_locks:
                self._init_locks[model_id] = asyncio.Lock()

        async with self._infer_locks_lock:
            if model_id not in self._infer_locks:
                self._infer_locks[model_id] = asyncio.Lock()

        # Parse idle timeout
        idle_min = model_config.get("idle_timeout_minutes", 0)
        try:
            idle_min = float(idle_min)
        except (TypeError, ValueError):
            idle_min = 0
        self._idle_timeout[model_id] = idle_min * 60 if idle_min > 0 else None

        # Default: lazy_load=True 
        self._lazy_pending.add(model_id)
        self._model_state[model_id] = ModelState.NOT_LOADED
        self._state_since[model_id] = time.monotonic()
        logger.info(f"[SessionManager] '{model_id}' registered (lazy — loads on first client request or warm-up)")

    async def _load_model(self, model_id: str, model_config: Dict) -> None:
        """Actually create the backend session (called at startup or on first request)."""
        key = self._resolve_backend_key(model_config)
        effective_config = dict(model_config)

        if model_config.get("backend", "").lower() == "genie":
            effective_config.setdefault("plugin_module", "sample_plugins/genie_plugin.py")
            effective_config.setdefault("plugin_class", "GeniePlugin")
            logger.info(
                f"[SessionManager] '{model_id}' backend:genie → Using Genie API's for running"
            )
        
        logger.info(f"[SessionManager] Initializing '{model_id}' → {key}")
        self._model_backend[model_id] = key
        self._model_state[model_id] = ModelState.LOADING
        self._state_since[model_id] = time.monotonic()

        # Only apply locks to on-device models and not cloud
        try:
            if key == "plugin":
                async with self._npu_load_lock:
                    await self._backends[key].create_session(model_id, effective_config)
            else:
                await self._backends[key].create_session(model_id, effective_config)
        except Exception as exc:
            # If load fails, remove the premature entry so the model can be retried.
            self._model_backend.pop(model_id, None)
            self._model_state[model_id] = ModelState.LOAD_FAILED
            self._state_since[model_id] = time.monotonic()
            self._load_error[model_id] = {
                "message": str(exc),
                "type": type(exc).__name__,
                "timestamp": datetime.utcnow().isoformat(),
            }
            # Re-register as lazy so the model can be retried on the next request
            self._lazy_pending.add(model_id)
            safe_inc(MODEL_LOAD_EVENTS_TOTAL, (model_id, "load_failed"))
            raise

        self._model_state[model_id] = ModelState.LOADED
        self._state_since[model_id] = time.monotonic()
        self._load_error.pop(model_id, None)
        safe_inc(MODEL_LOAD_EVENTS_TOTAL, (model_id, "loaded"))
        logger.info(f"[SessionManager] '{model_id}' ready")

    async def _ensure_loaded(self, model_id: str) -> None:
        """Load a lazy model on first use (thread-safe via per-model lock)."""
        if model_id in self._model_backend:
            return  # already loaded

        if model_id not in self._lazy_pending:
            raise RuntimeError(
                f"Model '{model_id}' is not initialized. "
                "Check models.yaml and server startup logs."
            )

        async with self._init_locks[model_id]:
            if model_id not in self._model_backend: 
                logger.info(f"[SessionManager] Lazy-loading '{model_id}'...")
                await self._load_model(model_id, self._model_configs[model_id])
                self._lazy_pending.discard(model_id)

    async def warm_models(self) -> None:
        """Pre-load every lazy_load: false model """

        warm_ids = [mid for mid, cfg in self._model_configs.items() if not cfg.get("lazy_load", True)]
        logger.info(f"[SessionManager] Warm-up starting for {len(warm_ids)} model(s)")
        for model_id in warm_ids:
            try:
                await self._ensure_loaded(model_id)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"[SessionManager] Warm-up failed for '{model_id}': {exc}")

    async def _get_infer_lock(self, model_id: str) -> asyncio.Lock:
        """Return (creating if needed) the per-model inference lock."""
        async with self._infer_locks_lock:
            if model_id not in self._infer_locks:
                self._infer_locks[model_id] = asyncio.Lock()
            return self._infer_locks[model_id]

    async def destroy_model(self, model_id: str, reason: Optional[ModelState] = None) -> None:
        key = self._model_backend.pop(model_id, None)
        if key:
            await self._backends[key].destroy_session(model_id)
        self._lazy_pending.discard(model_id)
        state = reason or ModelState.UNLOADED
        self._model_state[model_id] = state
        self._state_since[model_id] = time.monotonic()
        safe_inc(MODEL_LOAD_EVENTS_TOTAL, (model_id, state.value))

    async def reset_dialog(self, model_id: str) -> None:
        
        await self._ensure_loaded(model_id)
        backend = self._backend_for(model_id)
        reset_fn = getattr(backend, "reset_dialog", None)
        if reset_fn is None:
            logger.debug(
                f"[SessionManager] '{model_id}' backend has no reset_dialog() — no-op"
            )
            return

        lock = await self._get_infer_lock(model_id)
        async with lock:
            await reset_fn(model_id)


    def start_idle_checker(self, interval_seconds: float = 60.0) -> None:
        """Start the background task that auto-unloads idle models."""
        if self._idle_checker_task is not None:
            return  # already running
        self._idle_checker_task = asyncio.ensure_future(
            self._idle_checker_loop(interval_seconds)
        )
        logger.info(f"[SessionManager] Idle checker started (interval={interval_seconds}s)")

    async def _idle_checker_loop(self, interval: float) -> None:
        """Background loop: check for idle models and unload them."""
        while True:
            try:
                await asyncio.sleep(interval)
                # Add timeout to prevent hanging if model unload fails
                await asyncio.wait_for(
                    self._check_idle_models(),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning("[SessionManager] Idle checker timed out")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[SessionManager] Idle checker error: {exc}")

    async def _check_idle_models(self) -> None:
        """Unload any model that has exceeded its idle timeout."""
        now = time.monotonic()
        for model_id, timeout_secs in list(self._idle_timeout.items()):
            if timeout_secs is None:
                continue  # no timeout configured
            if model_id not in self._model_backend:
                continue  # not loaded
            last = self._last_used.get(model_id)
            if last is None:
                continue  # never used — don't unload pre-loaded models
            idle_secs = now - last
            if idle_secs >= timeout_secs:
                logger.info(
                    f"[SessionManager] '{model_id}' idle for {idle_secs/60:.1f} min "
                    f"(timeout={timeout_secs/60:.1f} min) — auto-unloading"
                )
                await self.destroy_model(model_id, reason=ModelState.UNLOADED_IDLE)
                # Re-register as lazy so it reloads on next request
                self._lazy_pending.add(model_id)

    def stop_idle_checker(self) -> None:
        """Cancel the idle checker task (called on server shutdown)."""
        if self._idle_checker_task is not None:
            self._idle_checker_task.cancel()
            self._idle_checker_task = None

    def _touch(self, model_id: str) -> None:
        """Update last-used timestamp for a model (called on every inference)."""
        self._last_used[model_id] = time.monotonic()

    def get_idle_status(self) -> List[Dict]:
        """ Return idle status for all models with a timeout configured."""
        now = time.monotonic()
        result = []
        for model_id, timeout_secs in self._idle_timeout.items():
            if timeout_secs is None:
                continue
            last = self._last_used.get(model_id)
            loaded = model_id in self._model_backend
            result.append({
                "model_id": model_id,
                "loaded": loaded,
                "idle_timeout_minutes": timeout_secs / 60,
                "idle_seconds": round(now - last, 1) if last else None,
                "last_used": "never" if last is None else f"{(now-last)/60:.1f} min ago",
            })
        return result

    def get_model_lifecycle_status(self, model_id: str) -> Dict:
        """Return lifecycle state (+ last load error, if any) for one model."""
        state = self._model_state.get(model_id, ModelState.NOT_LOADED)
        since = self._state_since.get(model_id)
        now = time.monotonic()
        entry = {
            "state": state.value,
            "state_since_seconds": round(now - since, 1) if since else None,
        }
        err = self._load_error.get(model_id)
        if err:
            entry["last_error"] = err

        # Prompt-formatting diagnostics 
        backend_key = self._model_backend.get(model_id)
        backend = self._backends.get(backend_key) if backend_key else None
        get_diag = getattr(backend, "get_prompt_diagnostics", None)
        if callable(get_diag):
            diag = get_diag(model_id)
            if diag:
                entry["prompt_diagnostics"] = diag
        return entry

    # Destroy
    async def destroy_all(self) -> None:
        self.stop_idle_checker()
        for mid in list(self._model_backend.keys()):
            await self.destroy_model(mid)
        logger.info("[SessionManager] All sessions destroyed")

    async def is_alive(self, model_id: str) -> bool:
        key = self._model_backend.get(model_id)
        return bool(key and await self._backends[key].is_session_alive(model_id))

    # For System Prompt management 
    def get_system_prompt(self, model_id: str) -> str:
        """Return the current default system prompt for a model."""
        cfg = self._model_configs.get(model_id, {})
        return cfg.get("system_prompt", "You are a helpful AI assistant.")

    def set_system_prompt(self, model_id: str, system_prompt: str) -> None:
        """
        Update the default system prompt for a model at runtime.
        Takes effect immediately on the next request.
        Does NOT persist to models.yaml — restart to revert.
        """
        if model_id not in self._model_configs:
            raise KeyError(f"Model '{model_id}' not found")
        self._model_configs[model_id]["system_prompt"] = system_prompt
        key = self._model_backend.get(model_id)
        if key:
            backend = self._backends.get(key)
            if backend and callable(getattr(backend, "set_system_prompt", None)):
                backend.set_system_prompt(model_id, system_prompt)

        logger.info(f"[SessionManager] '{model_id}' system_prompt updated at runtime")

    # Cloud client access 
    def get_cloud_client(self, model_id: Optional[str] = None) -> Tuple:
        """
        Return (client, model_name, backend_type) for a cloud model.
        Used by server.py for stateless APIs (files, fine-tuning, batches).
        """
        cloud: CloudBackend = self._backends["cloud"]  # type: ignore
        if model_id is not None:
            if model_id not in cloud._sessions:
                raise RuntimeError(f"Model '{model_id}' is not a cloud model.")
            return cloud._client_for(model_id)
        return cloud.get_any_client()

    # Capability routing
    def _backend_for(self, model_id: str) -> BaseBackend:
        key = self._model_backend.get(model_id)
        if key is None:
            raise RuntimeError(
                f"Model '{model_id}' is not initialized. "
                "Check models.yaml and server startup logs."
            )
        return self._backends[key]

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
        
        await self._ensure_loaded(model_id)
        self._touch(model_id)   # update idle timer

        backend_key = self._model_backend.get(model_id, "")
        is_cloud = (backend_key == "cloud")

        config_system = self._model_configs.get(model_id, {}).get("system_prompt", "")
        has_system = any(m.get("role") == "system" for m in messages)

        if is_cloud:
            if config_system and not has_system:
                resolved_messages = [{"role": "system", "content": config_system}] + list(messages)
            else:
                resolved_messages = list(messages)
        else:
            request_system = None
            non_system = []
            for msg in messages:
                if msg.get("role") == "system" and request_system is None:
                    request_system = msg.get("content", "")
                else:
                    non_system.append(msg)

            final_system = request_system or config_system or "You are a helpful AI assistant."
            resolved_messages = [{"role": "system", "content": final_system}] + non_system

        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                async for token in self._backend_for(model_id).generate(
                    model_id, resolved_messages, temperature, max_tokens, stop,
                    tools=tools, tool_choice=tool_choice, **kwargs
                ):
                    yield token
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "generate", "success"))
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "generate", "error"))
                raise

    async def embed(self, model_id: str, input: List[str], **kwargs) -> List[List[float]]:
        await self._ensure_loaded(model_id)
        self._touch(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).embed(model_id, input, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "embed", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "embed", "error"))
                raise

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
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).transcribe(
                    model_id, audio_bytes, filename, language, prompt,
                    response_format, temperature, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "transcribe", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "transcribe", "error"))
                raise

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
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).translate(model_id, audio_bytes,
                                                                   filename, prompt, response_format, temperature, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "translate", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "translate", "error"))
                raise

    async def synthesize(
        self,
        model_id: str,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).synthesize(
                    model_id, text, voice, response_format, speed, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "synthesize", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "synthesize", "error"))
                raise

    async def moderate(self, model_id: str, input: str, **kwargs) -> Dict:
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).moderate(model_id, input, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "moderate", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "moderate", "error"))
                raise

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
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).image_generate(
                    model_id, prompt, n, size, quality, response_format, style, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_generate", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_generate", "error"))
                raise
        
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
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).image_edit(
                    model_id, image, prompt, mask=mask, n=n, size=size,
                    response_format=response_format, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_edit", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_edit", "error"))
                raise

    async def image_variation(
        self,
        model_id: str,
        image: bytes,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs,
    ) -> List[Dict]:
        await self._ensure_loaded(model_id)
        backend_key = self._model_backend.get(model_id, "")
        lock = await self._get_infer_lock(model_id)
        async with lock:
            try:
                result = await self._backend_for(model_id).image_variation(
                    model_id, image, n=n, size=size,
                    response_format=response_format, **kwargs)
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_variation", "success"))
                return result
            except Exception:
                safe_inc(MODEL_INFERENCE_TOTAL, (model_id, backend_key, "image_variation", "error"))
                raise

    # Token counting 
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        return _count_tokens(text, model)

    @staticmethod
    def count_messages_tokens(messages: List[Dict], model: str = "gpt-4o") -> int:
        return _count_messages_tokens(messages, model)

    # Helpers methods
    @staticmethod
    def _resolve_backend_key(model_config: Dict) -> str:
        backend = model_config.get("backend", "").lower()
        if backend in ("plugin", "genie"):  return "plugin" 
        if backend == "onnx_qnn": return "onnx_qnn"
        return "cloud"