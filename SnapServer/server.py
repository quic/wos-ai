# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
SnapServer: Snapdragon X Series AI Model Server
===============================================
Full OpenAI API surface — 30 endpoints.

Run:
  python server.py [--host 0.0.0.0] [--port 8000] [--config config/models.yaml]

Auth (optional):
  Set SERVER_API_KEY env var to require Bearer token on all /v1/* requests.
  Leave unset to allow unauthenticated access (dev / local mode).
"""

import asyncio
import json
import hmac
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from core.config_manager import ConfigManager
from core.model_registry import ModelRegistry
from core.session_manager import SessionManager
from core.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_LATENCY_SECONDS,
    MODEL_INFERENCE_TOTAL,
    MODEL_LOAD_EVENTS_TOTAL,
    MODEL_TOKENS_TOTAL,
    safe_inc,
    safe_observe,
)
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import fastapi.routing as _fastapi_routing
    if hasattr(_fastapi_routing, "_extract_endpoint_context"):
        def _noop_extract_endpoint_context(endpoint):  # type: ignore[override]
            return {}
        _fastapi_routing._extract_endpoint_context = _noop_extract_endpoint_context
except Exception:
    pass

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

#  Globals (populated in lifespan)
config_manager: Optional[ConfigManager] = None
model_registry: Optional[ModelRegistry] = None
session_manager: Optional[SessionManager] = None
_models_filter: Optional[List[str]] = None   # None = load all; list = load only these
_config_path: str = os.environ.get("MODEL_CONFIG", "config/models.yaml")


# Optional API key auth — set SERVER_API_KEY env var to enable
_SERVER_API_KEY: Optional[str] = os.environ.get("SERVER_API_KEY")


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config_manager, model_registry, session_manager
    logger.info("=== Server starting ===")
    config_manager = ConfigManager(config_path=_config_path)
    await config_manager.load_config()
    model_registry = ModelRegistry(config_manager)
    await model_registry.initialize()
    session_manager = SessionManager()
    models = await model_registry.list_models()
    loaded = 0
    for model in models:
        if _models_filter and model["id"] not in _models_filter:
            logger.info(f"Skipping '{model['id']}' (not in --models filter)")
            continue
        try:
            await session_manager.initialize_model(model["id"], model)
            loaded += 1
        except Exception as exc:
            logger.error(f"Failed to initialize model '{model['id']}': {exc}")
    logger.info(f"=== Server ready — {loaded}/{len(models)} models loaded registered===")
    # Start idle timeout checker (auto-unloads models after idle_timeout_minutes)
    session_manager.start_idle_checker(interval_seconds=60.0)
    asyncio.ensure_future(session_manager.warm_models())
    yield
    logger.info("=== Server shutting down ===")
    if session_manager:
        await session_manager.destroy_all()


# Swagger UI App 

app = FastAPI(
    title="SnapServer",
    description="Snapdragon X Series AI Model Server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: Request ID + latency logging

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach X-Request-ID to every request/response and log latency."""
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = req_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = req_id
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} [{elapsed_ms:.1f}ms] req={req_id}"
    )
    route = request.scope.get("route")
    path_label = route.path if route else request.url.path
    safe_inc(HTTP_REQUESTS_TOTAL, (request.method, path_label, str(response.status_code)))
    safe_observe(HTTP_REQUEST_LATENCY_SECONDS, (request.method, path_label), elapsed_ms / 1000.0)
    return response


# Middleware: Optional API key auth

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    If SERVER_API_KEY is set, require 'Authorization: Bearer <key>' on all
    /v1/* routes.  /health, /status, /docs are always public.
    """
    if _SERVER_API_KEY and request.url.path.startswith("/v1"):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], _SERVER_API_KEY):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key",
                                   "type": "authentication_error"}},
            )
    return await call_next(request)


# Global exception handlers

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return clean JSON error — never expose raw stack traces."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error",
                            "code": type(exc).__name__}},
    )

@app.exception_handler(NotImplementedError)
async def not_implemented_handler(request: Request, exc: NotImplementedError):
    return JSONResponse(
        status_code=501,
        content={"error": {"message": str(exc), "type": "not_implemented_error"}},
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Model not loaded / not initialized → 503 instead of 500."""
    msg = str(exc)
    if "not initialized" in msg or "not loaded" in msg:
        logger.warning(f"Model unavailable on {request.url.path}: {msg}")
        return JSONResponse(
            status_code=503,
            content={"error": {"message": msg, "type": "model_unavailable_error"}},
        )
    logger.error(f"RuntimeError on {request.url.path}: {msg}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": msg, "type": "server_error",
                            "code": "RuntimeError"}},
    )


if _OPENAI_AVAILABLE:
    # Map openai SDK error types to their correct HTTP status codes
    _OPENAI_STATUS_MAP = {
        "AuthenticationError":   401,
        "PermissionDeniedError": 403,
        "NotFoundError":         404,
        "UnprocessableEntityError": 422,
        "RateLimitError":        429,
        "InternalServerError":   502,  # upstream error, not our fault → 502
        "APIConnectionError":    503,
        "APITimeoutError":       504,
    }

    @app.exception_handler(_openai_module.APIError)
    async def openai_error_handler(request: Request, exc: _openai_module.APIError):
        exc_type = type(exc).__name__
        status = _OPENAI_STATUS_MAP.get(exc_type, 502)
        msg = str(exc)
        logger.warning(f"OpenAI API error ({exc_type}) on {request.url.path}: {msg}")
        return JSONResponse(
            status_code=status,
            content={"error": {"message": msg, "type": "cloud_api_error",
                                "code": exc_type}},
        )


# Pydantic request models 
class Message(BaseModel):
    role: str
    content: Union[str, List, None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List] = None

    model_config = ConfigDict(extra="ignore")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    max_completion_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(0, ge=-2, le=2)
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[Dict] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(256, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ModerationRequest(BaseModel):
    model: Optional[str] = "omni-moderation-latest"
    input: Union[str, List[str]]


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = "dall-e-3"
    prompt: str
    n: Optional[int] = Field(1, ge=1, le=10)
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    style: Optional[str] = None
    user: Optional[str] = None


class FineTuningRequest(BaseModel):
    training_file: str
    model: str
    validation_file: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    suffix: Optional[str] = None
    seed: Optional[int] = None


class BatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: Optional[Dict] = None


# Shared helpers

def _stop_list(stop) -> Optional[List[str]]:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else stop


def _sse_chunk(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _sse_token_stream(token_gen, model: str, response_object: str, chunk_id: str, 
                            is_chat: bool, prompt_tokens: int = 0, req_id: str = ""):
    """Single SSE implementation shared by chat + legacy completions."""
    start = time.perf_counter()
    completion_tokens = 0
    try:
        async for token in token_gen:
            if is_chat and token.startswith("\x00TOOL_CALLS\x00"):
                envelope = json.loads(token[len("\x00TOOL_CALLS\x00"):])
                yield _sse_chunk({"id": chunk_id, "object": response_object,
                                   "created": int(time.time()), "model": model,
                                   "choices": [{"index": 0,
                                                "delta": {"role": "assistant",
                                                          "content": envelope.get("content"),
                                                          "tool_calls": envelope.get("tool_calls")},
                                                "finish_reason": "tool_calls"}]})
                yield "data: [DONE]\n\n"
                # tool_calls response has no completion text tokens to count
                if prompt_tokens:
                    safe_inc(MODEL_TOKENS_TOTAL, (model, "prompt"), prompt_tokens)
                logger.info(
                    f"SSE stream complete model={model} tool_calls=1 "
                    f"total={(time.perf_counter() - start) * 1000:.1f}ms req={req_id}"
                )
                return
            completion_tokens += SessionManager.count_tokens(token, model)
            choice = (
                {"index": 0, "delta": {"content": token}, "finish_reason": None}
                if is_chat
                else {"index": 0, "text": token, "finish_reason": None}
            )
            yield _sse_chunk({"id": chunk_id, "object": response_object,
                               "created": int(time.time()), "model": model,
                               "choices": [choice]})
        final = (
            {"index": 0, "delta": {}, "finish_reason": "stop"}
            if is_chat
            else {"index": 0, "text": "", "finish_reason": "stop"}
        )
        yield _sse_chunk({"id": chunk_id, "object": response_object,
                           "created": int(time.time()), "model": model,
                           "choices": [final]})
        yield "data: [DONE]\n\n"
        if prompt_tokens:
            safe_inc(MODEL_TOKENS_TOTAL, (model, "prompt"), prompt_tokens)
        if completion_tokens:
            safe_inc(MODEL_TOKENS_TOTAL, (model, "completion"), completion_tokens)
        logger.info(
            f"SSE stream complete model={model} completion_tokens={completion_tokens} "
            f"total={(time.perf_counter() - start) * 1000:.1f}ms req={req_id}"
        )
    except Exception as exc:
        logger.error(
            f"SSE stream error model={model} "
            f"total={(time.perf_counter() - start) * 1000:.1f}ms req={req_id}: {exc}"
        )
        yield _sse_chunk({"error": {"message": str(exc), "type": "server_error"}})


def _usage(prompt_msgs: List[Message], completion: str, model: str) -> Dict:
    """Build usage dict with accurate token counts (tiktoken when available)."""
    parts = []
    for m in prompt_msgs:
        if isinstance(m.content, str):
            parts.append(m.content)
        elif isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    prompt_text = " ".join(parts)
    pt = SessionManager.count_tokens(prompt_text, model)
    ct = SessionManager.count_tokens(completion, model)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


async def _require_model(model_id: str) -> None:
    """Raise 404 if model is not registered."""
    if not await model_registry.get_model(model_id):
        raise HTTPException(404, detail=f"Model '{model_id}' not found")


def _cloud_client(model_id: str):
    """Get cloud client for a specific model (via session_manager — no private access)."""
    try:
        return session_manager.get_cloud_client(model_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


def _any_cloud_client():
    """Get cloud client for stateless APIs (files, fine-tuning, batches)."""
    try:
        return session_manager.get_cloud_client()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


# Root

@app.get("/")
async def root():
    return {"message": "SnapServer v1", "docs": "/docs"}


# Models API's

@app.get("/v1/models")
async def list_models():
    models = await model_registry.list_models()
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "created": int(time.time()),
         "owned_by": m.get("owned_by", "local")} for m in models
    ]}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    model = await model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"id": model["id"], "object": "model", "created": int(time.time()),
            "owned_by": model.get("owned_by", "local")}


@app.delete("/v1/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a fine-tuned model (cloud pass-through)."""
    client, _, _ = _cloud_client(model_id)
    await client.models.delete(model_id)
    return {"id": model_id, "object": "model", "deleted": True}


# Model lifecycle management (client-accessible) 
@app.post("/v1/models/{model_id}/load")
async def load_model(model_id: str):
    """
    Useful for lazy-loaded models — the client can explicitly trigger loading before the first inference request to avoid the first-request stall.

    Returns 200 if already loaded, 201 if newly loaded.
    """
    model = await model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found in models.yaml")

    already_alive = await session_manager.is_alive(model_id)
    if already_alive:
        return JSONResponse(
            status_code=200,
            content={"model_id": model_id, "status": "already_loaded"},
        )

    try:
        await session_manager._ensure_loaded(model_id)
        return JSONResponse(
            status_code=201,
            content={"model_id": model_id, "status": "loaded"},
        )
    except Exception as exc:
        logger.error(f"Failed to load '{model_id}': {exc}")
        raise HTTPException(500, f"Failed to load model '{model_id}': {exc}")


@app.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model and free all its resources (NPU memory, ONNX session, etc.) without killing the server."""
    model = await model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found in models.yaml")

    if not await session_manager.is_alive(model_id):
        return {"model_id": model_id, "status": "not_loaded"}

    await session_manager.destroy_model(model_id)
    return {"model_id": model_id, "status": "unloaded"}


@app.post("/v1/models/{model_id}/reload")
async def reload_model(model_id: str):
    """Unload and reload a model without restarting the server."""
    model = await model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found in models.yaml")

    # Unload if currently loaded
    if await session_manager.is_alive(model_id):
        await session_manager.destroy_model(model_id)
        logger.info(f"Reload: '{model_id}' unloaded")

    # Reload config from disk (picks up any models.yaml changes)
    await config_manager.load_config()
    await model_registry.initialize()
    fresh_model = await model_registry.get_model(model_id)
    if not fresh_model:
        raise HTTPException(404, f"Model '{model_id}' not found after config reload")

    try:
        await session_manager.initialize_model(model_id, fresh_model)
        await session_manager._ensure_loaded(model_id)
        return {"model_id": model_id, "status": "reloaded"}
    except Exception as exc:
        logger.error(f"Failed to reload '{model_id}': {exc}")
        raise HTTPException(500, f"Failed to reload model '{model_id}': {exc}")


@app.post("/v1/models/{model_id}/reset_dialog")
async def reset_dialog(model_id: str):
    """ Reset the KV cache for a local model without unloading it."""
    model = await model_registry.get_model(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found in models.yaml")

    try:
        await session_manager.reset_dialog(model_id)
        return {"model_id": model_id, "status": "dialog_reset"}
    except RuntimeError as exc:
        # Model not loaded yet — that's fine, nothing to reset
        logger.info(f"reset_dialog: '{model_id}' not loaded, skipping: {exc}")
        return {"model_id": model_id, "status": "not_loaded_no_op"}
    except Exception as exc:
        logger.error(f"reset_dialog failed for '{model_id}': {exc}")
        raise HTTPException(500, f"Failed to reset dialog for '{model_id}': {exc}")


# API"s For system prompt management

class SystemPromptUpdate(BaseModel):
    system_prompt: str


@app.get("/v1/models/{model_id}/system_prompt")
async def get_system_prompt(model_id: str):
    await _require_model(model_id)
    try:
        prompt = session_manager.get_system_prompt(model_id)
        return {"model_id": model_id, "system_prompt": prompt}
    except (KeyError, AttributeError):
        raise HTTPException(404, f"Model '{model_id}' has no system prompt config")


@app.patch("/v1/models/{model_id}/system_prompt")
async def update_system_prompt(model_id: str, body: SystemPromptUpdate):
    await _require_model(model_id)
    try:
        session_manager.set_system_prompt(model_id, body.system_prompt)
        return {
            "model_id": model_id,
            "system_prompt": body.system_prompt,
            "note": "Updated in memory. Restart server to revert to models.yaml value.",
        }
    except KeyError:
        raise HTTPException(404, f"Model '{model_id}' not found in session manager")


# API's for Chat completions 

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    await _require_model(request.model)
    msgs = [m.model_dump() for m in request.messages]
    effective_max_tokens = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
    token_gen = session_manager.generate(
        model_id=request.model, messages=msgs,
        temperature=request.temperature,
        max_tokens=effective_max_tokens,
        stop=_stop_list(request.stop),
        tools=request.tools or None,
        tool_choice=request.tool_choice,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        seed=request.seed,
        response_format=request.response_format,
    )
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if request.stream:
        prompt_tokens = SessionManager.count_messages_tokens(msgs, request.model)
        return StreamingResponse(
            _sse_token_stream(token_gen, request.model, "chat.completion.chunk", chunk_id, is_chat=True, 
prompt_tokens=prompt_tokens, req_id=getattr(http_request.state, "request_id", "")),
            media_type="text/event-stream",
        )
    parts = []
    tool_envelope_json = None
    try:
        async for token in token_gen:
            if token.startswith("\x00TOOL_CALLS\x00"):
                tool_envelope_json = token[len("\x00TOOL_CALLS\x00"):]
            else:
                parts.append(token)
    except Exception as exc:
        # Client may have disconnected/timed out while non-streaming calls
        # unhandled — see the WinError 10054 note in plugin_backend.py.
        logger.error(f"chat_completions: generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    content = "".join(parts)
    usage = _usage(request.messages, content, request.model)
    safe_inc(MODEL_TOKENS_TOTAL, (request.model, "prompt"), usage["prompt_tokens"])
    safe_inc(MODEL_TOKENS_TOTAL, (request.model, "completion"), usage["completion_tokens"])
    if tool_envelope_json:
        envelope = json.loads(tool_envelope_json)
        message = {"role": "assistant", "content": envelope.get("content"),
                   "tool_calls": envelope.get("tool_calls")}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": content}
        finish_reason = "stop"
    return {
        "id": chunk_id, "object": "chat.completion",
        "created": int(time.time()), "model": request.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


# Legacy API's completions 
@app.post("/v1/completions")
async def completions(request: CompletionRequest, http_request: Request):
    await _require_model(request.model)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    msgs = [{"role": "user", "content": prompt}]
    token_gen = session_manager.generate(
        model_id=request.model, messages=msgs,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=_stop_list(request.stop),
    )
    chunk_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    if request.stream:
        prompt_tokens = SessionManager.count_tokens(prompt, request.model)
        return StreamingResponse(
            _sse_token_stream(token_gen, request.model,
                              "text_completion", chunk_id, is_chat=False,
                              prompt_tokens=prompt_tokens,
                              req_id=getattr(http_request.state, "request_id", "")),
            media_type="text/event-stream",
        )
    parts = []
    try:
        async for token in token_gen:
            parts.append(token)
    except Exception as exc:
        # Same rationale as chat_completions() — see the WinError 10054 note
        # in plugin_backend.py.
        logger.error(f"completions: generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    text = "".join(parts)
    pt = SessionManager.count_tokens(prompt, request.model)
    ct = SessionManager.count_tokens(text, request.model)
    safe_inc(MODEL_TOKENS_TOTAL, (request.model, "prompt"), pt)
    safe_inc(MODEL_TOKENS_TOTAL, (request.model, "completion"), ct)
    return {
        "id": chunk_id, "object": "text_completion",
        "created": int(time.time()), "model": request.model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


# API's for Embeddings 
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    await _require_model(request.model)
    inputs = [request.input] if isinstance(request.input, str) else request.input
    vectors = await session_manager.embed(model_id=request.model, input=inputs)
    total_tokens = sum(SessionManager.count_tokens(s, request.model) for s in inputs)
    safe_inc(MODEL_TOKENS_TOTAL, (request.model, "prompt"), total_tokens)
    return {
        "object": "list", "model": request.model,
        "data": [{"object": "embedding", "index": i, "embedding": vec}
                 for i, vec in enumerate(vectors)],
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


# API"s for Audio transcription 
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    await _require_model(model)
    audio_bytes = await file.read()
    result = await session_manager.transcribe(
        model_id=model, audio_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        language=language, prompt=prompt,
        response_format=response_format, temperature=temperature,
    )
    if response_format == "text":
        return Response(content=result.get("text", ""), media_type="text/plain")
    return result

@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """Translate audio to English text (Whisper translation mode)."""
    await _require_model(model)
    audio_bytes = await file.read()
    result = await session_manager.translate(
        model_id=model, audio_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        prompt=prompt, response_format=response_format, temperature=temperature,
    )
    if response_format == "text":
        return Response(content=result.get("text", ""), media_type="text/plain")
    return result


@app.post("/v1/audio/speech")
async def audio_speech(
    model: str = Form("tts-1"),
    input: str = Form(...),
    voice: str = Form("alloy"),
    response_format: str = Form("mp3"),
    speed: float = Form(1.0),
):
    await _require_model(model)
    audio_bytes = await session_manager.synthesize(
        model_id=model, text=input, voice=voice,
        response_format=response_format, speed=speed,
    )
    content_types = {
        "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
        "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
    }
    return Response(content=audio_bytes,
                    media_type=content_types.get(response_format, "audio/mpeg"))


# API's for Images
@app.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest):
    model_id = request.model or "dall-e-3"
    await _require_model(model_id)
    images = await session_manager.image_generate(
        model_id=model_id, prompt=request.prompt, n=request.n,
        size=request.size, quality=request.quality,
        response_format=request.response_format, style=request.style,
    )
    return {"created": int(time.time()), "data": images}


@app.post("/v1/images/edits")
async def images_edits(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: str = Form("dall-e-2"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
    user: Optional[str] = Form(None),
):
    """Edit an image — routes to local plugin or cloud depending on registered model."""
    await _require_model(model)
    image_bytes = await image.read()
    mask_bytes  = await mask.read() if mask else None
    images = await session_manager.image_edit(
        model_id=model, image=image_bytes, prompt=prompt,
        mask=mask_bytes, n=n, size=size, response_format=response_format,
    )
    return {"created": int(time.time()), "data": images}


@app.post("/v1/images/variations")
async def images_variations(
    image: UploadFile = File(...),
    model: str = Form("dall-e-2"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
    user: Optional[str] = Form(None),
):
    """Create image variations — routes to local plugin or cloud depending on registered model."""
    await _require_model(model)
    image_bytes = await image.read()
    images = await session_manager.image_variation(
        model_id=model, image=image_bytes,
        n=n, size=size, response_format=response_format,
    )
    return {"created": int(time.time()), "data": images}


# API's for Moderations 

@app.post("/v1/moderations")
async def moderations(request: ModerationRequest):
    model_id = request.model or "omni-moderation-latest"
    await _require_model(model_id)
    inputs = request.input if isinstance(request.input, list) else [request.input]
    results = []
    for text in inputs:
        r = await session_manager.moderate(model_id=model_id, input=text)
        results.extend(r.get("results", []))
    return {"id": f"modr-{uuid.uuid4().hex[:12]}", "model": model_id, "results": results}


# API's for Files 
@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """Upload a file (cloud pass-through)."""
    import io as _io
    client, _, _ = _any_cloud_client()
    file_bytes = await file.read()
    result = await client.files.create(
        file=(file.filename or "file.jsonl", _io.BytesIO(file_bytes), "application/octet-stream"),
        purpose=purpose,
    )
    return result.model_dump()


@app.get("/v1/files")
async def list_files(purpose: Optional[str] = None):
    """List uploaded files (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = {}
    if purpose:
        params["purpose"] = purpose
    result = await client.files.list(**params)
    return {"object": "list", "data": [f.model_dump() for f in result.data]}


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    """Retrieve file metadata (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    result = await client.files.retrieve(file_id)
    return result.model_dump()


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    await client.files.delete(file_id)
    return {"id": file_id, "object": "file", "deleted": True}


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    """Retrieve file content (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    content = await client.files.content(file_id)
    return Response(content=content.content, media_type="application/octet-stream")


# API's for Fine-tuning 
@app.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job(request: FineTuningRequest):
    """Create a fine-tuning job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = dict(training_file=request.training_file, model=request.model)
    if request.validation_file:
        params["validation_file"] = request.validation_file
    if request.hyperparameters:
        params["hyperparameters"] = request.hyperparameters
    if request.suffix:
        params["suffix"] = request.suffix
    if request.seed is not None:
        params["seed"] = request.seed
    result = await client.fine_tuning.jobs.create(**params)
    return result.model_dump()


@app.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs(after: Optional[str] = None, limit: int = 20):
    """List fine-tuning jobs (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = {"limit": limit}
    if after:
        params["after"] = after
    result = await client.fine_tuning.jobs.list(**params)
    return {"object": "list", "data": [j.model_dump() for j in result.data]}


@app.get("/v1/fine_tuning/jobs/{job_id}")
async def retrieve_fine_tuning_job(job_id: str):
    """Retrieve a fine-tuning job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    result = await client.fine_tuning.jobs.retrieve(job_id)
    return result.model_dump()


@app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(job_id: str):
    """Cancel a fine-tuning job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    result = await client.fine_tuning.jobs.cancel(job_id)
    return result.model_dump()


@app.get("/v1/fine_tuning/jobs/{job_id}/events")
async def list_fine_tuning_events(job_id: str, after: Optional[str] = None, limit: int = 20):
    """List fine-tuning job events (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = {"limit": limit}
    if after:
        params["after"] = after
    result = await client.fine_tuning.jobs.list_events(job_id, **params)
    return {"object": "list", "data": [e.model_dump() for e in result.data]}


@app.get("/v1/fine_tuning/jobs/{job_id}/checkpoints")
async def list_fine_tuning_checkpoints(job_id: str, after: Optional[str] = None, limit: int = 10):
    """List fine-tuning checkpoints (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = {"limit": limit}
    if after:
        params["after"] = after
    result = await client.fine_tuning.jobs.checkpoints.list(job_id, **params)
    return {"object": "list", "data": [c.model_dump() for c in result.data]}


# API's for Batch
@app.post("/v1/batches")
async def create_batch(request: BatchRequest):
    """Create a batch job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = dict(input_file_id=request.input_file_id,
                  endpoint=request.endpoint,
                  completion_window=request.completion_window)
    if request.metadata:
        params["metadata"] = request.metadata
    result = await client.batches.create(**params)
    return result.model_dump()


@app.get("/v1/batches")
async def list_batches(after: Optional[str] = None, limit: int = 20):
    """List batch jobs (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    params = {"limit": limit}
    if after:
        params["after"] = after
    result = await client.batches.list(**params)
    return {"object": "list", "data": [b.model_dump() for b in result.data]}


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    """Retrieve a batch job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    result = await client.batches.retrieve(batch_id)
    return result.model_dump()


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """Cancel a batch job (cloud pass-through)."""
    client, _, _ = _any_cloud_client()
    result = await client.batches.cancel(batch_id)
    return result.model_dump()


# API's for Health / Status

@app.get("/health")
async def health():
    models = await model_registry.list_models()
    alive = [m["id"] for m in models if await session_manager.is_alive(m["id"])]
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(alive),
        "models_alive": alive,
    }


@app.get("/status")
async def status():
    models = await model_registry.list_models()
    idle_info = {s["model_id"]: s for s in session_manager.get_idle_status()}
    model_statuses = []
    for m in models:
        mid = m["id"]
        entry = {
            "id": mid,
            "backend": m.get("backend", "unknown"),
            "alive": await session_manager.is_alive(mid),
        }
        entry.update(session_manager.get_model_lifecycle_status(mid))
        if mid in idle_info:
            entry["idle_timeout_minutes"] = idle_info[mid]["idle_timeout_minutes"]
            entry["last_used"] = idle_info[mid]["last_used"]
            entry["idle_seconds"] = idle_info[mid]["idle_seconds"]
        model_statuses.append(entry)
    return {
        "server": "SnapServer",
        "version": "1.0.0",
        "auth_enabled": bool(_SERVER_API_KEY),
        "models": model_statuses,
    }


@app.get("/metrics")
async def metrics_json():
    """JSON metrics snapshot — same counters as /metrics/prometheus, no new deps needed to read."""

    def _samples(metric):
        return [
            {"name": s.name, "labels": s.labels, "value": s.value}
            for fam in metric.collect() for s in fam.samples
        ]

    return {
        "http_requests_total": _samples(HTTP_REQUESTS_TOTAL),
        "http_request_latency_seconds": _samples(HTTP_REQUEST_LATENCY_SECONDS),
        "model_inference_total": _samples(MODEL_INFERENCE_TOTAL),
        "model_tokens_total": _samples(MODEL_TOKENS_TOTAL),
        "model_load_events_total": _samples(MODEL_LOAD_EVENTS_TOTAL),
    }


@app.get("/metrics/prometheus")
async def metrics_prometheus():
    """Prometheus text-exposition format — scrape target for Prometheus/Grafana."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Entry point/ Main method
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SnapServer: Snapdragon X Series AI Model Server v1")
    parser.add_argument("--host",    default="localhost", help="Bind host")
    parser.add_argument("--port",    type=int, default=8000, help="Bind port")
    parser.add_argument("--config",  default="config/models.yaml", help="Path to models.yaml")
    parser.add_argument("--models",  default=None,
                        help="Comma-separated model IDs to load (default: load all). "
                             "Example: --models my-genie-model,gpt-4o")
    parser.add_argument("--reload",  action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    os.environ["MODEL_CONFIG"] = args.config
    _config_path = args.config

    logger.info(f"Config: {args.config}  Port: {args.port}")

    if args.models:
        _models_filter = [m.strip() for m in args.models.split(",") if m.strip()]
        logger.info(f"Model filter: {_models_filter}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
        access_log=False,
    )