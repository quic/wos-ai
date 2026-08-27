# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Architecture

## Stack

```
Client (OpenAI SDK / curl / Gradio UI)
  │  HTTP / SSE
  ▼
FastAPI  server.py  (uvicorn)
  │
  ├── ConfigManager      core/config_manager.py   — loads models.yaml
  ├── ModelRegistry      core/model_registry.py   — maps model IDs to config dicts
  └── SessionManager     core/session_manager.py  — lifecycle + routing
        │
        ├── PluginBackend   backends/plugin_backend.py
        │     └── genie_plugin.py          ← text generation (genie-t2t-run)
        │     └── sample_asr_plugin.py     ← sample audio transcription file for custom integration
        │     └── (any custom plugin)
        │           └── genie_wrapper.py   ← ctypes, single source of truth
        │                 └── Genie.dll    ← GenieDialog
        │                       └── QNN HTP (NPU)
        │
        └── CloudBackend    backends/cloud_backend.py
              └── openai.AsyncOpenAI → cloud API
```

## Request Lifecycle

```
Client Request
  |
  v  [Middleware: CORS -> Request-ID -> Auth]
  |
  v  [FastAPI Route Handler]  -- 404 if model not registered
  |
  v  [SessionManager]
  |
  +-- Cloud model? --> async HTTP to OpenAI/Azure/vLLM  (fully concurrent)
  |
  +-- Local model? --> asyncio.Lock --> ThreadPoolExecutor
                                              |
                                    [Backend.generate()]
                                    runs in thread pool
                                              |
                                    Streaming?  -> queue.Queue -> SSE chunks
                                    Non-stream? -> collect all -> JSON
```

---

## Two-Phase Model Lifecycle

```
STARTUP  — create_session() called ONCE per model
  Load weights | Allocate NPU memory | Create runtime context | Register Lock

PER REQUEST  — generate() / transcribe() / embed()
  Acquire Lock | Run in ThreadPool | Stream via queue.Queue | Release Lock

SHUTDOWN  — destroy_session() called ONCE per model
  plugin.unload() | Free NPU memory | Close runtime context
```

---

## Concurrency Model

```
Cloud requests  -----------------------------------------> async HTTP (concurrent)
Local request 1 --> asyncio.Lock --> ThreadPool --> inference --> SSE
Local request 2 --> Wait for Lock --------------------------------> (queued)
```

Local models are serialized per-model (Genie dialog is not thread-safe).
Different models run concurrently.

---

## Genie Streaming Data Flow (LLM's - text generation)

```
[C++ NPU decode loop]  (in ThreadPoolExecutor thread)
  │  fires token callback synchronously from within dialog.query()
  ▼
[genie_wrapper._raw_cb()]
  │  response_ptr.decode("utf-8")   ~3–7 µs/token
  │  calls genie_plugin._callback()
  ▼
[on_token() → loop.call_soon_threadsafe(async_queue.put_nowait, token)]
  ▼
[asyncio event loop → await async_queue.get() → yield token]
  ▼
[server.py _sse_token_stream() → json.dumps → yield SSE chunk]
  ▼
[HTTP SSE stream → client]
```

## Module Responsibilities

| Module | Role |
|--------|------|
| `server.py` | FastAPI app, all endpoints, Pydantic models, SSE, middleware |
| `core/config_manager.py` | Parse `models.yaml`, expand `${ENV_VAR}` |
| `core/model_registry.py` | In-memory model lookup by ID |
| `core/session_manager.py` | Backend routing, Lock, ThreadPool, token counting, idle timeout, lifecycle state |
| `core/metrics.py` | Prometheus Counter/Histogram definitions shared by `/metrics` and `/metrics/prometheus` |
| `backends/base_backend.py` | Abstract interface — all backends implement this |
| `backends/genie_wrapper.py` | ctypes bindings to `Genie.dll` / `libGenie.so` (lazy init) |
| `backends/plugin_backend.py` | Loads user plugin via `importlib`, `queue.Queue` SSE streaming |
| `backends/onnx_qnn_backend.py` | ONNX Runtime + QNN EP — embeddings, ASR |
| `backends/cloud_backend.py` | Async HTTP proxy, cached `AsyncOpenAI` clients |
| `utils/inference_plugin.py` | `InferencePlugin` ABC — user contract |
| `utils/prompt_formatter.py` | HF Jinja2 chat templates with generic fallback |