# Release Notes

## v1.0.0 — Initial Release

SnapServer : a FastAPI server that exposes a full OpenAI-compatible REST API and routes requests to local Qualcomm Snapdragon NPU inference or cloud providers, behind a single drop-in `base_url`.

### Highlights

- **Drop-in OpenAI compatibility** — any client using the OpenAI Python SDK, LangChain, or plain `curl` works unchanged by pointing `base_url` at `http://localhost:8000/v1`.
- **Three inference backends behind one API**:
  - `plugin` — wrap any existing Python inference code (including the Qualcomm Genie SDK for LLMs on NPU/HTP) in ~30 lines via the `InferencePlugin` contract.
  - `onnx_qnn` — ONNX Runtime with the QNN Execution Provider, for embeddings and CNN-style models on NPU.
  - `openai` / `azure` / `vllm` — async pass-through to cloud or self-hosted OpenAI-compatible endpoints.
- **Plugin system** as the primary extension point — implement `load()` + `generate()` against `utils/inference_plugin.py`, with optional hooks for multi-turn KV-cache generation (`generate_with_messages`), lowest-TTFT token callbacks (`generate_with_messages_cb`), embeddings, transcription/translation, TTS, moderation, and image generation/edit/variation. Ships with ready-to-use templates in `sample_plugins/` (Genie, ONNX+QNN, C++ shared library, Whisper ASR, minimal template) and a `VenvPlugin` base class for plugins that need an isolated virtual environment.
- **Full OpenAI API surface** (~38 endpoints): chat/text completions (streaming + non-streaming), embeddings, audio transcription/translation/speech, image generation/edits/variations, moderations, and cloud pass-through for files, fine-tuning jobs, and batches.
- **Model lifecycle management** — lazy or eager loading, on-demand `load`/`unload`/`reload`, idle-timeout auto-unload, and `reset_dialog` to clear KV cache in milliseconds without reloading weights. Runtime-updatable system prompt via `GET`/`PATCH /v1/models/{id}/system_prompt`.
- **Concurrency model** — cloud requests run fully async and concurrently; local NPU models are serialized per-model via `asyncio.Lock` + `ThreadPoolExecutor` (Genie dialogs are not thread-safe), while different models run concurrently with each other.
- **Multi-instance / multi-port support** — run several `server.py` processes, each with its own `models.yaml` and port, to separate local and cloud models or isolate model sets.
- **Observability** — `/health` and `/status` for liveness and per-model state, plus `/metrics` (JSON) and `/metrics/prometheus` (Prometheus text exposition) covering HTTP request counts/latency, per-model inference counts, token counts, and load/unload events.
- **Optional Bearer-token auth** — set `SERVER_API_KEY` to require a token on all `/v1/*` routes; `/health`, `/status`, `/docs`, and `/metrics/prometheus` stay public.
- **One-click environment setup** — `setup_env.bat` / `setup_env.ps1` / `setup_env.sh` auto-detect Python, create a `.venv`, and install dependencies.
- **Sample Streamlit UIs** (`sample_ui/chat_ui.py`, `sample_ui/asr_ui.py`) for manual testing of chat and ASR models.

### Known limitations

- Local model inference is single-threaded per model (by design, due to NPU/Genie dialog constraints); throughput scales by running multiple model instances, not concurrent requests to one model.

### Documentation

- [README.md](README.md) — quick start and repository layout
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, request lifecycle, concurrency, streaming data flow
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — full `models.yaml` reference
- [docs/API.md](docs/API.md) — endpoint reference with examples
- [docs/PLUGINS.md](docs/PLUGINS.md) — plugin contract and sample plugin index


<!-- Footer -->
<div style="position: fixed; bottom: 0; width: 100%; text-align: center; font-size: small; color: gray; opacity: 0.5;">
  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. <br>
  SPDX-License-Identifier: BSD-3-Clause-Clear
</div>