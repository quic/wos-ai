# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Model Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List all registered models |
| GET | `/v1/models/{id}` | Get model details |
| DELETE | `/v1/models/{id}` | Delete fine-tuned model (cloud pass-through) |
| POST | `/v1/models/{id}/load` | Load model on demand |
| POST | `/v1/models/{id}/unload` | Unload model, free NPU/GPU memory |
| POST | `/v1/models/{id}/reload` | Unload + reload (picks up config changes) |
| POST | `/v1/models/{id}/reset_dialog` | Reset KV cache only (~ms, weights stay on NPU) |
| GET | `/v1/models/{id}/system_prompt` | Get current system prompt |
| PATCH | `/v1/models/{id}/system_prompt` | Update system prompt at runtime |

## Chat & Completions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat — streaming + non-streaming, all backends |
| POST | `/v1/completions` | Legacy text completion |

**Chat request:**
```json
{
  "model": "my-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 1.0,
  "max_tokens": 512
}
```

**Streaming:** set `"stream": true` — returns SSE (`text/event-stream`).

## Embeddings

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/embeddings` | Compute text embeddings |

```json
{"model": "embedding-qnn", "input": "Hello world"}
```

## Audio

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/transcriptions` | Speech to text (multipart form: `file`, `model`) |
| POST | `/v1/audio/translations` | Speech to English text |
| POST | `/v1/audio/speech` | Text to speech |

## Images

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/images/generations` | Generate image from prompt |
| POST | `/v1/images/edits` | Edit image with mask (DALL-E) |
| POST | `/v1/images/variations` | Create image variations (DALL-E) |

## Moderations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/moderations` | Content moderation |

## Files (cloud pass-through)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/files` | Upload file |
| GET | `/v1/files` | List files |
| GET | `/v1/files/{id}` | Get file metadata |
| DELETE | `/v1/files/{id}` | Delete file |
| GET | `/v1/files/{id}/content` | Download file content |

## Fine-tuning (cloud pass-through)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/fine_tuning/jobs` | Create fine-tuning job |
| GET | `/v1/fine_tuning/jobs` | List jobs |
| GET | `/v1/fine_tuning/jobs/{id}` | Get job |
| POST | `/v1/fine_tuning/jobs/{id}/cancel` | Cancel job |
| GET | `/v1/fine_tuning/jobs/{id}/events` | Job events |
| GET | `/v1/fine_tuning/jobs/{id}/checkpoints` | Job checkpoints |

## Batch (cloud pass-through)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/batches` | Create batch job |
| GET | `/v1/batches` | List batches |
| GET | `/v1/batches/{id}` | Get batch |
| POST | `/v1/batches/{id}/cancel` | Cancel batch |

## Health & Observability

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + loaded models |
| GET | `/status` | Detailed status (backend, alive, lifecycle state, idle info per model) |
| GET | `/metrics` | JSON metrics snapshot (HTTP + inference + token + lifecycle counters) |
| GET | `/metrics/prometheus` | Prometheus text-exposition format (scrape target for Grafana/Prometheus) |
| GET | `/docs` | Swagger UI |
| GET | `/` | Server info |

---

## Authentication

Set `SERVER_API_KEY` env var to require Bearer token on all `/v1/*` routes.

```bash
export SERVER_API_KEY=my-secret-key
python server.py

# Call with token
curl http://localhost:8000/v1/models -H "Authorization: Bearer my-secret-key"
```

`/health`, `/status`, `/metrics`, `/metrics/prometheus`, `/docs` are always public.

---

## curl Examples

```bash
# List models
curl http://localhost:8000/v1/models

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# ASR
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.wav" -F "model=whisper-base"

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embedding-qnn","input":"Hello world"}'

# Reset KV cache
curl -X POST http://localhost:8000/v1/models/my-llm/reset_dialog

# Health
curl http://localhost:8000/health
# Status (per-model lifecycle state + idle info)
curl http://localhost:8000/status

# Metrics (JSON)
curl http://localhost:8000/metrics

# Metrics (Prometheus scrape format)
curl http://localhost:8000/metrics/prometheus