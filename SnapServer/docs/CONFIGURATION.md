# Configuration — models.yaml (if you want to have all models registered in same server)

`config/models.yaml` is the **only file you need edit** to add, remove, or configure models.

---

## Multiple Config Files (different models on different ports)

You can run multiple server instances, each with its own `models.yaml`, on different ports:

```bash
# Terminal 1 — local NPU models on port 8000
python server.py --port 8000 --config config/models-local-1.yaml

# Terminal 2 — cloud models on port 8001
python server.py --port 8001 --config config/models-cloud.yaml
```
---

## Examples

### Cloud: OpenAI / Azure / vLLM

```yaml
models:
  - id: gpt-4o
    backend: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}       # reads from env var
    model_name: gpt-4o
    owned_by: openai

  - id: azure-gpt4
    backend: azure
    base_url: https://my-resource.openai.azure.com
    api_key: ${AZURE_OPENAI_KEY}
    model_name: gpt-4
    api_version: "2024-02-01"
    owned_by: azure

  - id: llama3-vllm
    backend: vllm
    base_url: http://my-vllm-server:8000/v1
    api_key: token-abc123
    model_name: meta-llama/Meta-Llama-3-8B-Instruct
    owned_by: self-hosted
```

### Local: Genie LLM on NPU (plugin — recommended as it has support with Genie API)

```yaml
  - id: my-llm-genie
    backend: plugin
    plugin_module: sample_plugins/genie_plugin.py
    plugin_class: GeniePlugin
    genie_model_dir: C:/path/to/model_folder      # folder with Genie.dll + model files
    genie_config: genie_config.json               # filename, resolved relative to genie_model_dir
    tokenizer_path: C:/path/to/model_folder       # optional, for HF chat templates
    system_prompt: "You are a helpful AI assistant."
    performance_policy: burst                     # burst | high_performance | balanced | power_saver
    owned_by: qualcomm
```

### Local: Other CNN models from ORT path (used whisper models as sample)

```yaml
  - id: whisper-base
    backend: plugin
    plugin_module: sample_plugins/sample_asr_plugin.py
    plugin_class: WhisperASRPlugin
    encoder_path: /path/to/whisper_encoder.onnx
    decoder_path: /path/to/whisper_decoder.onnx
    tokenizer_path: /path/to/whisper-base
    use_qnn: true
    owned_by: qualcomm
```

### Local: ONNX QNN Embedding Model

```yaml
  - id: embedding-qnn
    backend: onnx_qnn
    model_type: embedding
    model_path: /path/to/embedding.onnx
    tokenizer_path: /path/to/all-MiniLM-L6-v2
    use_qnn: true
    qnn_backend: QnnHtp.dll
    owned_by: qualcomm
```

---

## Key Reference

### Universal keys

| Key | Required | Description |
|-----|----------|-------------|
| `id` | Yes | Unique model ID used in API calls |
| `backend` | Yes | `plugin`, `onnx_qnn`, `openai`, `azure`, `vllm` |
| `owned_by` | No | Shown in `/v1/models` response |
| `lazy_load` | No | `true` (default) = load on first request; `false` = load at startup |
| `idle_timeout_minutes` | No | Auto-unload after N minutes idle (0 = never) |

### Local model keys

| Key | Description |
|-----|-------------|
| `max_tokens` | Default max generation length |
| `system_prompt` | Default system prompt (overridable per-request) |
| `chat_template` | Jinja2 template string (overrides tokenizer template) |
| `tokenizer_path` | HF tokenizer path for chat template formatting |

### Plugin backend keys

| Key | Required | Description |
|-----|----------|-------------|
| `plugin_module` | Yes | Path to `.py` file or importable module name |
| `plugin_class` | Yes | Class name inside the module |

### Genie backend keys

| Key | Required | Description |
|-----|----------|-------------|
| `genie_config` | Yes | Path to `genie_config.json` |
| `genie_model_dir` | Yes | Folder containing Genie.dll + model files (simple setup) |
| `genie_lib_path` | No | Explicit path to `Genie.dll` / `libGenie.so` |
| `genie_lib_dirs` | No | Dirs added to `PATH` / `LD_LIBRARY_PATH` |
| `genie_hexagon_dirs` | No | Hexagon DSP dirs for `ADSP_LIBRARY_PATH` |
| `performance_policy` | No | `burst`, `high_performance`, `balanced`, `power_saver` |

### ONNX QNN backend keys

| Key | Required | Description |
|-----|----------|-------------|
| `model_path` | Yes | Path to `.onnx` model file |
| `model_type` | No | `embedding`, `classification`, `asr`, `text` |
| `use_qnn` | No | Enable QNN EP (default: `true`) |
| `qnn_backend` | No | `QnnHtp.dll` / `QnnCpu.dll` / `libQnnHtp.so` |
| `qnn_fp16` | No | Use FP16 precision |

### Cloud backend keys

| Key | Required | Description |
|-----|----------|-------------|
| `base_url` | Yes | API base URL |
| `api_key` | Yes | API key — supports `${ENV_VAR}` expansion |
| `model_name` | Yes | Model name on the remote server |
| `api_version` | Azure only | Azure API version (e.g. `"2024-02-01"`) |
| `timeout` | No | Request timeout in seconds |

---

## System Prompt Priority

Highest to lowest:
1. `{"role": "system", "content": "..."}` in the request `messages[]`
2. `system_prompt` in `models.yaml`
3. Built-in default: `"You are a helpful AI assistant."`

Update at runtime (no restart):
```bash
curl -X PATCH http://localhost:8000/v1/models/my-llm/system_prompt \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a coding assistant."}'
```

---

## Model Lifecycle Endpoints

```bash
POST /v1/models/{id}/load          # load on demand (lazy models)
POST /v1/models/{id}/unload        # free NPU memory without restart
POST /v1/models/{id}/reload        # unload + reload (picks up config changes)
POST /v1/models/{id}/reset_dialog  # clear KV cache only (~ms, weights stay on NPU)
GET  /v1/models/{id}/system_prompt # read current system prompt
PATCH /v1/models/{id}/system_prompt # update system prompt at runtime


<!-- Footer -->
<div style="position: fixed; bottom: 0; width: 100%; text-align: center; font-size: small; color: gray; opacity: 0.5;">
  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. <br>
  SPDX-License-Identifier: BSD-3-Clause-Clear
</div>