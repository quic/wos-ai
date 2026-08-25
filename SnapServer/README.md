
# SnapServer : Snapdragon X Series AI Model Server

> Run local AI models on Qualcomm Snapdragon NPU and cloud models (on Qualcomm AIC100 or others) behind a single drop-in OpenAI API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991.svg)](https://platform.openai.com/docs/api-reference)

---

## What Is This?

This is a FastAPI server that exposes a full OpenAI-compatible REST API and routes requests to the right inference engine:

| Request | Local (Qualcomm NPU) | Cloud |
|---|---|---|
| Chat / text completion | Qualcomm Genie SDK (LLMs on HTP) | OpenAI / Azure / vLLM |
| Embeddings | ONNX Runtime QNN EP | OpenAI |
| CNN models (Non GenAI) | Custom plugin | OpenAI |

Any client using the `OpenAI Python SDK`, `LangChain`, or `curl` works unchanged — just by pointing `base_url` to `http://localhost:8000/v1` or `http://<ip_of_the_device>:<port_started>/v1`.

---

## Quick Start

```bash
# 1. Setup (auto-detects Python, downloads if missing, creates .venv)
setup_env.bat          # Windows
./setup_env.sh         # Linux / macOS

# 2. Configure models
#    Edit config/models.yaml — set your model paths
#    Create 1 models.yaml for 1 model so that each model can be started in new port

# 3. Start
.venv\Scripts\activate && python server.py      # Windows
source .venv/bin/activate && python server.py   # Linux / macOS

# 4. Test
curl http://localhost:8000/v1/models # List registered models
curl http://localhost:8000/health 
open http://localhost:8000/docs   # Swagger UI
```

**Use with OpenAI SDK:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="my-llm",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Repository Layout

```
WoS_ORT_GenAI_openai/
├── server.py              # FastAPI app — nearly 40+ OpenAI endpoints
├── config/models.yaml     # Model registry (edit this to add models either in single or as multiple yaml files)
├── core/                  # Config loader, model registry, session manager
├── backends/              # PluginBackend, OnnxQnnBackend, CloudBackend
├── utils/                 # InferencePlugin base class, prompt formatter, logger
├── sample_plugins/        # Ready-to-use plugin templates
├── sample_ui/             # Streamlit UI templates for user testing
├── requirements.txt       # Minimal server dependencies required to run
├── setup_env.bat/.ps1/.sh # One-click setup scripts
└── docs/                  # Detailed documentation
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagram, request lifecycle, concurrency model, backend internals |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | `models.yaml` full reference — all keys, examples for every backend |
| [docs/API.md](docs/API.md) | All 40+ endpoints with descriptions |
| [docs/PLUGINS.md](docs/PLUGINS.md) | Plugin system contract, sample plugins, adding a new model |

---

## Backends at a Glance

| Backend | Use case | `models.yaml` |
|---------|----------|---------------|
| `plugin` | Any Python inference code (recommended) and also supports Qualcomm Genie SDK | `backend: plugin` |
| `onnx_qnn` | ONNX embeddings / CNN models on QNN | `backend: onnx_qnn` |
| `openai` / `azure` / `vllm` | Cloud / self-hosted | `backend: openai` |

---

## Authentication

```bash
export SERVER_API_KEY=my-secret-key   # enable auth
python server.py
curl http://localhost:8000/v1/models -H "Authorization: Bearer my-secret-key"
```

## If you want to have multiple models.yamls in different ports

```bash
export SERVER_API_KEY=my-secret-key   # enable auth
python server.py --port 8001 --config config/model.yaml1 
curl http://localhost:8000/v1/models -H "Authorization: Bearer my-secret-key"
```

Public endpoints (no auth): `/health`, `/status`, `/metrics`, `/metrics/prometheus`, `/docs`
