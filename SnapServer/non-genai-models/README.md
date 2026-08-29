# Scripts to run AIHub models on-device with OpenAI

Pre/post-processing "codebase" for running a set of on-device vision, audio, and OCR
models (mostly ONNX exports of Qualcomm AI Hub models, plus a ported PaddleOCR
pipeline) either as a **standalone CLI** or behind an **OpenAI-compatible API** served
by the companion `WoS_ServerClientIntegration` server.

Every model folder plugs into the same two entry points:

1. `run_model.py` — run a model directly from the command line, no server involved.
2. The **WoS server** (external repo, not included here) — loads `configs/models_prepost.yaml`,
   exposes each model behind an OpenAI-compatible REST API (`/v1/audio/transcriptions`,
   `/v1/images/variations`, etc.), and lets you call it with the official `openai` Python SDK.

Models can run on CPU (`onnxruntime` / `DmlExecutionProvider`) or on a Qualcomm NPU
(`QNNExecutionProvider` / QNN SDK), selectable per-model via `use_qnn` / `--qnn` / `--cpu`.

---

## Repository layout

```
.
├── run_model.py                # Standalone CLI runner (no server, no OpenAI)
├── setup.py                    # Installs pipeline_core as a package
├── setup.ps1                   # Windows bootstrap: venv + per-model requirements + smoke tests
├── requirements.txt            # Shared dependencies (numpy, Pillow, onnxruntime, PyYAML, ...)
├── configs/
│   └── models_prepost.yaml     # Model registry consumed by run_model.py and the WoS server
├── pipeline_core/               # Shared pre/post-processing + onnxruntime session helpers
│
├── paddle-ocr/                  # PaddleOCR (detection + recognition), ONNX / QNN — see below
├── easyocr/                     # CRAFT detector + CRNN recognizer OCR
├── yamnet/                      # Audio event classification
├── whisper_base/                # Speech-to-text
├── zipformer/                   # Speech-to-text (streaming-style encoder/decoder/joiner)
├── inception_v3/                # Image classification
├── mobilenet_v2/                # Image classification
├── yolov3/                      # Object detection
├── centernet_pose/              # Pose estimation
├── unet_segmentation/           # Image segmentation
└── aotgan/                      # Image inpainting
```

Each model folder (except `paddle-ocr/`, see below) follows the same shape:

| File | Purpose |
|---|---|
| `plugin.py` | Loads the ONNX model and implements the pre/post-processing + inference call (`transcribe`, `image_variation`, or `generate`) |
| `config.yaml` | Preprocessing parameters (resize, normalization, thresholds, etc.) |
| `README.md` | Model-specific usage: standalone CLI and OpenAI/WoS server examples |
| assorted `*.json`/`*.txt` | Vocabularies, label maps, character dictionaries used by that model |

`configs/models_prepost.yaml` is the single source of truth mapping a model `id` to its
plugin module/class and model file paths, using `${PREPOST_CODEBASE}` and `${MODEL_BASE}`
env-var expansion. It also contains one `backend: cloud` entry (`gpt-4o`) that proxies
straight to the OpenAI API using `OPENAI_API_KEY`.

---

## Setup

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

This locates a Python 3.12 (AMD64) interpreter, creates a `.venv`, installs the root
`requirements.txt` plus each model folder's own `requirements.txt`, and runs import
smoke tests. **`paddle-ocr/` is not covered by `setup.ps1`** — see its own
[README](paddle-ocr/README.md) for setup.

### Manual

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # installs pipeline_core
```

Set the two environment variables used throughout the config and CLI:

```powershell
$env:PREPOST_CODEBASE = "C:\path\to\this\repo"
$env:MODEL_BASE        = "C:\path\to\XElite_models"   # directory containing compiled .onnx models
```

---

## Running a model standalone (no server, no OpenAI)

```bash
python run_model.py --model <id> --input <file> [--cpu | --qnn] [--mask mask.png]
```

Examples:

```bash
python run_model.py --model yamnet           --input audio.wav
python run_model.py --model whisper-base     --input audio.wav
python run_model.py --model zipformer        --input audio.wav
python run_model.py --model inception-v3     --input photo.jpg
python run_model.py --model mobilenet-v2     --input photo.jpg
python run_model.py --model yolov3           --input photo.jpg
python run_model.py --model centernet-pose   --input photo.jpg
python run_model.py --model unet-segmentation --input photo.jpg
python run_model.py --model quicksrnet-small  --input photo.jpg
python run_model.py --model easyocr          --input photo.jpg
python run_model.py --model aotgan           --input photo.jpg --mask mask.png
```

`--model` must match an `id` in `configs/models_prepost.yaml`. `run_model.py`
dynamically loads that model's `plugin.py`, calls `load()`, dispatches to
`transcribe()` / `image_variation()` / `generate()` based on the input file's
extension, and prints the result.

`paddle-ocr` is **not** currently registered in `configs/models_prepost.yaml`, so it
cannot be run through `run_model.py` today — run it via its own standalone script
instead (see below).

---

## Running behind the OpenAI-compatible API (WoS server)

The WoS server (`WoS_ServerClientIntegration`, a sibling repo/folder not included here)
loads `configs/models_prepost.yaml` and exposes each `backend: plugin` model behind an
OpenAI-compatible REST endpoint.

```bash
cd WoS_ServerClientIntegration-main
python server.py --config ..\configs\models_prepost.yaml 
```

Then call it with the official `openai` Python SDK pointed at the local server:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Audio models (yamnet, whisper-base, zipformer)
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-base", file=f)

# Image models (inception-v3, mobilenet-v2, yolov3, centernet-pose,
# unet-segmentation, easyocr, aotgan, ...)
with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="easyocr", image=f)
```

See each model's own `README.md` for its exact request/response shape (e.g.
[easyocr/README.md](easyocr/README.md)).

---

## PaddleOCR

`paddle-ocr/` is handled slightly differently from the other model folders — it ships
its own dependency list and model-download steps rather than a `config.yaml`/`requirements.txt`,
and is not yet wired into `configs/models_prepost.yaml`. **Follow the steps in
[paddle-ocr/README.md](paddle-ocr/README.md)** to install its dependencies, download and
convert the detection/recognition ONNX models, run it standalone on CPU/NPU, and register
it so it can be called through the OpenAI-compatible API.

---

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear
