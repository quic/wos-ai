# MobileNet-v2 — Image Classification

Classifies images into 1000 ImageNet categories. Lightweight model optimized for edge inference; expects 224×224 stretched input.

**Input:** JPEG/PNG image file  
**Output:** Top-5 class labels with confidence scores

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model mobilenet-v2 --input path\to\photo.jpg
```

**Sample output:**
```
{'labels': ['tabby cat', 'Egyptian cat', ...], 'scores': [0.72, 0.11, ...]}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model mobilenet-v2 --input photo.jpg --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models mobilenet-v2
```

### Call the API

**Image variations endpoint** (`POST /v1/images/variations`):

```bash
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@photo.jpg" \
  -F "model=mobilenet-v2"
```

```python
import base64, json, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="mobilenet-v2", image=f)

payload = json.loads(result.data[0].b64_json)
print(payload["labels"])   # ['tabby cat', 'Egyptian cat', ...]
print(payload["scores"])   # [0.72, 0.11, ...]
```

**Response:**
```json
{"data": [{"b64_json": "{\"labels\": [\"tabby cat\", ...], \"scores\": [0.72, ...]}"}]}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `image` |
| `input_type` | `classification` |
| `output_type` | `classification` |
| `resize` | `[224, 224]` |
| `resize_mode` | `stretch` |
| `color_format` | `RGB` |
| `mean / std` | ImageNet mean/std `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` |
| `labels_file` | not set by default — not shipped in this repo; point it at your own ImageNet class-name file to get named labels, otherwise results return numeric class indices |
| `top_k` | 5 |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Single-session inference wiring |
| `config.yaml` | Preprocessing and classification parameters |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear