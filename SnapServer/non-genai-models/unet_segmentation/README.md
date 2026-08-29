# Unet-Segmentation — Binary Image Segmentation

Produces a binary foreground/background segmentation mask for an image using a U-Net architecture.

**Input:** JPEG/PNG image file  
**Output:** Grayscale PNG mask (base64-encoded) — white pixels = foreground, black = background

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model unet-segmentation --input path\to\photo.jpg
```

**Sample output** (base64 PNG bytes in a list):
```
[{'b64_json': '<base64-encoded PNG mask>'}]
```

To decode and save the mask:
```python
import base64, json
result = [{'b64_json': '...'}]   # output from run_model.py
with open("mask.png", "wb") as f:
    f.write(base64.b64decode(result[0]["b64_json"]))
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model unet-segmentation --input photo.jpg --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models unet-segmentation
```

### Call the API

**Image variations endpoint** (`POST /v1/images/variations`):

```bash
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@photo.jpg" \
  -F "model=unet-segmentation"
```

```python
import base64, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="unet-segmentation", image=f)

# Save the segmentation mask
with open("mask.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
```

**Response:** `data[0].b64_json` is a base64-encoded grayscale PNG (0 = background, 255 = foreground).

---

## Config

| Field | Value |
|---|---|
| `modality` | `image` |
| `input_type` | `segmentation` |
| `output_type` | `segmentation` |
| `resize` | `[1280, 640]` (width × height) |
| `resize_mode` | `contain` (letterbox) |
| `color_format` | `RGB` |
| `mean / std` | `[0, 0, 0] / [1, 1, 1]` |
| `pad_mode` | `reflect` (avoids hard edges at padding boundaries) |
| `score_threshold` | 0.3 |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Single-session inference, mask → PNG encoding |
| `config.yaml` | Segmentation parameters |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear