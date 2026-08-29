# EasyOCR — Optical Character Recognition

Two-stage OCR pipeline: a CRAFT text detector finds text regions, then a CRNN recognizer reads each region. Supports English (80+ language vocab available).

**Input:** JPEG/PNG image file  
**Output:** List of detected text regions — bounding box, recognized text, and confidence score

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model easyocr --input path\to\photo.jpg
```

**Sample output:**
```
{'results': [{'box': [50, 20, 300, 60], 'text': 'STOP', 'confidence': 0.97}, ...]}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model easyocr --input photo.jpg --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models easyocr
```

### Call the API

**Image variations endpoint** (`POST /v1/images/variations`):

```bash
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@photo.jpg" \
  -F "model=easyocr"
```

```python
import json, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="easyocr", image=f)

payload = json.loads(result.data[0].b64_json)
for item in payload["results"]:
    print(item["text"], item["confidence"], item["box"])
```

**Response:**
```json
{
  "data": [{
    "b64_json": "{\"results\": [{\"box\": [50, 20, 300, 60], \"text\": \"STOP\", \"confidence\": 0.97}]}"
  }]
}
```

Box format for horizontal detections: `[xmin, xmax, ymin, ymax]`  
Box format for free-angle detections: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` (4 corners)

---

## Config

| Field | Value |
|---|---|
| `modality` | `image` |
| `input_type` | `ocr_detection` |
| `output_type` | `mask_list` |
| `resize` | `[800, 608]` (width × height) for detector |
| `resize_mode` | `contain` (letterbox) |
| `color_format` | `RGB` |
| `mean / std` | ImageNet `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` |
| `score_threshold` | 0.3 |
| `recognizer_h / w` | `64 × 800` (recognizer crop size) |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Two-stage pipeline: detector → crop → recognizer per box |
| `config.yaml` | Detector preprocessing and recognizer crop parameters |
| `easyocr_en_vocab.json` | English character vocabulary for the CTC recognizer |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear