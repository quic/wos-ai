# YOLOv3 — Object Detection

Detects objects from 80 COCO classes in images. Uses letterbox (contain) resize to 640×640 and returns bounding boxes with labels and scores after NMS.

**Input:** JPEG/PNG image file  
**Output:** List of detections — bounding box `[x1, y1, x2, y2]`, label, and confidence score

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model yolov3 --input path\to\photo.jpg
```

**Sample output:**
```
{'detections': [{'label': 'car', 'score': 0.87, 'box': [120, 45, 380, 290]}, ...]}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model yolov3 --input photo.jpg --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models yolov3
```

### Call the API

**Image variations endpoint** (`POST /v1/images/variations`):

```bash
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@photo.jpg" \
  -F "model=yolov3"
```

```python
import base64, json, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="yolov3", image=f)

payload = json.loads(result.data[0].b64_json)
for det in payload["detections"]:
    print(det["label"], det["score"], det["box"])
```

**Response:**
```json
{"data": [{"b64_json": "{\"detections\": [{\"label\": \"car\", \"score\": 0.87, \"box\": [120, 45, 380, 290]}, ...]}"}]}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `image` |
| `input_type` | `object_detection` |
| `output_type` | `detection` |
| `resize` | `[640, 640]` |
| `resize_mode` | `contain` (letterbox, zero-padded) |
| `color_format` | `RGB` |
| `score_threshold` | 0.50 |
| `iou_threshold` | 0.45 |
| `labels_file` | not set by default — not shipped in this repo; point it at your own COCO class-name file to get named labels, otherwise detections return numeric class indices |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Single-session inference wiring with NMS postprocessing |
| `config.yaml` | Detection parameters (score/IoU thresholds, resize) |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear