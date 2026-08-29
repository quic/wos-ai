# CenterNet-Pose — Human Pose Estimation

Detects 17 COCO keypoints (nose, shoulders, elbows, wrists, hips, knees, ankles) per person in an image.

**Input:** JPEG/PNG image file  
**Output:** Keypoint coordinates `[x, y]` and per-keypoint confidence scores

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model centernet-pose --input path\to\photo.jpg
```

**Sample output:**
```
{'keypoints': [[320, 100], [310, 140], ...], 'scores': [0.95, 0.91, ...]}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model centernet-pose --input photo.jpg --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models centernet-pose
```

### Call the API

**Image variations endpoint** (`POST /v1/images/variations`):

```bash
curl -X POST http://localhost:8000/v1/images/variations \
  -F "image=@photo.jpg" \
  -F "model=centernet-pose"
```

```python
import base64, json, openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    result = client.images.create_variation(model="centernet-pose", image=f)

payload = json.loads(result.data[0].b64_json)
print(payload["keypoints"])  # [[x, y], ...] — 17 keypoints
print(payload["scores"])     # [0.95, 0.91, ...] per keypoint
```

**Response:**
```json
{"data": [{"b64_json": "{\"keypoints\": [[320, 100], ...], \"scores\": [0.95, ...]}"}]}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `image` |
| `input_type` | `pose_estimation` |
| `output_type` | `pose` |
| `resize` | `[256, 256]` |
| `resize_mode` | `contain` (letterbox) |
| `color_format` | `RGB` |
| `mean / std` | `[0, 0, 0] / [1, 1, 1]` (model applies CenterNet normalization internally) |
| `score_threshold` | 0.9 |
| `num_keypoints` | 17 |
| `keypoint_format` | `xyc` |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Single-session inference wiring |
| `config.yaml` | Heatmap decode and keypoint parameters |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear