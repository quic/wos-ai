# YamNet — Audio Event Classification

Classifies environmental sounds into 521 AudioSet categories (e.g. "Speech", "Dog", "Music").

**Input:** WAV/FLAC/OGG audio file  
**Output:** Top-5 event labels with confidence scores

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model yamnet --input path\to\audio.wav
```

**Sample output:**
```
{'text': 'Speech', 'classifications': [{'label': 'Speech', 'score': 0.94}, ...]}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model yamnet --input audio.wav --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models yamnet
```

### Call the API

**Transcription endpoint** (`POST /v1/audio/transcriptions`):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=yamnet"
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="yamnet", file=f)

print(result.text)  # top predicted label, e.g. "Speech"
```

**Response:**
```json
{"text": "Speech", "classifications": [{"label": "Speech", "score": 0.94}, ...]}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `audio` |
| `input_type` | `log_mel` |
| `output_type` | `softmax_top_k` |
| `sample_rate` | 16000 Hz |
| `chunk_duration_ms` | 960 ms |
| `top_k` | 5 |
| `labels_file` | `yamnet_labels.txt` (must be present in this folder) |

> **Note:** `yamnet_labels.txt` is not included in this repo. Place the 521-class AudioSet label file here before running.

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear