# Whisper-Base — Multilingual ASR

Transcribes speech to text in 99 languages using OpenAI Whisper-Base (encoder + decoder split ONNX).

**Input:** WAV/FLAC/OGG audio file (up to ~30 seconds)  
**Output:** Transcribed text string

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model whisper-base --input path\to\audio.wav
```

**Sample output:**
```
{'text': 'The quick brown fox jumps over the lazy dog.'}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model whisper-base --input audio.wav --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models whisper-base
```

### Call the API

**Transcription endpoint** (`POST /v1/audio/transcriptions`):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-base"
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-base", file=f)

print(result.text)
```

**Response:**
```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `audio` |
| `input_type` | `log_mel` |
| `output_type` | `attention_greedy` |
| `sample_rate` | 16000 Hz |
| `chunk_duration_ms` | 30000 ms (30 s window) |
| `n_mels` | 80 |
| `vocab_file` | `whisper_vocab.json` (in this folder) |
| `max_decode_len` | 224 tokens (set in `models_prepost.yaml`) |

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Encoder+decoder session wiring, KV-cache autoregressive decode |
| `config.yaml` | Preprocessing and postprocessing parameters |
| `whisper_vocab.json` | Token vocabulary for the attention decoder |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear
