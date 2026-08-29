# Zipformer — Streaming RNN-T ASR (English + Chinese)

Transcribes English and Mandarin speech using a Zipformer encoder with RNN-T transducer decoding (encoder + decoder + joiner, three ONNX files).

**Input:** WAV/FLAC/OGG audio file  
**Output:** Transcribed text string

---

## Standalone (run_model.py)

```bash
# From repo root — activate venv first
venv\Scripts\activate

set PREPOST_CODEBASE=<path to this repo>
set MODEL_BASE=<path to your XElite_models directory>

python %PREPOST_CODEBASE%\run_model.py --model zipformer --input path\to\audio.wav
```

**Sample output:**
```
{'text': 'the weather today is sunny'}
```

**Force CPU (no QNN):**
```bash
python %PREPOST_CODEBASE%\run_model.py --model zipformer --input audio.wav --cpu
```

---

## OpenAI API (WoS server)

### Start server

```bash
cd WoS_ServerClientIntegration-main
python server.py --config %PREPOST_CODEBASE%\configs\models_prepost.yaml --models zipformer
```

### Call the API

**Transcription endpoint** (`POST /v1/audio/transcriptions`):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=zipformer"
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="zipformer", file=f)

print(result.text)
```

**Response:**
```json
{"text": "the weather today is sunny"}
```

---

## Config

| Field | Value |
|---|---|
| `modality` | `audio` |
| `input_type` | `kaldi_fbank` |
| `output_type` | `transducer_greedy` |
| `sample_rate` | 16000 Hz |
| `n_mels` | 80 |
| `feature_layout` | `BTC` — `[batch, T, n_mels]` |
| `vocab_file` | `../../../models/zipformer/tokens.txt` |
| `blank_id` | 0 |
| `context_size` | 2 |

## Dependencies

- **k2** — required for CTC beam decoding (installed in venv)
- **torchaudio** — Kaldi fbank feature extraction
- `tokens.txt` — token vocabulary located at `models/zipformer/tokens.txt` (in the `qai_hub_models` directory, not committed here)
- Positional embeddings — `pos_emb/pos_emb_*.bin` in the repo root

## Files in this folder

| File | Purpose |
|---|---|
| `plugin.py` | Encoder → decoder → joiner RNN-T greedy loop |
| `config.yaml` | Kaldi fbank parameters and transducer settings |

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
SPDX-License-Identifier: BSD-3-Clause-Clear