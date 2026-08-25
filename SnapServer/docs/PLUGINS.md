# Plugin System

The plugin system lets you wrap any existing Python inference code in ~30 lines without rewriting it.

---

## Minimal Plugin

```python
# my_plugin.py
from utils.inference_plugin import InferencePlugin

class MyPlugin(InferencePlugin):

    def load(self, model_id: str, config: dict) -> None:
        """Called once at startup. Initialize your model here."""
        self.model_id = model_id
        # config contains all fields from models.yaml for this model

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs):
        """Yield tokens one at a time."""
        yield "Hello "
        yield "world!"

    def unload(self) -> None:
        """Free all resources."""
        pass
```

Register in `models.yaml`:

```yaml
- id: my-model
  backend: plugin
  plugin_module: my_plugin.py        # path relative to project root
  plugin_class: MyPlugin
  # any other fields are passed to load() as config
```

---

## Text Generation → `POST /v1/chat/completions`, `POST /v1/completions`


### Multi-turn with KV Cache

For models that maintain conversation state (e.g. Genie SDK), implement
`generate_with_messages()` instead of `generate()`. The server passes the full
messages list and your plugin manages the KV cache:

```python
def generate_with_messages(self, messages, max_tokens=512, **kwargs):
    """
    Called instead of generate() when the plugin implements it.
    messages = [{"role": "user", "content": "..."}, ...]
    """
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    # Build delta prompt, call model, yield tokens
    yield from self._run_inference(last_user, max_tokens)
```

---

### Genie API's Callback for LLM's (lowest TTFT)

For the lowest Time-To-First-Token, implement `generate_with_messages_cb()`.

```python
def generate_with_messages_cb(self, messages, max_tokens=512,
                               on_token=None, **kwargs) -> None:
    """
    Called by plugin_backend.py when detected (preferred over generate_with_messages).
    on_token(token_str) is called synchronously for each generated token.
    dialog.query() runs in the current thread — no inner thread spawned.
    """
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")

    def _callback(token, sentence_code, user_data):
        if token and on_token:
            on_token(token)

    self._dialog.query(last_user, SentenceCode.COMPLETE, _callback)
```

See `sample_plugins/genie_plugin.py` for a complete implementation.

`GeniePlugin.load()` auto-detects the model's real chat template and a
default end-of-turn stop sequence from `tokenizer_config.json` (found via
`genie_model_dir`/`tokenizer_path`, or overridden directly with a
`chat_template` string in `models.yaml`), without it, prompts fall back to
a generic, model-mismatched format and generation won't reliably stop at
the model's own turn boundary. 
If that file/field is missing, a `WARNING` is logged at load time, 
and `GET /status`'s `prompt_diagnostics` field for that model shows 
`chat_template_loaded: false` so it's visible without digging through logs. 
Client-supplied `stop` values in a request always override the derived default for that turn.

---

### Regular method

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 1.0,
    stop: list = [],
    **kwargs,
) -> Iterator[str]:
    yield "token1"
    yield "token2"
```

---

## Embeddings → `POST /v1/embeddings`

```python
def embed(
    self,
    input,           # str or List[str]
    **kwargs,
) -> List[List[float]]:
    # return one float vector per input string
    return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

---

## Audio Transcription → `POST /v1/audio/transcriptions`

```python
def transcribe(
    self,
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: str = None,       # e.g. "en", "fr" — None = auto-detect
    prompt: str = None,
    response_format: str = "json",
    temperature: float = 0.0,
    **kwargs,
) -> dict:
    return {"text": "transcribed text here"}
```

---

## Audio Translation → `POST /v1/audio/translations`

```python
def translate(
    self,
    audio_bytes: bytes,
    filename: str = "audio.wav",
    prompt: str = None,
    response_format: str = "json",
    temperature: float = 0.0,
    **kwargs,
) -> dict:
    return {"text": "translated to English text"}
```

---

## Text-to-Speech → `POST /v1/audio/speech`

```python
def synthesize(
    self,
    text: str,
    voice: str = "alloy",
    response_format: str = "mp3",  # mp3 | wav | opus | aac | flac | pcm
    speed: float = 1.0,
    **kwargs,
) -> bytes:
    return audio_bytes  # raw audio in requested format
```

---

## Content Moderation → `POST /v1/moderations`

```python
def moderate(
    self,
    input,      # str or List[str]
    **kwargs,
) -> dict:
    return {
        "results": [
            {
                "flagged": False,
                "categories": {"hate": False, "violence": False, ...},
                "category_scores": {"hate": 0.001, "violence": 0.002, ...}
            }
        ]
    }
```

---

## Image Generation → `POST /v1/images/generations`

```python
def image_generate(
    self,
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    quality: str = "standard",
    response_format: str = "url",   # "url" or "b64_json"
    style: str = None,
    **kwargs,
) -> List[dict]:
    return [{"b64_json": base64_encoded_png}]
    # or: return [{"url": "https://..."}]
```

---

## Image Editing → `POST /v1/images/edits`

```python
def image_edit(
    self,
    image: bytes,           # original image PNG bytes
    prompt: str,            # description of desired edit
    mask: bytes = None,     # optional mask — transparent areas are edited
    n: int = 1,
    size: str = "1024x1024",
    response_format: str = "url",
    **kwargs,
) -> List[dict]:
    return [{"b64_json": base64_encoded_png}]
```

---

## Image Variation → `POST /v1/images/variations`

```python
def image_variation(
    self,
    image: bytes,
    n: int = 1,
    size: str = "1024x1024",
    response_format: str = "url",
    **kwargs,
) -> List[dict]:
    return [{"b64_json": base64_encoded_png}]
```

---

## Plugin Templates

| File | Description |
|---|---|
| `sample_plugins/example_template_plugin.py` | Minimal template — start here |
| `sample_plugins/genie_plugin.py` | Full Genie SDK plugin with multi-turn KV cache |
| `sample_plugins/cpp_plugin.py` | Template for wrapping any C++ shared library |
| `sample_plugins/onnx_qnn_plugin.py` | ONNX Runtime + QNN Execution Provider |
| `sample_plugins/sample_asr_plugin.py` | Sample ASR (speech-to-text) template |

---

## Utilizing Plugins

### Option A: Plugin (recommended)

1. Copy `sample_plugins/example_template_plugin.py` to `sample_plugins/my_model.py`
2. Implement `load()` and `generate()` with your existing code
3. Add to `config/models.yaml`:

```yaml
- id: my-model
  backend: plugin
  plugin_module: sample_plugins/my_model.py
  plugin_class: MyModelPlugin
  model_path: /path/to/model
```

4. Restart server or call `POST /v1/models/my-model/reload`

### Option B: Cloud

```yaml
- id: gpt-4o
  backend: openai
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model_name: gpt-4o
```

---

## Sample UIs

| File | Launch |
|------|--------|
| `sample_ui/chat_ui.py` | `streamlit run sample_ui/chat_ui.py` |
| `sample_ui/asr_ui.py` | `streamlit run sample_ui/asr_ui.py` |

---

Reset KV cache without unloading weights:
```bash
curl -X POST http://localhost:8000/v1/models/my-llm/reset_dialog