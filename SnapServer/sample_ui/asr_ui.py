# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Streamlit ASR UI
=================
Audio transcription interface matching demo_onnx_pipeline.py modes:
  file        – upload an existing audio file
  microphone  – record from browser microphone (requires --mic flag)

The UI sends audio to the server via POST /v1/audio/transcriptions.
The server calls whisper_asr_plugin.py → ONNXWhisperPipeline.transcribe()

Usage:
    # Start the server first:
    python server.py --port 8000

    # File upload only (default — no mic permissions needed):
    streamlit run examples/asr_ui.py

    # Enable microphone tab (requires Streamlit 1.35+ and browser mic permission):
    streamlit run examples/asr_ui.py -- --mic

    # Custom server URL + mic:
    streamlit run examples/asr_ui.py -- --server-url http://localhost:8000 --mic

Requirements:
    pip install streamlit>=1.35.0 openai httpx

NOTE: Microphone uses st.audio_input() — built into Streamlit 1.35+.
      No extra packages needed. Browser must allow microphone access.
      On non-localhost, HTTPS is required for browser mic permissions.
"""

import argparse
import io
import sys
import time

import streamlit as st

# ── Parse CLI args (streamlit passes args after "--") ─────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--mic", action="store_true",
                        help="Enable microphone recording tab (requires Streamlit 1.35+)")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

_args = _parse_args()
DEFAULT_SERVER_URL = _args.server_url
MIC_ENABLED = _args.mic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASR sample UI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .transcription-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.1em;
        line-height: 1.6;
        min-height: 80px;
        white-space: pre-wrap;
    }
    .stats-box {
        background: #e8f4f8;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9em;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "server_url":       DEFAULT_SERVER_URL,
        "model_id":         None,
        "available_models": [],
        "server_status":    "unknown",
        "transcriptions":   [],   # List[dict]
        "language":         "",
        "response_format":  "json",
        "request_count":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_client(server_url: str):
    try:
        from openai import OpenAI
        return OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="local")
    except ImportError:
        st.error("openai package not installed. Run: pip install openai")
        return None


def _fetch_models(server_url: str):
    try:
        import httpx
        r = httpx.get(f"{server_url.rstrip('/')}/v1/models", timeout=5)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        return []


def _check_health(server_url: str) -> bool:
    try:
        import httpx
        r = httpx.get(f"{server_url.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _get_audio_info(audio_bytes: bytes, filename: str) -> dict:
    """Get audio duration and sample rate from bytes using soundfile."""
    try:
        import soundfile as sf
        import numpy as np
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(audio_np) / sr
        return {
            "duration": duration,
            "sample_rate": sr,
            "shape": audio_np.shape,
            "size_kb": len(audio_bytes) / 1024,
        }
    except Exception:
        return {"size_kb": len(audio_bytes) / 1024}


def _transcribe(audio_bytes: bytes, filename: str, source: str) -> dict:
    """
    Send audio bytes to POST /v1/audio/transcriptions via OpenAI SDK.

    How the file is passed to the server:
      - audio_bytes: raw bytes from st.audio_input() or st.file_uploader()
      - Wrapped in BytesIO so the SDK can read it as a file stream
      - Passed as tuple: (filename, BytesIO(bytes), "audio/wav")
      - Equivalent to: curl -F "file=@speech.wav" -F "model=whisper-base"

    The server receives it in:
      server.py → audio_transcriptions() → session_manager.transcribe()
      → plugin_backend.transcribe() → whisper_asr_plugin.transcribe()
      → soundfile.read(BytesIO(bytes)) → ONNXWhisperPipeline.transcribe(audio_np, sr)
    """
    client = _get_client(st.session_state.server_url)
    if client is None:
        return {"error": True, "text": "No client — openai not installed"}

    model_id = st.session_state.model_id
    if not model_id:
        return {"error": True, "text": "No model selected. Connect to server and select a model."}

    # Get audio info for display
    info = _get_audio_info(audio_bytes, filename)

    t0 = time.time()
    try:
        # Pass audio as (filename, file_object, mime_type) tuple
        # BytesIO wraps the raw bytes so the SDK can read it as a file
        audio_file = (filename, io.BytesIO(audio_bytes), "audio/wav")

        params = dict(
            model=model_id,
            file=audio_file,
            response_format=st.session_state.response_format,
        )
        if st.session_state.language.strip():
            params["language"] = st.session_state.language.strip()

        result = client.audio.transcriptions.create(**params)
        elapsed = time.time() - t0

        text = result.text if hasattr(result, "text") else str(result)

        entry = {
            "text":        text,
            "source":      source,
            "filename":    filename,
            "elapsed":     elapsed,
            "ts":          time.strftime("%H:%M:%S"),
            "model":       model_id,
            "error":       False,
            "duration":    info.get("duration"),
            "sample_rate": info.get("sample_rate"),
            "size_kb":     info.get("size_kb", 0),
        }

    except Exception as exc:
        elapsed = time.time() - t0
        entry = {
            "text":    f"Error: {exc}",
            "source":  source,
            "filename": filename,
            "elapsed": elapsed,
            "ts":      time.strftime("%H:%M:%S"),
            "model":   model_id,
            "error":   True,
            "size_kb": info.get("size_kb", 0),
        }

    st.session_state.transcriptions.insert(0, entry)
    st.session_state.request_count += 1
    return entry


def _show_result(entry: dict) -> None:
    """Display a transcription result with stats matching demo_onnx_pipeline.py output."""
    if entry.get("error"):
        st.error(f"❌ {entry['text']}")
        return

    st.success("✅ Transcription completed!")

    # Transcription text
    st.markdown(
        f'<div class="transcription-box">{entry["text"]}</div>',
        unsafe_allow_html=True
    )

    # Performance stats (matching demo_onnx_pipeline.py output style)
    stats_lines = [
        f"⚡ Performance Summary:",
        f"  • Audio source      : {entry['source']}",
        f"  • Inference time    : {entry['elapsed']:.2f}s",
        f"  • Model             : {entry['model']}",
    ]
    if entry.get("duration"):
        stats_lines.append(f"  • Audio duration    : {entry['duration']:.2f}s")
    if entry.get("sample_rate"):
        stats_lines.append(f"  • Sample rate       : {entry['sample_rate']} Hz")
    stats_lines.append(f"  • File size         : {entry['size_kb']:.1f} KB")

    st.markdown(
        f'<div class="stats-box">' + "<br>".join(stats_lines) + "</div>",
        unsafe_allow_html=True
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎙️ ASR sample UI")
    st.caption("Audio Transcription Interface")

    # Server connection
    st.subheader("🔌 Server")
    new_url = st.text_input("Server URL", value=st.session_state.server_url,
                             placeholder="http://localhost:8000")
    if new_url != st.session_state.server_url:
        st.session_state.server_url = new_url
        st.session_state.available_models = []
        st.session_state.model_id = None
        st.cache_resource.clear()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Connect", use_container_width=True):
            with st.spinner("Connecting..."):
                ok = _check_health(st.session_state.server_url)
                st.session_state.server_status = "ok" if ok else "error"
                if ok:
                    st.session_state.available_models = _fetch_models(st.session_state.server_url)
                    if st.session_state.available_models:
                        st.session_state.model_id = st.session_state.available_models[0]
    with col2:
        status = st.session_state.server_status
        if status == "ok":
            st.success("Online", icon="✅")
        elif status == "error":
            st.error("Offline", icon="❌")
        else:
            st.info("Unknown", icon="❓")

    # Model selection
    st.subheader("🧠 Model")
    if st.session_state.available_models:
        idx = 0
        if st.session_state.model_id in st.session_state.available_models:
            idx = st.session_state.available_models.index(st.session_state.model_id)
        st.session_state.model_id = st.selectbox(
            "Select model", st.session_state.available_models, index=idx,
            label_visibility="collapsed"
        )
    else:
        st.info("Click Connect to load models")
        manual = st.text_input("Or enter model ID manually",
                               value=st.session_state.model_id or "whisper-base")
        st.session_state.model_id = manual

    # ASR settings
    st.subheader("⚙️ Settings")
    st.session_state.language = st.text_input(
        "Language (optional)",
        value=st.session_state.language,
        placeholder="e.g. en, fr, de — leave empty for auto-detect",
        help="ISO 639-1 language code. Leave empty for automatic detection."
    )
    st.session_state.response_format = st.selectbox(
        "Response format",
        ["json", "text", "verbose_json"],
        index=0,
        help="json = {text: ...}, text = plain string, verbose_json = with timestamps"
    )

    # Model management
    st.subheader("🔧 Model Controls")
    if st.session_state.model_id and st.session_state.server_status == "ok":
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📥 Load", use_container_width=True,
                        help="Pre-load model into memory"):
                with st.spinner(f"Loading {st.session_state.model_id}..."):
                    try:
                        import httpx
                        resp = httpx.post(
                            f"{st.session_state.server_url.rstrip('/')}/v1/models"
                            f"/{st.session_state.model_id}/load",
                            timeout=60,
                        )
                        if resp.status_code in (200, 201):
                            st.success("✅ Loaded!")
                        else:
                            st.error(f"❌ {resp.text[:100]}")
                    except Exception as e:
                        st.error(f"❌ {e}")
        with col_b:
            if st.button("📤 Unload", use_container_width=True,
                        help="Unload model from memory"):
                with st.spinner(f"Unloading {st.session_state.model_id}..."):
                    try:
                        import httpx
                        resp = httpx.post(
                            f"{st.session_state.server_url.rstrip('/')}/v1/models"
                            f"/{st.session_state.model_id}/unload",
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            st.success("✅ Unloaded!")
                        else:
                            st.error(f"❌ {resp.text[:100]}")
                    except Exception as e:
                        st.error(f"❌ {e}")

    # Stats
    st.divider()
    st.subheader("📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Transcriptions", st.session_state.request_count)
    c2.metric("In History", len(st.session_state.transcriptions))

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.transcriptions = []
        st.session_state.request_count = 0
        st.rerun()

    # Mic flag status
    st.divider()
    if MIC_ENABLED:
        st.success("🎤 Microphone enabled", icon="✅")
        st.caption("Using `st.audio_input()` — built into Streamlit 1.35+")
    else:
        st.info("🎤 Mic disabled", icon="ℹ️")
        st.caption("Run with `-- --mic` to enable microphone tab")

    st.caption("Swagger UI: [/docs](%s/docs)" % st.session_state.server_url)


# ── Main area ─────────────────────────────────────────────────────────────────
st.header("🎙️ Audio Transcription", divider="gray")
st.caption(
    "Matches `demo_onnx_pipeline.py` modes: **file** (upload) and **microphone** (browser). "
    "Audio is sent to the server → `whisper_asr_plugin.py` → `ONNXWhisperPipeline.transcribe()`"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
if MIC_ENABLED:
    tab_mic, tab_file = st.tabs(["🎤 Microphone", "📁 File Upload"])
else:
    tab_file = st.container()
    st.info(
        "🎤 **Microphone tab is disabled.** "
        "Run with `-- --mic` to enable:\n"
        "```\nstreamlit run examples/asr_ui.py -- --mic\n```",
        icon="ℹ️"
    )


# ── Microphone Tab ────────────────────────────────────────────────────────────
if MIC_ENABLED:
    with tab_mic:
        st.subheader("🎤 Record from Microphone")
        st.caption(
            "Equivalent to `--mode microphone` in `demo_onnx_pipeline.py`. "
            "Uses `st.audio_input()` — built into Streamlit 1.35+, no extra packages needed."
        )

        try:
            # st.audio_input() — built into Streamlit 1.35+
            # Returns an UploadedFile (same type as st.file_uploader)
            # Contains WAV audio recorded in the browser
            audio_value = st.audio_input(
                "🔴 Click the microphone icon to start/stop recording",
            )

            if audio_value is not None:
                # Read bytes from the UploadedFile object
                audio_bytes = audio_value.read()

                st.audio(audio_bytes, format="audio/wav")

                # Show audio info
                info = _get_audio_info(audio_bytes, "recording.wav")
                if info.get("duration"):
                    st.caption(
                        f"📊 Duration: **{info['duration']:.2f}s**  ·  "
                        f"Sample rate: **{info.get('sample_rate', '?')} Hz**  ·  "
                        f"Size: **{info['size_kb']:.1f} KB**"
                    )

                col_btn, col_info = st.columns([1, 3])
                with col_btn:
                    transcribe_btn = st.button(
                        "🔄 Transcribe",
                        use_container_width=True,
                        disabled=not st.session_state.model_id,
                        type="primary",
                        key="transcribe_mic",
                    )
                with col_info:
                    st.caption(
                        f"Model: **{st.session_state.model_id or 'none selected'}**  ·  "
                        f"Language: **{st.session_state.language or 'auto-detect'}**"
                    )

                if transcribe_btn:
                    with st.spinner("🔄 Running transcription..."):
                        entry = _transcribe(audio_bytes, "recording.wav", "🎤 Microphone")
                    _show_result(entry)

        except AttributeError:
            st.error(
                "❌ `st.audio_input()` requires **Streamlit 1.35+**.\n\n"
                "Upgrade: `pip install --upgrade streamlit`"
            )


# ── File Upload Tab ───────────────────────────────────────────────────────────
with tab_file:
    st.subheader("📁 Upload Audio File")
    st.caption(
        "Equivalent to `--mode file --audio-file speech.wav` in `demo_onnx_pipeline.py`. "
        "Supports the same formats: WAV, MP3, FLAC, OGG, M4A."
    )

    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
        help="Supported: WAV, MP3, FLAC, OGG, M4A, WebM — same as demo_onnx_pipeline.py",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Show audio player
        st.audio(uploaded)

        # Show file info (matching demo_onnx_pipeline.py output)
        audio_bytes = uploaded.read()
        info = _get_audio_info(audio_bytes, uploaded.name)

        info_parts = [f"📄 **{uploaded.name}**"]
        if info.get("duration"):
            info_parts.append(f"Duration: **{info['duration']:.2f}s**")
        if info.get("sample_rate"):
            info_parts.append(f"Sample rate: **{info['sample_rate']} Hz**")
        info_parts.append(f"Size: **{info['size_kb']:.1f} KB**")
        st.caption("  ·  ".join(info_parts))

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            transcribe_btn = st.button(
                "🔄 Transcribe",
                use_container_width=True,
                disabled=not st.session_state.model_id,
                type="primary",
                key="transcribe_file",
            )
        with col_info:
            st.caption(
                f"Model: **{st.session_state.model_id or 'none selected'}**  ·  "
                f"Language: **{st.session_state.language or 'auto-detect'}**"
            )

        if transcribe_btn:
            with st.spinner("🔄 Running transcription..."):
                # audio_bytes already read above
                # Passed to server as: (filename, BytesIO(bytes), "audio/wav")
                entry = _transcribe(audio_bytes, uploaded.name, f"📁 {uploaded.name}")
            _show_result(entry)


# ── Transcription History ─────────────────────────────────────────────────────
if st.session_state.transcriptions:
    st.divider()
    st.subheader(f"📜 History ({len(st.session_state.transcriptions)} transcriptions)")

    for i, item in enumerate(st.session_state.transcriptions):
        preview = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
        icon = "❌" if item.get("error") else "✅"

        with st.expander(
            f"{icon} [{item['ts']}] {item['source']} — {preview}",
            expanded=(i == 0)
        ):
            if item.get("error"):
                st.error(item["text"])
            else:
                st.markdown(
                    f'<div class="transcription-box">{item["text"]}</div>',
                    unsafe_allow_html=True
                )
                col1, col2, col3, col4 = st.columns(4)
                col1.caption(f"⏱ {item['elapsed']:.2f}s")
                col2.caption(f"🧠 {item['model']}")
                col3.caption(f"📥 {item['source']}")
                if item.get("duration"):
                    col4.caption(f"🎵 {item['duration']:.1f}s audio")

                if st.button(f"📋 Show full text", key=f"copy_{i}"):
                    st.code(item["text"], language=None)