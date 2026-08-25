"""
Streamlit Chat UI
==================
Multi-turn chat interface that connects to the local server via the OpenAI SDK.

Usage:
    # Start the server first:
    python server.py --port 8000

    # Then launch the UI:
    streamlit run examples/chat_ui.py

    # Point to a different server:
    streamlit run examples/chat_ui.py -- --server-url http://localhost:8000
"""

import argparse
import sys
import time
from typing import List, Dict

import streamlit as st

# ── Parse --server-url from CLI (streamlit passes args after "--") ────────────
def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server-url", default="http://localhost:8000")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

_args = _parse_args()
DEFAULT_SERVER_URL = _args.server_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM's Chat UI with Genie API's",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 8px; margin-bottom: 4px; }
    .sidebar-section { background: #f0f2f6; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .metric-row { display: flex; gap: 8px; }
    .status-ok  { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],          # List[{"role": str, "content": str}]
        "server_url":     DEFAULT_SERVER_URL,
        "model_id":       None,
        "available_models": [],
        "system_prompt":  "You are a helpful AI assistant.",
        "temperature":    0.7,
        "max_tokens":     1024,
        "stream":         True,
        "server_status":  "unknown",   # "ok" | "error" | "unknown"
        "total_tokens":   0,
        "request_count":  0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── OpenAI client factory ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_client(server_url: str):
    try:
        from openai import OpenAI
        return OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="local")
    except ImportError:
        st.error("openai package not installed. Run: pip install openai")
        return None


def _fetch_models(server_url: str) -> List[str]:
    """Fetch model list from /v1/models."""
    try:
        import httpx
        r = httpx.get(f"{server_url.rstrip('/')}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        return sorted(m["id"] for m in data)
    except Exception:
        return []


def _check_health(server_url: str) -> bool:
    try:
        import httpx
        r = httpx.get(f"{server_url.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Sample Chat UI")
    st.caption("SnapServer : Snapdragon X Series AI Model Server")

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
                               value=st.session_state.model_id or "gpt-4o-mini")
        st.session_state.model_id = manual

    # Generation parameters (optional overrides)
    st.subheader("⚙️ Parameters")
    with st.expander("Advanced (Optional)", expanded=False):
        st.caption("Leave empty to use model defaults from genie_config")
        
        use_temp = st.checkbox("Override temperature", value=False)
        if use_temp:
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 2.0, st.session_state.temperature, 0.05,
                help="Higher = more creative, lower = more deterministic"
            )
        else:
            st.session_state.temperature = None
        
        use_max = st.checkbox("Override max tokens", value=False)
        if use_max:
            st.session_state.max_tokens = st.slider(
                "Max tokens", 64, 8192, st.session_state.max_tokens or 1024, 64
            )
        else:
            st.session_state.max_tokens = None
    
    st.session_state.stream = st.toggle("Streaming", value=st.session_state.stream,
                                         help="Stream tokens as they are generated")

    # System prompt
    st.subheader("📋 System Prompt")
    st.session_state.system_prompt = st.text_area(
        "System prompt", value=st.session_state.system_prompt,
        height=100, label_visibility="collapsed"
    )

    # Stats
    st.subheader("📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Requests", st.session_state.request_count)
    c2.metric("~Tokens", st.session_state.total_tokens)

    # Model management actions
    st.divider()
    
    # Model management buttons
    if st.session_state.model_id and st.session_state.server_status == "ok":
        col_a, col_b, col_c = st.columns(3)
        
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
                            st.success("✅ Loaded!", icon="✅")
                        else:
                            st.error(f"❌ Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col_b:
            if st.button("📤 Unload", use_container_width=True,
                        help="Unload model from memory (frees NPU/GPU)"):
                with st.spinner(f"Unloading {st.session_state.model_id}..."):
                    try:
                        import httpx
                        resp = httpx.post(
                            f"{st.session_state.server_url.rstrip('/')}/v1/models"
                            f"/{st.session_state.model_id}/unload",
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            st.success("✅ Unloaded!", icon="✅")
                        else:
                            st.error(f"❌ Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col_c:
            if st.button("🗑️ Clear", use_container_width=True,
                        help="Clear conversation and reset dialog"):
                # 1. Clear local message history
                st.session_state.messages = []
                st.session_state.total_tokens = 0
                st.session_state.request_count = 0

                # 2. Reset server-side KV cache (clears dialog state without unloading model)
                try:
                    import httpx
                    httpx.post(
                        f"{st.session_state.server_url.rstrip('/')}/v1/models"
                        f"/{st.session_state.model_id}/reset_dialog",
                        timeout=5,
                    )
                except Exception:
                    pass  # Non-fatal — local history is already cleared

                st.rerun()
    else:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.request_count = 0
            st.rerun()

    st.caption("Swagger UI: [/docs](%s/docs)" % st.session_state.server_url)


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("💬 Chat", divider="gray")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type a message…", disabled=not st.session_state.model_id):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages list (system + history)
    api_messages = []
    if st.session_state.system_prompt.strip():
        api_messages.append({"role": "system", "content": st.session_state.system_prompt})
    api_messages.extend(st.session_state.messages)

    # Call the server
    client = _get_client(st.session_state.server_url)
    if client is None:
        st.stop()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        t0 = time.time()

        try:
            # Build API parameters (only include non-None values)
            api_params = {
                "model": st.session_state.model_id,
                "messages": api_messages,
                "stream": st.session_state.stream,
            }
            if st.session_state.temperature is not None:
                api_params["temperature"] = st.session_state.temperature
            if st.session_state.max_tokens is not None:
                api_params["max_tokens"] = st.session_state.max_tokens
            
            if st.session_state.stream:
                # Streaming response
                stream = client.chat.completions.create(**api_params)
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                # Non-streaming response
                with st.spinner("Thinking…"):
                    resp = client.chat.completions.create(**api_params)
                full_response = resp.choices[0].message.content or ""
                placeholder.markdown(full_response)

            elapsed = time.time() - t0
            approx_tokens = len(full_response.split())
            st.caption(f"⏱ {elapsed:.1f}s  ·  ~{approx_tokens} tokens")

        except Exception as exc:
            full_response = f"❌ Error: {exc}"
            placeholder.error(full_response)

    # Save assistant message and update stats
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.request_count += 1
    st.session_state.total_tokens += len(full_response.split()) + len(prompt.split())