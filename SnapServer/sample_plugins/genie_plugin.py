# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Genie Plugin
====================
Thin plugin wrapper is for Qualcomm Genie SDK LLM inference.


models.yaml entry 
-------------------
  - id: my-llm
    backend: plugin
    plugin_module: sample_plugins/genie_plugin.py
    plugin_class: GeniePlugin
    genie_model_dir: C:/path/to/model_folder        # folder with Genie.dll + model files
    genie_config: genie_config.json                 # filename, resolved relative to genie_model_dir
    max_tokens: 4096                                # optional
    system_prompt: "You are a helpful AI assistant."# optional
    performance_policy: balanced                    # optional
    owned_by: qualcomm


PATH FORMAT NOTE
----------------
Use forward slashes in YAML. Do NOT use Python r"..." syntax.
  CORRECT:  genie_model_dir: C:/Users/me/workspace/my_model
  WRONG:    genie_model_dir: r"C:\\Users\\me\\workspace\\my_model"

Multi-turn behavior
-------------------
GeniePlugin implements generate_with_messages() which plugin_backend.py calls instead of generate() when the plugin supports it.

  Turn 1: system + user1 + <asst>          → KV: [sys, user1]
  Turn 2: </asst> + user2 + <asst>         → KV: [sys, user1, asst1, user2]
  Turn N: </asst> + userN + <asst>         → KV: [..., asst(N-1), userN]

Only the NEW delta is sent each turn — the KV cache retains prior context.
Call reset_dialog() (POST /v1/models/{id}/reset_dialog) to start fresh.
"""

import json
import os
import queue
import sys
import threading
from typing import Dict, Iterator, List, Optional, Tuple

# Add project root to path so we can import from backends/ and utils/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import backends.genie_wrapper as _gw
from utils.inference_plugin import InferencePlugin
from utils.logger import get_logger
from utils.prompt_formatter import PromptFormatter

logger = get_logger(__name__)

_SENTINEL = object()

_PERF_POLICY_MAP = {
    "burst":                       10,
    "sustained_high_performance":  20,
    "high_performance":            30,
    "balanced":                    40,
    "low_balanced":                50,
    "high_power_saver":            60,
    "power_saver":                 70,
    "low_power_saver":             80,
    "extreme_power_saver":         90,
}

_STOP_SEQUENCE_SHAPES = [
    lambda seqs: {"stop-sequence": seqs},
    lambda seqs: {"dialog": {"stop-sequence": seqs}},
    lambda seqs: seqs,
]


class GeniePlugin(InferencePlugin):
    """
    Thin plugin wrapper for Qualcomm Genie SDK LLM inference.

    Lifecycle
    ---------
    load()                    → initialize Genie library + create GenieDialog
    generate_with_messages()  → multi-turn: send delta, maintain KV cache
    generate()                → single-turn fallback (not used when above present)
    reset_dialog()            → clear KV cache, start fresh conversation
    unload()                  → explicitly free dialog then config (this is correct teardown order)
    """

    # Load method

    def load(self, model_id: str, model_config: dict) -> None:
        
        self.model_id      = model_id
        self.system_prompt = model_config.get("system_prompt", "You are a helpful AI assistant.")
        self.formatter     = PromptFormatter()
        self.tokenizer     = None
        self._config       = None
        self._dialog       = None
        self.chat_template = None
        self._default_stop_sequences: List[str] = []
        self._prompt_source = "none — generic fallback"
        self._stop_sequence_shape: Optional[int] = None  # index into _STOP_SEQUENCE_SHAPES once discovered
        self._turn_count              = 0
        self._prev_assistant_response = ""

        # Initialize Genie library 
        model_dir    = model_config.get("genie_model_dir")
        lib_path     = model_config.get("genie_lib_path")
        lib_dirs     = model_config.get("genie_lib_dirs") or []
        hexagon_dirs = model_config.get("genie_hexagon_dirs") or []

        if isinstance(lib_dirs, str):
            lib_dirs = [lib_dirs]
        if isinstance(hexagon_dirs, str):
            hexagon_dirs = [hexagon_dirs]

        # genie_wrapper.initialize() is idempotent — safe to call every load()
        _gw.initialize(
            model_dir=model_dir,
            lib_path=lib_path,
            lib_dirs=lib_dirs,
            hexagon_dirs=hexagon_dirs,
        )
        logger.info(f"[GeniePlugin:{model_id}] Genie library initialized")

        genie_config_path = model_config.get("genie_config")
        config_data = {} # Empty JSON object use SDK's own defaults.
        config_dir  = os.path.abspath(model_dir) if model_dir else os.getcwd()

        if genie_config_path:
            if model_dir and not os.path.isabs(genie_config_path):
                candidate = os.path.join(os.path.abspath(model_dir), genie_config_path)
                if os.path.exists(candidate):
                    genie_config_path = candidate

            config_path = os.path.abspath(genie_config_path)
            config_dir  = os.path.dirname(config_path)

            if not os.path.exists(config_path):
                logger.warning(
                    f"[GeniePlugin:{model_id}] genie_config not found: {config_path} — proceeding without it (SDK defaults will be used)"
                )
            else:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    config_data = _resolve_paths(config_data, config_dir)
                    logger.info(f"[GeniePlugin:{model_id}] Loaded genie_config: {config_path}")
                except Exception as cfg_err:
                    logger.warning(
                        f"[GeniePlugin:{model_id}] Failed to load genie_config {config_path}: {cfg_err!r} — proceeding without it"
                    )
                    config_data = {}
        else:
            logger.info(
                f"[GeniePlugin:{model_id}] No genie_config specified — using SDK defaults (set genie_config in models.yaml for custom settings)"
            )

        config_json = json.dumps(config_data)

        # Create GenieDialogConfig + GenieDialog 
        self._config = _gw.GenieDialogConfig(config_json)
        self._dialog = _gw.GenieDialog(self._config)

        # Performance policy by default burst
        perf_str = (
            config_data.get("performance_policy") or config_data.get("dialog", {}).get("performance_policy") or model_config.get("performance_policy") or "burst"
        )
        policy_val = _PERF_POLICY_MAP.get(perf_str.lower(), _PERF_POLICY_MAP["burst"])
        try:
            self._dialog.set_performance_policy(_gw.GeniePerformancePolicy(policy_val))
            logger.info(f"[GeniePlugin:{model_id}] Performance policy: {perf_str}")
        except Exception as _perf_err:
            logger.warning(
                f"[GeniePlugin:{model_id}] set_performance_policy({perf_str}) FAILED: {_perf_err}. "
                f"NPU may run at reduced performance (DCVS default instead of burst)."
            )

        # Chat template + tokenizer resolution 
        tok_path = model_config.get("tokenizer_path")
        if not tok_path and model_dir:
            candidate = os.path.join(os.path.abspath(model_dir), "tokenizer.json")
            if os.path.exists(candidate):
                tok_path = os.path.abspath(model_dir)
                logger.info(f"[GeniePlugin:{model_id}] tokenizer.json found in genie_model_dir")
        if not tok_path:
            tok_from_cfg = (
                config_data.get("tokenizer", {}).get("path")
                or config_data.get("tokenizer_path")
            )
            if tok_from_cfg:
                resolved = tok_from_cfg if os.path.isabs(tok_from_cfg) else os.path.normpath(
                    os.path.join(config_dir, tok_from_cfg)
                )
                if os.path.exists(resolved):
                    tok_path = resolved
                    logger.info(f"[GeniePlugin:{model_id}] Tokenizer from genie_config.json: {tok_path}")

        # Resolve the actual chat template + default stop sequence 
        override_template = model_config.get("chat_template")
        if override_template:
            self.chat_template = override_template
            self._prompt_source = "models.yaml override"
            logger.info(f"[GeniePlugin:{model_id}] Using chat_template override from models.yaml")
        else:
            tok_cfg_path = _find_tokenizer_config_file(tok_path)
            if tok_cfg_path:
                try:
                    template, stops = _load_chat_template_and_stop(tok_cfg_path)
                    if template:
                        self.chat_template = template
                        self._prompt_source = "tokenizer_config.json"
                        logger.info(f"[GeniePlugin:{model_id}] chat_template loaded from {tok_cfg_path}")
                    else:
                        logger.warning(
                            f"[GeniePlugin:{model_id}] {tok_cfg_path} has no 'chat_template' field — "
                            f"falling back to generic prompt formatting. Tool calling and multi-turn "
                            f"prompts will not match this model's real training format. Add a "
                            f"'chat_template' field to that file, or set 'chat_template'/'tokenizer_path' "
                            f"in models.yaml — see docs/CONFIGURATION.md."
                        )
                    self._default_stop_sequences = stops
                except Exception as tok_err:
                    logger.warning(
                        f"[GeniePlugin:{model_id}] Failed to read {tok_cfg_path}: {tok_err!r} — "
                        f"falling back to generic prompt formatting."
                    )
            else:
                logger.warning(
                    f"[GeniePlugin:{model_id}] No tokenizer_config.json found (looked under "
                    f"'{tok_path or model_dir}') and no 'chat_template' set in models.yaml — "
                    f"falling back to generic prompt formatting. Tool calling and multi-turn prompts "
                    f"will not match this model's real training format. Place a tokenizer_config.json "
                    f"(with a 'chat_template' field) under genie_model_dir, or set "
                    f"'chat_template'/'tokenizer_path' in models.yaml — see docs/CONFIGURATION.md."
                )

        if self._default_stop_sequences:
            logger.info(f"[GeniePlugin:{model_id}] Default stop sequence: {self._default_stop_sequences}")
            self._apply_stop_sequence(None)

        logger.info(f"[GeniePlugin:{model_id}] Ready")

    # Genie API way: single-thread inference

    def generate_with_messages_cb(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
        on_token=None,
        tools: Optional[List[Dict]] = None,
        tool_choice=None,
    ) -> None:
        if self._dialog is None:
            raise RuntimeError(f"[GeniePlugin:{self.model_id}] Not loaded")

        self._apply_stop_sequence(stop)
        prompt, needs_reset = self._build_prompt(messages, tools=tools)

        if needs_reset:
            self._do_reset()

        logger.info(
            f"[GeniePlugin:{self.model_id}] Turn {self._turn_count + 1} "
            f"(fast-path cb) — {'tool-aware full prompt' if tools else 'delta'} {len(prompt)} chars"
        )

        generated_tokens: List[str] = []
        _tokens_generated = [0]

        def _callback(response: str, sentence_code: _gw.GenieDialogSentenceCode, user_data):
            if response:
                generated_tokens.append(response)
                _tokens_generated[0] += 1
                if on_token is not None:
                    on_token(response)
                if max_tokens and _tokens_generated[0] >= max_tokens:
                    try:
                        self._dialog.signal(_gw.GenieDialogAction.ABORT)
                    except Exception as _sig_err:
                        logger.warning(
                            f"[GeniePlugin:{self.model_id}] signal(ABORT) failed: {_sig_err}"
                        )

        try:
            self._dialog.set_max_num_tokens(max_tokens)
        except Exception as e:
            logger.warning(
                f"[GeniePlugin:{self.model_id}] set_max_num_tokens({max_tokens}) "
                f"failed (will rely on signal(ABORT) fallback): {e}"
            )

        # dialog.query() blocks until generation is complete.
        # _callback fires synchronously from within the C++ decode loop.
        # No thread spawn, no queue — this is the lowest-latency path.
        self._dialog.query(prompt, _gw.GenieDialogSentenceCode.COMPLETE, _callback)

        self._turn_count += 1
        self._prev_assistant_response = "".join(generated_tokens)

    # Multi-turn generate

    def generate_with_messages(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice=None,
    ) -> Iterator[str]:
        """
        Multi-turn inference using Genie KV cache.

        Called by plugin_backend.py instead of generate() when this method exists.
        The KV cache accumulates context — only new tokens are processed each turn.
        """
        if self._dialog is None:
            raise RuntimeError(f"[GeniePlugin:{self.model_id}] Not loaded")

        self._apply_stop_sequence(stop)
        prompt, needs_reset = self._build_prompt(messages, tools=tools)

        if needs_reset:
            self._do_reset()

        logger.info(
            f"[GeniePlugin:{self.model_id}] Turn {self._turn_count + 1} — "
            f"{'tool-aware full prompt' if tools else 'delta KV'} {len(prompt)} chars"
        )
        
        # Run inference with the delta (KV cache retains prior context)
        generated_tokens: List[str] = []
        for token in self._run_query(prompt, max_tokens=max_tokens):
            generated_tokens.append(token)
            yield token

        # Update multi-turn state after successful generation
        self._turn_count += 1
        self._prev_assistant_response = "".join(generated_tokens)

    # Single-turn generate (normal with Genie API's way)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        
        if self._dialog is None:
            raise RuntimeError(f"[GeniePlugin:{self.model_id}] Not loaded")
        self._apply_stop_sequence(stop)
        logger.info(f"[GeniePlugin:{self.model_id}] generate() (single-turn), prompt={len(prompt)} chars")
        yield from self._run_query(prompt, max_tokens=max_tokens)

    # Prompt building helpers

    def _apply_stop_sequence(self, stop: Optional[List[str]]) -> None:
        effective = stop or self._default_stop_sequences
        if not effective:
            try:
                self._dialog.set_stop_sequence(None)
            except Exception as exc:
                logger.warning(f"[GeniePlugin:{self.model_id}] set_stop_sequence(None) failed: {exc}")
            return

        if self._stop_sequence_shape is not None:
            shape_fn = _STOP_SEQUENCE_SHAPES[self._stop_sequence_shape]
            try:
                self._dialog.set_stop_sequence(json.dumps(shape_fn(effective)))
                return
            except Exception as exc:
                logger.debug(
                    f"[GeniePlugin:{self.model_id}] previously-working stop-sequence shape "
                    f"#{self._stop_sequence_shape} failed this time ({exc}) — re-probing"
                )
                self._stop_sequence_shape = None

        for idx, shape_fn in enumerate(_STOP_SEQUENCE_SHAPES):
            try:
                self._dialog.set_stop_sequence(json.dumps(shape_fn(effective)))
                self._stop_sequence_shape = idx
                logger.info(
                    f"[GeniePlugin:{self.model_id}] set_stop_sequence() JSON shape #{idx} accepted "
                    f"(auto-detected at runtime — SDK docs don't fully specify the schema)"
                )
                return
            except Exception as exc:
                logger.debug(f"[GeniePlugin:{self.model_id}] stop-sequence shape #{idx} rejected: {exc}")

        logger.warning(
            f"[GeniePlugin:{self.model_id}] set_stop_sequence({effective}) failed for all known JSON shapes." 
        )

    def get_prompt_diagnostics(self) -> Dict:
        """Surfaced via GET /status so a missing/broken chat template is visible without having to dig through startup logs."""
        return {
            "chat_template_loaded": self.chat_template is not None,
            "prompt_source": self._prompt_source,
            "default_stop_sequences": self._default_stop_sequences,
            "stop_sequence_applied": self._stop_sequence_shape is not None,
        }

    def _build_prompt(self, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """
        Return (prompt_str, needs_reset).

        - No tools, no tool history: use KV-cache delta (fast path, no reset needed).
        - Tools present, or the conversation has used tools at some earlier turn:
          send the full conversation as one prompt so that tool-result messages
          (role=tool) appear in context. Reset the dialog first to avoid
          KV-cache contamination from a prior turn.

          The "used tools at some earlier turn" check matters even when this
          particular call has no tools: once a tool round-trip happens, the
          actual KV cache holds the full tool-aware prompt from that turn, not
          the simple "system + placeholder-user + prev_assistant_response"
          shape get_genie_delta() assumes. Staying on the full-prompt path for
          the rest of the conversation avoids that mismatch entirely.
        """
        used_tools_before = any(
            m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("tool_calls"))
            for m in messages
        )
        if tools or used_tools_before:
            # Full-context path — format the entire conversation in one shot.
            prompt = self.formatter.format(
                messages,
                tokenizer=self.tokenizer,
                template=self.chat_template,
                default_system=self.system_prompt,
                add_generation_prompt=True,
                tools=tools,
            )
            return prompt, True  # caller must reset dialog before querying

        # Normal KV-cache delta path (unchanged from original logic).
        last_user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        is_first_turn = (self._turn_count == 0)
        try:
            delta = self.formatter.get_genie_delta(
                new_user_message=last_user_msg,
                prev_assistant_response=self._prev_assistant_response,
                tokenizer=self.tokenizer,
                template=self.chat_template,
                is_first_turn=is_first_turn,
                system_prompt=self.system_prompt,
            )
        except Exception as _delta_err:
            logger.error(
                f"[GeniePlugin:{self.model_id}] get_genie_delta() failed: {_delta_err!r}",
                exc_info=True,
            )
            delta = last_user_msg
        return delta, False

    def _do_reset(self) -> None:
        """Reset the Genie dialog KV cache without clearing turn tracking."""
        if self._dialog is not None:
            try:
                self._dialog.reset()
                logger.debug(f"[GeniePlugin:{self.model_id}] KV cache reset for tool-aware turn")
            except Exception as exc:
                logger.warning(f"[GeniePlugin:{self.model_id}] dialog.reset() failed: {exc}")


    def _run_query(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """
        Send a prompt to GenieDialog and yield tokens.
        Used by both generate() and generate_with_messages().
        """
        token_queue: queue.Queue = queue.Queue()
        _tokens_generated = [0]  # mutable list so nested callback can mutate it

        def _callback(response: str, sentence_code: _gw.GenieDialogSentenceCode, user_data):
            if response:
                token_queue.put(response)
                _tokens_generated[0] += 1
                if max_tokens and _tokens_generated[0] >= max_tokens:
                    try:
                        self._dialog.signal(_gw.GenieDialogAction.ABORT)
                    except Exception as _sig_err:
                        logger.warning(
                            f"[GeniePlugin:{self.model_id}] signal(ABORT) failed: {_sig_err}"
                        )

        def _run():
            try:
                try:
                    self._dialog.set_max_num_tokens(max_tokens)
                    logger.debug(
                        f"[GeniePlugin:{self.model_id}] set_max_num_tokens({max_tokens}) OK"
                    )
                except Exception as e:
                    logger.warning(
                        f"[GeniePlugin:{self.model_id}] set_max_num_tokens({max_tokens}) "
                        f"failed (will rely on signal(ABORT) fallback): {e}"
                    )
                self._dialog.query(prompt, _gw.GenieDialogSentenceCode.COMPLETE, _callback)
            except Exception as exc:
                logger.error(f"[GeniePlugin:{self.model_id}] dialog.query() raised: {exc}")
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)


        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        try:
            while True:
                item = token_queue.get(timeout=120)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            thread.join(timeout=5)

    # Reset dialog 
    def reset_dialog(self) -> None:
        """ Reset dialog KV cache to start a fresh conversation. """
        if self._dialog is not None:
            try:
                self._dialog.reset()
                self._turn_count              = 0
                self._prev_assistant_response = ""
                logger.info(f"[GeniePlugin:{self.model_id}] Dialog reset — new conversation")
            except Exception as exc:
                logger.warning(f"[GeniePlugin:{self.model_id}] reset_dialog failed: {exc}")

    # Unload 
    def unload(self) -> None:
        """ Free all Genie resources """
        logger.info(f"[GeniePlugin:{self.model_id}] Unloading...")

        if self._dialog is not None:
            try:
                self._dialog.free()
                logger.debug(f"[GeniePlugin:{self.model_id}] GenieDialog freed")
            except Exception as exc:
                logger.warning(f"[GeniePlugin:{self.model_id}] dialog.free() error: {exc}")
            finally:
                self._dialog = None

        if self._config is not None:
            try:
                self._config.free()
                logger.debug(f"[GeniePlugin:{self.model_id}] GenieDialogConfig freed")
            except Exception as exc:
                logger.warning(f"[GeniePlugin:{self.model_id}] config.free() error: {exc}")
            finally:
                self._config = None

        self.tokenizer = None
        logger.info(f"[GeniePlugin:{self.model_id}] Unloaded")



# Helper methods
def _resolve_paths(obj, base_dir: str):
    if isinstance(obj, dict):
        return {k: _resolve_paths(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_paths(item, base_dir) for item in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if (
            stripped
            and not os.path.isabs(stripped)
            and (os.sep in stripped or "/" in stripped or "." in os.path.basename(stripped))
        ):
            candidate = os.path.normpath(os.path.join(base_dir, stripped))
            if os.path.exists(candidate):
                return candidate
    return obj

def _find_tokenizer_config_file(tok_path: Optional[str]) -> Optional[str]:
    """Locate tokenizer_config.json given a resolved tokenizer path."""
    if not tok_path:
        return None
    if os.path.isdir(tok_path):
        candidate = os.path.join(tok_path, "tokenizer_config.json")
        return candidate if os.path.exists(candidate) else None
    if os.path.isfile(tok_path) and tok_path.endswith(".json"):
        return tok_path
    return None


def _load_chat_template_and_stop(tokenizer_config_path: str) -> Tuple[Optional[str], List[str]]:
    """
    Read chat_template + a default end-of-turn stop sequence from a tokenizer_config.json file.

    Returns (chat_template, stop_sequences) — either half may be empty if
    the file doesn't have that field. Only eos_token is used for the default
    stop sequence (not the full additional_special_tokens list), several of
    those tokens are things like <tool_call>'s opening tag, which we must
    NOT stop on mid-generation.
    """
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    chat_template = tok_cfg.get("chat_template")
    if not isinstance(chat_template, str):
        chat_template = None

    stops: List[str] = []
    eos = tok_cfg.get("eos_token")
    if isinstance(eos, str):
        stops.append(eos)
    elif isinstance(eos, dict) and isinstance(eos.get("content"), str):
        stops.append(eos["content"])

    return chat_template, stops