# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear


"""
Example: C++ Shared Library Plugin
====================================
Shows how to call a C++ shared library (.dll / .so) from a plugin using ctypes.

The plugin system is Python-facing, but Python can call C/C++ code via:
  1. ctypes   — load any .dll/.so, call C functions directly (shown here)
  2. pybind11 — compile C++ as a Python extension (.pyd/.so)
  3. CFFI     — similar to ctypes with a different API style
  4. subprocess — NOT recommended (breaks multi-turn KV cache)

models.yaml entry example
--------------------------
  - id: my-cpp-llm
    backend: plugin
    plugin_module: examples/example_cpp_plugin.py
    plugin_class: CppInferencePlugin
    lib_path: /path/to/my_inference_engine.dll
    model_path: /path/to/model_weights.bin
    max_tokens: 2048
    system_prompt: "You are a helpful AI assistant."
    owned_by: my-org

C API contract (what your .dll/.so must export)
------------------------------------------------
The example below assumes your C library exports these functions:

    // Initialize the model — call once
    int my_engine_init(const char* model_path, int max_tokens);

    // Generate tokens — call per request or any api's
    // callback is called for each token
    int my_engine_generate(
        const char* prompt,
        int max_tokens,
        void (*callback)(const char* token, void* user_data),
        void* user_data
    );

    // Free resources
    void my_engine_free(void);

Adapt the ctypes signatures below to match your actual C API.

C++ compatibility note
-----------------------
The plugin wrapper is Python-facing, but it is fully compatible with C/C++
code via ctypes.  The Genie SDK itself (backends/genie_wrapper.py) is a
real-world example of this pattern — it wraps a C++ SDK via ctypes.

For pybind11 extensions (.pyd / .so compiled from C++):
  - Build your extension normally: cmake / setup.py / scikit-build
  - Import it in load() like any Python module: import my_cpp_module
  - Call its functions in generate() as normal Python calls
  - No ctypes needed — pybind11 handles the Python/C++ bridge

For ONNX Runtime (C++ inference engine with Python bindings):
  - Use the built-in OnnxQnnBackend (backend: onnx_qnn) for single-pass models
  - Or use PluginBackend with onnxruntime in your plugin for custom pipelines
"""

import ctypes
import os
import queue
import sys
import threading
from typing import Dict, Iterator, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.inference_plugin import InferencePlugin
from utils.logger import get_logger

logger = get_logger(__name__)

_SENTINEL = object()


# ctypes callback type
# Matches C signature: void (*callback)(const char* token, void* user_data)
_TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class CppInferencePlugin(InferencePlugin):
    """
    Plugin that calls a C++ shared library via ctypes.

    The C library is loaded once at startup
    The library path is read from models.yaml (lib_path key).

    For logics KV cache reuse with a C++ library, your library needs to expose:
    - A way to save/restore state (like Qualcomm Genie API's i.e., GenieDialog.save/restore)
    """

    def load(self, model_id: str, model_config: dict) -> None:
        """
        Called ONCE at server startup.
        Load the C++ shared library and initialize the model.
        """
        self.model_id   = model_id
        self.max_tokens = int(model_config.get("max_tokens", 2048))

        lib_path   = model_config.get("lib_path")
        model_path = model_config.get("model_path", "")

        if not lib_path:
            raise ValueError(
                f"[CppPlugin:{model_id}] 'lib_path' is required in models.yaml. "
                "Set it to the path of your C++ shared library (.dll / .so)."
            )

        # Add library directory to PATH so dependencies are found
        lib_dir = os.path.dirname(os.path.abspath(lib_path))
        if lib_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")

        # Load the shared library
        logger.info(f"[CppPlugin:{model_id}] Loading library: {lib_path}")
        self._lib = ctypes.CDLL(lib_path)

        # Declare C function signatures 
        # ADAPT THESE TO MATCH YOUR ACTUAL C API
        self._lib.my_engine_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.my_engine_init.restype  = ctypes.c_int

        self._lib.my_engine_generate.argtypes = [
            ctypes.c_char_p,    # prompt
            ctypes.c_int,       # max_tokens
            _TokenCallback,     # callback(token, user_data)
            ctypes.c_void_p,    # user_data (passed back to callback)
        ]
        self._lib.my_engine_generate.restype = ctypes.c_int

        self._lib.my_engine_free.argtypes = []
        self._lib.my_engine_free.restype  = None

        # Initialize the model
        ret = self._lib.my_engine_init(
            model_path.encode("utf-8"),
            self.max_tokens,
        )
        if ret != 0:
            raise RuntimeError(
                f"[CppPlugin:{model_id}] my_engine_init() returned {ret}. "
                "Check model_path and library compatibility."
            )

        # Keep a reference to callbacks to prevent GC while C code holds them
        self._active_callbacks: list = []

        logger.info(f"[CppPlugin:{model_id}] Ready. max_tokens={self.max_tokens}")

    # Define the API calls needed for your application, generat() is used as example
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:

        token_queue: queue.Queue = queue.Queue()

        # Create ctypes callback 
        # IMPORTANT: keep a reference to prevent garbage collection while the C library holds the function pointer.
        def _c_callback(token_bytes: bytes, user_data) -> None:
            if token_bytes:
                token_queue.put(token_bytes.decode("utf-8", errors="replace"))

        c_cb = _TokenCallback(_c_callback)
        self._active_callbacks.append(c_cb)  # prevent GC

        # Run inference in background thread
        def _run():
            try:
                ret = self._lib.my_engine_generate(
                    prompt.encode("utf-8"),
                    max_tokens,
                    c_cb,
                    None,  # user_data
                )
                if ret != 0:
                    token_queue.put(RuntimeError(
                        f"[CppPlugin:{self.model_id}] my_engine_generate() returned {ret}"
                    ))
            except Exception as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)
                # Remove callback reference after use
                if c_cb in self._active_callbacks:
                    self._active_callbacks.remove(c_cb)

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

    def unload(self) -> None:
        """Called on server shutdown. Free C++ resources."""
        if hasattr(self, "_lib"):
            try:
                self._lib.my_engine_free()
            except Exception:
                pass
        logger.info(f"[CppPlugin:{self.model_id}] Unloaded")


# Alternative: pybind11 extension example

# If your C++ code is compiled as a Python extension (pybind11 / CFFI),
# use this simpler pattern instead:
#
# class Pybind11Plugin(InferencePlugin):
#     def load(self, model_id, model_config):
#         import my_cpp_module  # your compiled .pyd / .so
#         self.engine = my_cpp_module.Engine(model_config["model_path"])
#
#     def generate(self, prompt, max_tokens=512, temperature=1.0, stop=None):
#         # If your C++ engine has a streaming API:
#         for token in self.engine.generate_stream(prompt, max_tokens):
#             yield token
#
#         # If your C++ engine returns the full response at once:
#         # response = self.engine.generate(prompt, max_tokens)
#         # yield response
#
#     def unload(self):
#         del self.engine