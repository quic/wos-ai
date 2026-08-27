# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Genie C API — Python ctypes bindings : Wraps Qualcomm Genie C API via ctypes.
"""

import ctypes
import logging
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE, POINTER, Structure, byref,
    c_char_p, c_int32, c_uint32, c_void_p,
)
from enum import IntEnum
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level library handle (populated by initialize())
_genie_lib: Optional[ctypes.CDLL] = None

# Keep os.add_dll_directory() handles alive (they are context managers)
_dll_dir_handles: List[Any] = []


# Path utilities
def _clean_path(p: str) -> str:
    """Strip Python r"..." syntax accidentally pasted into YAML."""
    if not p:
        return p
    p = p.strip()
    # Strip r"..." or r'...'
    if len(p) >= 3 and p[0].lower() == 'r' and p[1] in ('"', "'"):
        quote = p[1]
        if p.endswith(quote):
            p = p[2:-1]
    # Strip plain surrounding quotes
    elif len(p) >= 2 and p[0] in ('"', "'") and p[-1] == p[0]:
        p = p[1:-1]
    return p


def _clean_paths(paths) -> List[str]:
    if isinstance(paths, str):
        paths = [paths]
    return [_clean_path(p) for p in (paths or []) if p]



# Genie Status codes
GENIE_STATUS_SUCCESS                  =  0
GENIE_STATUS_WARNING_CONTEXT_EXCEEDED =  4
GENIE_STATUS_ERROR_GENERAL            = -1

Genie_Status_t = c_int32


# Enums
class GeniePerformancePolicy(IntEnum):
    BURST                      = 10
    SUSTAINED_HIGH_PERFORMANCE = 20
    HIGH_PERFORMANCE           = 30
    BALANCED                   = 40
    LOW_BALANCED               = 50
    HIGH_POWER_SAVER           = 60
    POWER_SAVER                = 70
    LOW_POWER_SAVER            = 80
    EXTREME_POWER_SAVER        = 90


class GenieDialogAction(IntEnum):
    ABORT = 0x01
    PAUSE = 0x02


class GenieDialogSentenceCode(IntEnum):
    COMPLETE = 0
    BEGIN    = 1
    CONTINUE = 2
    END      = 3
    ABORT    = 4
    REWIND   = 5
    RESUME   = 6


class GenieDialogPriority(IntEnum):
    LOW         = 0
    NORMAL      = 100
    NORMAL_HIGH = 150
    HIGH        = 200



# Opaque handle structures
class _GenieDialogConfig_Handle_t(Structure): pass
class _GenieDialog_Handle_t(Structure):       pass

GenieDialogConfig_Handle_t = POINTER(_GenieDialogConfig_Handle_t)
GenieDialog_Handle_t       = POINTER(_GenieDialog_Handle_t)

# Callback: void cb(const char* response, int32_t sentence_code, void* user_data)
GenieDialog_QueryCallback_t = CFUNCTYPE(None, c_char_p, c_int32, c_void_p)



# Exception catching methods
class GenieError(Exception):
    def __init__(self, status: int, message: str = ""):
        self.status = status
        super().__init__(message or f"GenieError status={status}")


def _check(status: int, allow_warnings: bool = False, context: str = ""):
    if status is None:
        return  
    if status < 0:
        raise GenieError(status, f"{context} status={status}" if context else f"status={status}")
    if status == GENIE_STATUS_WARNING_CONTEXT_EXCEEDED:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"Genie context length exceeded (status=4). "
            f"Call reset_dialog() to start a fresh conversation. {context}"
        )
    elif status > 0 and not allow_warnings:
        raise GenieError(status, f"{context} status={status}" if context else f"status={status}")


# Library initialization
def _setup_argtypes(lib: ctypes.CDLL):
    """Bind argtypes/restype for all Genie C functions."""
    # Version
    lib.Genie_getApiMajorVersion.argtypes = []
    lib.Genie_getApiMajorVersion.restype  = c_uint32
    lib.Genie_getApiMinorVersion.argtypes = []
    lib.Genie_getApiMinorVersion.restype  = c_uint32
    lib.Genie_getApiPatchVersion.argtypes = []
    lib.Genie_getApiPatchVersion.restype  = c_uint32

    # DialogConfig — correct camelCase C API names from QAIRT SDK
    lib.GenieDialogConfig_createFromJson.argtypes = [c_char_p, POINTER(GenieDialogConfig_Handle_t)]
    lib.GenieDialogConfig_createFromJson.restype  = Genie_Status_t
    lib.GenieDialogConfig_free.argtypes = [GenieDialogConfig_Handle_t]
    lib.GenieDialogConfig_free.restype  = Genie_Status_t

    # Dialog — correct camelCase C API names from QAIRT SDK
    lib.GenieDialog_create.argtypes = [GenieDialogConfig_Handle_t, POINTER(GenieDialog_Handle_t)]
    lib.GenieDialog_create.restype  = Genie_Status_t
    lib.GenieDialog_query.argtypes  = [GenieDialog_Handle_t, c_char_p, c_int32,
                                        GenieDialog_QueryCallback_t, c_void_p]
    lib.GenieDialog_query.restype   = Genie_Status_t
    lib.GenieDialog_reset.argtypes  = [GenieDialog_Handle_t]
    lib.GenieDialog_reset.restype   = Genie_Status_t
    lib.GenieDialog_setMaxNumTokens.argtypes = [GenieDialog_Handle_t, c_uint32]
    lib.GenieDialog_setMaxNumTokens.restype  = Genie_Status_t
    lib.GenieDialog_setStopSequence.argtypes = [GenieDialog_Handle_t, c_char_p]
    lib.GenieDialog_setStopSequence.restype  = Genie_Status_t
    lib.GenieDialog_setPerformancePolicy.argtypes = [GenieDialog_Handle_t, c_int32]
    lib.GenieDialog_setPerformancePolicy.restype  = Genie_Status_t
    lib.GenieDialog_signal.argtypes = [GenieDialog_Handle_t, c_int32]
    lib.GenieDialog_signal.restype  = Genie_Status_t
    lib.GenieDialog_free.argtypes   = [GenieDialog_Handle_t]
    lib.GenieDialog_free.restype    = Genie_Status_t


def _add_dll_dirs(dirs: List[str]) -> None:
    """
    Register directories for DLL dependency resolution on Windows.
    """
    global _dll_dir_handles
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return  # Python < 3.8 — PATH modification is sufficient
    for d in dirs:
        if os.path.isdir(d):
            try:
                handle = os.add_dll_directory(d)
                _dll_dir_handles.append(handle)
            except Exception:
                pass


def initialize(
    model_dir: Optional[str] = None,
    lib_path: Optional[str] = None,
    lib_dirs: Optional[List[str]] = None,
    hexagon_dirs: Optional[List[str]] = None,
) -> None:
    """
    Load the Genie shared library and set up ctypes bindings.
    """
    global _genie_lib

    model_dir    = _clean_path(model_dir)    if model_dir    else None
    lib_path     = _clean_path(lib_path)     if lib_path     else None
    lib_dirs     = _clean_paths(lib_dirs)
    hexagon_dirs = _clean_paths(hexagon_dirs)

    # Apply model_dir shortcut
    if model_dir:
        model_dir = os.path.abspath(model_dir)
        if not lib_dirs:
            lib_dirs = [model_dir]
        elif model_dir not in lib_dirs:
            lib_dirs = [model_dir] + lib_dirs
        if not hexagon_dirs:
            hexagon_dirs = [model_dir]
        elif model_dir not in hexagon_dirs:
            hexagon_dirs = [model_dir] + hexagon_dirs

    # ── Register DLL directories 
    all_dirs = list(lib_dirs) + [d for d in hexagon_dirs if d not in lib_dirs]
    _add_dll_dirs(all_dirs)

    system = platform.system()

    # update PATH / LD_LIBRARY_PATH (needed in linux or old python)
    if lib_dirs:
        _prepend_env("PATH" if system == "Windows" else "LD_LIBRARY_PATH", lib_dirs)

    # Set ADSP_LIBRARY_PATH 
    if hexagon_dirs:
        valid = [d for d in hexagon_dirs if os.path.isdir(d)]
        if valid:
            existing = os.environ.get("ADSP_LIBRARY_PATH", "")
            new_dirs = [d for d in valid if d not in existing]
            if new_dirs:
                os.environ["ADSP_LIBRARY_PATH"] = (
                    os.pathsep.join(new_dirs) +
                    (os.pathsep + existing if existing else "")
                )

    # DLL is loaded once — skip if already initialised
    if _genie_lib is not None:
        return

    # Build candidate paths
    candidates: List[str] = []
    if lib_path:
        candidates.append(os.path.abspath(lib_path))

    lib_name = "Genie.dll" if system == "Windows" else (
        "libGenie.so" if system == "Linux" else "libGenie.dylib"
    )

    for d in lib_dirs:
        candidates.append(os.path.join(d, lib_name))
    candidates.append(lib_name)  # rely on PATH / LD_LIBRARY_PATH

    # Try to load
    last_err: Optional[OSError] = None
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            _setup_argtypes(lib)
            _genie_lib = lib
            return
        except OSError as e:
            last_err = e

    # Build a helpful error message
    err_str = str(last_err) if last_err else ""
    arch_hint = ""
    if "193" in err_str or "not a valid Win32" in err_str:
        import struct
        py_bits = struct.calcsize("P") * 8
        py_arch = platform.machine()
        arch_hint = (
            "\n*** ARCHITECTURE MISMATCH ***\n"
            f"  Your Python is {py_bits}-bit ({py_arch}).\n"
            "  Genie.dll from QAIRT is compiled for ARM64 (aarch64-windows-msvc).\n"
            "  You must run the server with ARM64 Python.\n\n"
            "  Download ARM64 Python from:\n"
            "    https://www.python.org/downloads/windows/\n"
            "  Choose: 'Windows installer (ARM64)'\n\n"
            "  Then recreate your venv with ARM64 Python:\n"
            "    C:\\path\\to\\arm64\\python.exe -m venv venv\n"
            "    venv\\Scripts\\activate\n"
            "    pip install -r requirements.txt\n"
            "\n Or\n"
            "Replace you Genie.dll with ARM64 EC libs (arm64x-windows-msvc)."
        )

    raise OSError(
        f"Could not load Genie library (Genie.dll or libGenie.so).\n"
        f"Tried paths: {candidates}\n"
        f"Last error: {last_err}"
        f"{arch_hint}\n\n"
    )


def _prepend_env(var: str, dirs: List[str]) -> None:
    """Prepend directories to an environment variable."""
    existing = os.environ.get(var, "")
    new_dirs = [d for d in dirs if d and d not in existing]
    if new_dirs:
        os.environ[var] = os.pathsep.join(new_dirs) + (
            os.pathsep + existing if existing else ""
        )


def _require_lib() -> ctypes.CDLL:
    if _genie_lib is None:
        raise RuntimeError(
            "Genie library not initialized. Call genie_wrapper.initialize() first."
        )
    return _genie_lib


# Version
def get_api_version() -> Tuple[int, int, int]:
    lib = _require_lib()
    return (
        lib.Genie_getApiMajorVersion(),
        lib.Genie_getApiMinorVersion(),
        lib.Genie_getApiPatchVersion(),
    )


# GenieDialogConfig
class GenieDialogConfig:
    """Wraps GenieDialogConfig C handle."""

    def __init__(self, json_config: str):
        lib = _require_lib()
        self._handle = GenieDialogConfig_Handle_t()
        status = lib.GenieDialogConfig_createFromJson(
            json_config.encode("utf-8"), byref(self._handle)
        )
        _check(status)

    def free(self) -> None:
        """
        Explicitly free the C handle.  Safe to call multiple times.

        Nulls ``_handle`` after the first call so the subsequent ``__del__``
        becomes a no-op — prevents double-free.

        IMPORTANT: always call ``GenieDialog.free()`` BEFORE calling this,
        because the dialog was created from this config and must be destroyed
        first.
        """
        if _genie_lib and getattr(self, "_handle", None):
            try:
                _genie_lib.GenieDialogConfig_free(self._handle)
            except Exception:
                pass
            finally:
                self._handle = None  # idempotent guard

    def __del__(self):
        self.free()  # safe — free() is idempotent

    @property
    def handle(self):
        return self._handle



# GenieDialog
class GenieDialog:
    """Wraps GenieDialog C handle — maintains KV cache across turns."""

    def __init__(self, config: GenieDialogConfig):
        lib = _require_lib()
        self._handle = GenieDialog_Handle_t()
        status = lib.GenieDialog_create(config._handle, byref(self._handle))
        _check(status)
        self._callbacks: list = []  # prevent GC of ctypes callbacks

    def query(
        self,
        query_str: str,
        sentence_code: GenieDialogSentenceCode,
        callback: Callable[[str, GenieDialogSentenceCode, Any], None],
        user_data: Any = None,
    ) -> None:
        """
        Execute a text query. Blocks until the full response is generated.
        The callback is invoked for each token/sentence chunk.
        """
        lib = _require_lib()

        def _c_cb(response: bytes, code: int, _data: Any):
            text = response.decode("utf-8") if response else ""
            callback(text, GenieDialogSentenceCode(code), user_data)

        self._callbacks.clear() # clearing previous queries - that are not needed
        c_cb = GenieDialog_QueryCallback_t(_c_cb)
        self._callbacks.append(c_cb)  # keep alive

        status = lib.GenieDialog_query(
            self._handle,
            query_str.encode("utf-8"),
            int(sentence_code),
            c_cb,
            None,
        )
        _check(status, allow_warnings=True)

    def reset(self) -> None:
        """Reset dialog state (clears KV cache)."""
        _check(_require_lib().GenieDialog_reset(self._handle))

    def set_max_num_tokens(self, max_tokens: int) -> None:
        _check(_require_lib().GenieDialog_setMaxNumTokens(self._handle, max_tokens))

    def set_stop_sequence(self, stop_sequences: Optional[str]) -> None:
        seq = stop_sequences.encode("utf-8") if stop_sequences else None
        _check(_require_lib().GenieDialog_setStopSequence(self._handle, seq))

    def set_performance_policy(self, policy: GeniePerformancePolicy) -> None:
        _check(_require_lib().GenieDialog_setPerformancePolicy(self._handle, int(policy)))

    def signal(self, action: GenieDialogAction) -> None:
        _check(_require_lib().GenieDialog_signal(self._handle, int(action)))

    def free(self) -> None:
        """
        Explicitly free the C handle and all associated resources.
        """
        # Release ctypes callback references (prevents dangling C pointers)
        if hasattr(self, "_callbacks"):
            self._callbacks.clear()

        # Free the C-level dialog (KV-cache + model weights on NPU/HTP)
        if _genie_lib and getattr(self, "_handle", None):
            try:
                _genie_lib.GenieDialog_free(self._handle)
            except Exception:
                pass
            finally:
                self._handle = None  # idempotent guard

    def __del__(self):
        self.free()  # safe — free() is idempotent

    @property
    def handle(self):
        return self._handle


# Public exports
__all__ = [
    "initialize",
    "get_api_version",
    "GenieDialogConfig",
    "GenieDialog",
    "GenieDialogSentenceCode",
    "GenieDialogAction",
    "GenieDialogPriority",
    "GeniePerformancePolicy",
    "GenieError",
    "GENIE_STATUS_SUCCESS",
    "GENIE_STATUS_WARNING_CONTEXT_EXCEEDED",
    "_clean_path",
    "_clean_paths",
]