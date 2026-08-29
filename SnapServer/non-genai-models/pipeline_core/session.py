# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import threading

_qnn_lock = threading.Lock()
_qnn_registered = False
_LIB_NAME = "QNNExecutionProvider"


def create_session(model_path: str, use_qnn: bool = False,
                   perf_mode: str = "burst", rpc_control_latency: str = "100"):
    """
    Create an ORT InferenceSession using the ORT 2.x API.

    Returns (session, run_options).
    run_options is None when not using QNN — pass it to session.run() regardless,
    ORT accepts None there.
    """
    import onnxruntime as ort

    if not use_qnn:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session, None

    import onnxruntime_qnn as qnn_ep

    # Register once per process; safe to call from multiple plugins concurrently.
    global _qnn_registered
    with _qnn_lock:
        if not _qnn_registered:
            ort.register_execution_provider_library(_LIB_NAME, qnn_ep.get_library_path())
            _qnn_registered = True

    all_devices      = ort.get_ep_devices()
    selected_devices = [d for d in all_devices if d.ep_name == _LIB_NAME]

    ep_options      = {"backend_path": qnn_ep.get_qnn_htp_path()}
    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(selected_devices, ep_options)

    session = ort.InferenceSession(model_path, sess_options=session_options)

    run_options = ort.RunOptions()
    run_options.add_run_config_entry("qnn.perf_mode", perf_mode)
    run_options.add_run_config_entry("qnn.rpc_control_latency", rpc_control_latency)

    return session, run_options
