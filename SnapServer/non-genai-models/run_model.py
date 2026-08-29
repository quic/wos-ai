# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Standalone model runner — no server, no OpenAI API.

Usage:
    python run_model.py --model yamnet          --input audio.wav
    python run_model.py --model whisper-base    --input audio.wav
    python run_model.py --model zipformer       --input audio.wav
    python run_model.py --model inception-v3    --input photo.jpg
    python run_model.py --model mobilenet-v2    --input photo.jpg
    python run_model.py --model yolov3          --input photo.jpg
    python run_model.py --model centernet-pose  --input photo.jpg
    python run_model.py --model unet-segmentation --input photo.jpg
    python run_model.py --model quicksrnet-small  --input photo.jpg
    python run_model.py --model easyocr           --input photo.jpg

Set PREPOST_CODEBASE and MODEL_BASE env vars, or pass --model-base / --codebase.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import yaml


def _expand(value: str, env: dict) -> str:
    for k, v in env.items():
        value = value.replace(f"${{{k}}}", v)
    return value


def _load_models_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)
    env = {**os.environ}
    def _walk(obj):
        if isinstance(obj, str):
            return _expand(obj, env)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(i) for i in obj]
        return obj
    return _walk(raw)


def _load_plugin(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("_plugin", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def main():
    parser = argparse.ArgumentParser(description="Run a model locally without the WoS server.")
    parser.add_argument("--model",      required=True,  help="Model ID from models_prepost.yaml")
    parser.add_argument("--input",      required=True,  help="Path to input file (audio or image)")
    parser.add_argument("--models-yaml", default=None,  help="Path to models_prepost.yaml (auto-detected if omitted)")
    parser.add_argument("--codebase",   default=None,   help="Override PREPOST_CODEBASE")
    parser.add_argument("--model-base", default=None,   help="Override MODEL_BASE")
    parser.add_argument("--mask",        default=None,   help="Path to mask image (white = inpaint, black = keep). Used by inpainting models.")
    parser.add_argument("--wos-dir",    default=None,   help="Path to WoS server root (for utils.inference_plugin)")
    parser.add_argument("--cpu",        action="store_true", help="Force CPUExecutionProvider (disable QNN)")
    parser.add_argument("--qnn",        action="store_true", help="Force QNNExecutionProvider (enable QNN)")
    args = parser.parse_args()

    # Allow env-var overrides from CLI flags
    if args.codebase:
        os.environ["PREPOST_CODEBASE"] = args.codebase
    if args.model_base:
        os.environ["MODEL_BASE"] = args.model_base

    codebase_dir = os.environ.get(
        "PREPOST_CODEBASE",
        os.path.dirname(os.path.abspath(__file__))
    )
    # Ensure ${PREPOST_CODEBASE} and ${MODEL_BASE} expand even without env vars set
    os.environ.setdefault("PREPOST_CODEBASE", codebase_dir)
    os.environ.setdefault("MODEL_BASE", os.path.join(os.path.dirname(codebase_dir), "XElite_models"))
    sys.path.insert(0, codebase_dir)

    # Add the WoS server directory so plugins can import utils.inference_plugin
    wos_dir = args.wos_dir or os.path.join(
        os.path.dirname(codebase_dir),
        "WoS_ServerClientIntegration-main",
        "WoS_ServerClientIntegration-main",
    )
    if os.path.isdir(wos_dir):
        sys.path.insert(0, wos_dir)

    yaml_path = args.models_yaml or os.path.join(codebase_dir, "configs", "models_prepost.yaml")
    config    = _load_models_yaml(yaml_path)

    model_cfg = next(
        (m for m in config.get("models", []) if m["id"] == args.model),
        None
    )
    if model_cfg is None:
        ids = [m["id"] for m in config.get("models", [])]
        print(f"Unknown model '{args.model}'. Available: {', '.join(ids)}", file=sys.stderr)
        sys.exit(1)

    if model_cfg.get("backend") != "plugin":
        print(f"Model '{args.model}' uses backend '{model_cfg.get('backend')}', not 'plugin'. "
              "Only plugin-backend models are supported by this runner.", file=sys.stderr)
        sys.exit(1)

    if args.cpu:
        model_cfg["use_qnn"] = False
    if args.qnn:
        model_cfg["use_qnn"] = True

    PluginClass = _load_plugin(model_cfg["plugin_module"], model_cfg["plugin_class"])
    plugin      = PluginClass()
    plugin.load(args.model, model_cfg)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "rb") as fh:
        data = fh.read()

    # Dispatch to the right method based on what the plugin implements
    result = None

    if hasattr(plugin, "transcribe") and _is_audio(input_path):
        result = plugin.transcribe(data)

    elif hasattr(plugin, "image_variation") and _is_video(input_path):
        result = plugin.image_variation(data)

    elif hasattr(plugin, "generate") and _is_image(input_path):
        import base64
        tokens = list(plugin.generate(base64.b64encode(data).decode()))
        result = "".join(tokens)

    elif hasattr(plugin, "image_variation") and _is_image(input_path):
        kwargs = {}
        if args.mask:
            mask_path = os.path.abspath(args.mask)
            if not os.path.exists(mask_path):
                print(f"Mask file not found: {mask_path}", file=sys.stderr)
                sys.exit(1)
            with open(mask_path, "rb") as fh:
                kwargs["mask"] = fh.read()
        result = plugin.image_variation(data, **kwargs)

    else:
        print(f"Cannot determine how to run '{args.model}' with input '{input_path}'.\n"
              "Implement a matching plugin method or check --input file type.", file=sys.stderr)
        sys.exit(1)

    print(result)
    plugin.unload()


def _is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


if __name__ == "__main__":
    main()
