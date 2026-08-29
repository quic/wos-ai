# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .inspector import OnnxModelInspector


_AUDIO_INPUT_TYPES = {"log_mel", "mfcc", "waveform", "tokens", "kaldi_fbank"}
_IMAGE_INPUT_TYPES = {
    "object_detection", "pose_estimation", "classification", "segmentation",
    "ocr_detection", "super_resolution", "inpainting", "video_classification",
    "denoising", "colorization",
}

_AUDIO_OUTPUT_TYPES = {
    "ctc_greedy", "ctc_beam", "attention_greedy", "attention_beam",
    "softmax_top_k", "waveform_reconstruction", "transducer_greedy",
}
_IMAGE_OUTPUT_TYPES = {
    "detection", "classification", "pose", "segmentation",
    "mask_list", "super_resolution", "inpainting", "denoising", "colorization",
    "multiclass_mask",
}


class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self._config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r", encoding="utf-8") as fh:
            if config_path.endswith(".json"):
                raw = json.load(fh)
                self._params = _flatten_metadata_json(raw)
            else:
                self._params = yaml.safe_load(fh) or {}
        # Resolve relative file paths against the config file's directory
        for key in ("vocab_file", "labels_file"):
            val = self._params.get(key)
            if val and not os.path.isabs(val):
                self._params[key] = os.path.join(self._config_dir, val)
        for required in ("input_type", "output_type"):
            if required not in self._params:
                raise ValueError(
                    f"Config '{config_path}' is missing required field '{required}'. "
                    f"Add '{required}: <value>' to the config file."
                )
        if "modality" not in self._params:
            it = self._params.get("input_type", "")
            if it in _AUDIO_INPUT_TYPES:
                self._params["modality"] = "audio"
            elif it in _IMAGE_INPUT_TYPES:
                self._params["modality"] = "image"
            else:
                raise ValueError(
                    f"Config '{config_path}' is missing required field 'modality'. "
                    "Add 'modality: audio' or 'modality: image'."
                )

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def get(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)


def _flatten_metadata_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata.json format to flat config dict."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k != "model_files":
            out[k] = v
    return out


@dataclass
class ModelConfig:
    # From OnnxModelInspector
    input_name: str
    input_shape: List[Optional[int]]
    input_dtype: str
    output_name: str
    output_shape: List[Optional[int]]
    output_dtype: str
    metadata_props: Dict[str, str] = field(default_factory=dict)

    # From ConfigLoader
    modality: str = ""
    input_type: str = ""
    output_type: str = ""

    # Audio-specific
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    amplitude_norm: bool = False
    pre_emphasis: float = 0.0
    chunk_duration_ms: Optional[int] = None
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    n_mels: Optional[int] = None
    feature_norm: str = "none"
    vocab_file: Optional[str] = None
    beam_width: int = 5
    feature_layout: str = "BCT"   # "BCT" = [batch,n_mels,T] | "BTC" = [batch,T,n_mels]
    blank_id: int = 0
    context_size: int = 2
    high_freq: float = -400.0

    # Image-specific
    resize: Optional[List[int]] = None
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    color_format: str = "RGB"
    resize_mode: str = "contain"
    interpolation: str = "bilinear"
    pad_value: float = 0.0
    pad_mode: str = "constant"   # "constant" | "reflect" | "edge"
    input_layout: str = "NCHW"
    iou_threshold: float = 0.45
    score_threshold: float = 0.25
    mask_threshold: float = 0.3
    box_format: str = "xyxy"
    keypoint_format: str = "xy"
    num_keypoints: int = 17

    # Shared
    labels_file: Optional[str] = None
    top_k: int = 1
    softmax_applied: bool = False

    # Video-specific
    num_frames: Optional[int] = None
    tubelet_size: Optional[int] = None

    def __init__(self, inspector: OnnxModelInspector, loader: ConfigLoader) -> None:
        inputs  = inspector.inputs
        outputs = inspector.outputs

        if not inputs:
            raise ValueError("ONNX model has no inputs.")
        if not outputs:
            raise ValueError("ONNX model has no outputs.")

        self.input_name    = inputs[0].name
        self.input_shape   = inputs[0].shape
        self.input_dtype   = inputs[0].dtype
        self.output_name   = outputs[0].name
        self.output_shape  = outputs[0].shape
        self.output_dtype  = outputs[0].dtype
        self.metadata_props = inspector.metadata_props

        p = loader.params
        self.modality    = p["modality"]
        self.input_type  = p["input_type"]
        self.output_type = p["output_type"]

        # Audio fields
        self.sample_rate      = p.get("sample_rate")
        self.channels         = p.get("channels")
        self.amplitude_norm   = bool(p.get("amplitude_norm", False))
        self.pre_emphasis     = float(p.get("pre_emphasis", 0.0))
        self.chunk_duration_ms = p.get("chunk_duration_ms")
        self.n_fft            = p.get("n_fft")
        self.hop_length       = p.get("hop_length")
        self.n_mels           = p.get("n_mels")
        self.feature_norm     = p.get("feature_norm", "none")
        self.vocab_file       = p.get("vocab_file")
        self.beam_width       = int(p.get("beam_width", 5))
        self.feature_layout   = p.get("feature_layout", "BCT")
        self.blank_id         = int(p.get("blank_id", 0))
        self.context_size     = int(p.get("context_size", 2))
        self.high_freq        = float(p.get("high_freq", -400.0))

        # Image fields
        self.resize        = p.get("resize")
        self.mean          = p.get("mean")
        self.std           = p.get("std")
        self.color_format  = p.get("color_format", "RGB")
        self.resize_mode   = p.get("resize_mode", "contain")
        self.interpolation = p.get("interpolation", "bilinear")
        self.pad_value     = float(p.get("pad_value", 0.0))
        self.pad_mode      = p.get("pad_mode", "constant")
        self.input_layout  = p.get("input_layout", inspector.input_layout)
        self.iou_threshold   = float(p.get("iou_threshold", p.get("nms_iou_threshold", 0.45)))
        self.score_threshold = float(p.get("score_threshold", 0.25))
        self.mask_threshold  = float(p.get("mask_threshold", 0.3))
        self.box_format      = p.get("box_format", "xyxy")
        self.keypoint_format = p.get("keypoint_format", "xy")
        self.num_keypoints   = int(p.get("num_keypoints", 17))

        # Shared
        self.labels_file      = p.get("labels_file")
        self.top_k            = int(p.get("top_k", 1))
        self.softmax_applied  = bool(p.get("softmax_applied", False))

        # Video-specific
        self.num_frames   = p.get("num_frames")
        self.tubelet_size = p.get("tubelet_size")

        self._validate()

    def _validate(self) -> None:
        def _require(field_name: str, context: str) -> None:
            if getattr(self, field_name, None) is None:
                raise ValueError(
                    f"ModelConfig validation failed: '{field_name}' is required for {context}. "
                    f"Add '{field_name}: <value>' to your config file."
                )

        if self.modality == "audio":
            _require("sample_rate", "audio models")

        if self.input_type == "log_mel":
            _require("n_fft",       "input_type=log_mel")
            _require("hop_length",  "input_type=log_mel")
            _require("n_mels",      "input_type=log_mel")
            _require("sample_rate", "input_type=log_mel")

        if self.input_type == "kaldi_fbank":
            _require("n_mels",      "input_type=kaldi_fbank")
            _require("sample_rate", "input_type=kaldi_fbank")

        if self.input_type == "mfcc":
            _require("n_fft",      "input_type=mfcc")
            _require("hop_length", "input_type=mfcc")
            _require("sample_rate","input_type=mfcc")

        if self.output_type in ("ctc_greedy", "ctc_beam", "attention_greedy", "attention_beam"):
            _require("vocab_file", f"output_type={self.output_type}")

        if self.modality == "image":
            if self.input_type not in ("denoising", "colorization"):
                _require("resize", "image models")
                _require("mean",   "image models")
                _require("std",    "image models")
            elif self.input_type == "colorization":
                _require("resize", "input_type=colorization")
                # mean/std default to ImageNet values if omitted — no hard require

        if self.input_type == "video_classification":
            _require("num_frames",   "input_type=video_classification")
            _require("tubelet_size", "input_type=video_classification")
