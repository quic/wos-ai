# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
EasyOCR plugin — two-stage OCR (detector + recognizer).

Stage 1 — detector.onnx
  input:  [1, 608, 800, 3]  NHWC float32 (RGB, ImageNet-normalised)
  output: [1, 304, 400, 2]  NHWC float32 (score_text, score_link)

Stage 2 — recognizer.onnx
  input:  [1, 64, 800, 1]   NHWC float32 (grayscale, [0,1])
  output: [1, 199, 97]      float32 CTC logits over english_g2 charset

Output: JSON  {"results": [{"box": [xmin,xmax,ymin,ymax], "text": "...", "confidence": 0.98}, ...]}
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict

import numpy as np

try:
    from utils.venv_plugin import VenvPlugin
except ModuleNotFoundError:
    class VenvPlugin:   # minimal stand-in when running outside the WoS server
        pass

# EasyOCR english_g2 characters: index 0 = blank (CTC), 1..96 = chars in order
_EASYOCR_EN_CHARS = (
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
# id→char: 0=blank, 1='0', 2='1', ..., 96='z'
_EN_ID2CHAR: Dict[int, str] = {i + 1: c for i, c in enumerate(_EASYOCR_EN_CHARS)}


def _ctc_decode(logits: np.ndarray) -> tuple[str, float]:
    """Greedy CTC decode of [1, T, vocab] logits. Returns (text, mean_max_prob)."""
    from pipeline_core.postprocessor import ctc_greedy_decode

    # softmax for confidence, argmax for greedy path
    probs = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)  # [T, vocab]

    ids = probs.argmax(axis=-1)          # [T]
    max_probs = probs[np.arange(len(ids)), ids]

    text = ctc_greedy_decode(logits, _EN_ID2CHAR, blank_id=0)

    # confidence = geometric mean of max-prob over non-blank frames
    non_blank = ids != 0
    confidence = float(max_probs[non_blank].prod() ** (1.0 / max(non_blank.sum(), 1)))
    return text, confidence


class EasyOCRPlugin(VenvPlugin):
    def load(self, model_id: str, config: Dict) -> None:
        from pipeline_core import (
            ConfigLoader, ModelConfig, OnnxModelInspector,
            Preprocessor, Postprocessor, create_session,
            easyocr_detector_postprocess, prepare_recognizer_crop,
        )

        detector_path   = config["model_path"]
        recognizer_path = config["recognizer_path"]
        config_path     = config["config_path"]
        use_qnn         = config.get("use_qnn", False)

        inspector  = OnnxModelInspector(detector_path, use_qnn=use_qnn)
        loader     = ConfigLoader(config_path)
        self._cfg  = ModelConfig(inspector, loader)

        self._text_threshold = float(loader.params.get("text_threshold", 0.7))
        self._link_threshold = float(loader.params.get("link_threshold", 0.4))

        self._pre  = Preprocessor(self._cfg)
        self._post = Postprocessor(self._cfg)

        self._detector,   self._run_opts = create_session(detector_path,   use_qnn=use_qnn)
        self._recognizer, _              = create_session(recognizer_path, use_qnn=use_qnn)

        # recognizer input shape: [1, H, W, 1]
        rec_shape      = self._recognizer.get_inputs()[0].shape  # [1, 64, 800, 1]
        self._rec_h    = rec_shape[1]
        self._rec_w    = rec_shape[2]
        self._rec_name = self._recognizer.get_inputs()[0].name

        # detector input shape for coord scaling
        det_shape   = self._detector.get_inputs()[0].shape  # [1, 608, 800, 3]
        self._det_h = det_shape[1]
        self._det_w = det_shape[2]

        self._easyocr_detector_postprocess = easyocr_detector_postprocess
        self._prepare_recognizer_crop      = prepare_recognizer_crop

    def unload(self) -> None:
        for attr in ("_detector", "_recognizer"):
            if hasattr(self, attr):
                delattr(self, attr)

    def image_variation(self, image: bytes, n: int = 1, size: str = "1024x1024",
                        response_format: str = "b64_json", **kwargs) -> list:
        # ── Stage 1: detector ─────────────────────────────────────────────────
        tensor  = self._pre.process(image)              # [1, 608, 800, 3] NHWC
        det_out = self._detector.run(
            None, {self._cfg.input_name: tensor}, self._run_opts
        )                                             # [[1, 304, 400, 2]]

        h_list, f_list = self._easyocr_detector_postprocess(
            det_out[0], self._det_h, self._det_w,
            text_threshold=self._text_threshold,
            link_threshold=self._link_threshold,
        )

        if not h_list and not f_list:
            if response_format == "url":
                from pipeline_core import draw_ocr_boxes
                png = draw_ocr_boxes(self._pre._last_rgb_uint8, [])
                return [{"image_bytes": png}]
            return [{"b64_json": json.dumps({"results": []})}]

        # ── Prepare grayscale image for crops ────────────────────────────────
        # Must match exactly the letterboxed image the detector saw.
        # tensor is [1, H, W, 3] NHWC float32 in detector space — convert to uint8 gray.
        # Denormalise with ImageNet mean/std, clamp to [0,255], average channels.
        det_np = tensor[0]  # [H, W, 3] float32, normalised
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_f32 = np.clip((det_np * std + mean) * 255.0, 0, 255)
        gray_arr = rgb_f32.mean(axis=-1).astype(np.uint8)  # [H, W]

        # ── Stage 2: recognizer per box ───────────────────────────────────────
        results = []

        for box in h_list:
            crop   = self._prepare_recognizer_crop(
                gray_arr, box, self._rec_h, self._rec_w
            )                                         # [1, 64, 800, 1]
            rec_out = self._recognizer.run(
                None, {self._rec_name: crop}
            )                                         # [[1, 199, 97]]
            text, conf = _ctc_decode(rec_out[0])
            text = text.strip()
            if text and text[-1] in ("]", "|"):
                text = text[:-1].strip()
            if text:
                results.append({
                    "box":        list(map(int, box)),
                    "text":       text,
                    "confidence": round(conf, 4),
                })

        for box in f_list:
            crop   = self._prepare_recognizer_crop(
                gray_arr, box, self._rec_h, self._rec_w
            )
            rec_out = self._recognizer.run(
                None, {self._rec_name: crop}
            )
            text, conf = _ctc_decode(rec_out[0])
            text = text.strip()
            if text and text[-1] in ("]", "|"):
                text = text[:-1].strip()
            if text:
                results.append({
                    "box":        [list(map(int, corner)) for corner in box],
                    "text":       text,
                    "confidence": round(conf, 4),
                })

        if response_format == "url":
            from pipeline_core import draw_ocr_boxes
            png = draw_ocr_boxes(self._pre._last_rgb_uint8, results)
            return [{"image_bytes": png}]

        return [{"b64_json": json.dumps({"results": results})}]
