# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .config import ModelConfig


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    boxes: np.ndarray      # [N, 4]
    scores: np.ndarray     # [N]
    class_ids: np.ndarray  # [N]
    labels: List[str]      # [N]


@dataclass
class ClassificationResult:
    labels: List[str]
    scores: np.ndarray


@dataclass
class PoseResult:
    keypoints: np.ndarray  # [N, num_keypoints, 2]
    scores: np.ndarray     # [N, num_keypoints]


# ── Audio postprocessing helpers ──────────────────────────────────────────────

def load_vocab(vocab_file: str) -> Dict:
    with open(vocab_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data
    return {tok: idx for idx, tok in enumerate(data)}


def load_vocab_txt(vocab_file: str) -> Dict[int, str]:
    """Load a space-separated 'token id' tokens.txt file → {id: token}."""
    id_to_token: Dict[int, str] = {}
    with open(vocab_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                token, idx = parts[0], int(parts[1])
            else:
                token, idx = line, len(id_to_token)
            id_to_token[idx] = token
    return id_to_token


def _vocab_id_to_token(vocab: Dict) -> Dict[int, str]:
    if vocab and isinstance(next(iter(vocab)), str):
        return {v: k for k, v in vocab.items()}
    return vocab


def ctc_greedy_decode(logits: np.ndarray, id_to_token: Dict[int, str],
                      blank_id: int = 0) -> str:
    # logits: [T, vocab] or [1, T, vocab]
    if logits.ndim == 3:
        logits = logits[0]
    ids = logits.argmax(axis=-1).tolist()
    # collapse repeats and remove blank
    out, prev = [], -1
    for idx in ids:
        if idx != prev and idx != blank_id:
            out.append(idx)
        prev = idx
    return "".join(id_to_token.get(i, "") for i in out)


def ctc_beam_decode(logits: np.ndarray, id_to_token: Dict[int, str],
                    blank_id: int = 0, beam_width: int = 5) -> str:
    # Simple beam-search CTC (no language model)
    if logits.ndim == 3:
        logits = logits[0]
    log_probs = _log_softmax(logits)
    T, V = log_probs.shape
    beams = [("", 0.0)]
    for t in range(T):
        new_beams: Dict[str, float] = {}
        for seq, score in beams:
            top_k = np.argsort(log_probs[t])[-beam_width:]
            for idx in top_k:
                lp = log_probs[t, idx]
                if idx == blank_id:
                    key = seq
                elif seq and id_to_token.get(idx, "") == seq[-1]:
                    key = seq
                else:
                    key = seq + id_to_token.get(idx, "")
                new_beams[key] = max(new_beams.get(key, -1e9), score + lp)
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0] if beams else ""


def attention_greedy_decode(logits: np.ndarray,
                            id_to_token: Dict[int, str]) -> str:
    if logits.ndim == 3:
        logits = logits[0]
    ids = logits.argmax(axis=-1).tolist()
    return "".join(id_to_token.get(i, "") for i in ids)


def attention_beam_decode(logits: np.ndarray, id_to_token: Dict[int, str],
                          beam_width: int = 5) -> str:
    if logits.ndim == 3:
        logits = logits[0]
    log_probs = _log_softmax(logits)
    beams = [("", 0.0)]
    for t in range(len(log_probs)):
        new_beams: Dict[str, float] = {}
        for seq, score in beams:
            top_k = np.argsort(log_probs[t])[-beam_width:]
            for idx in top_k:
                key = seq + id_to_token.get(int(idx), "")
                new_beams[key] = max(new_beams.get(key, -1e9),
                                     score + log_probs[t, idx])
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0] if beams else ""


def softmax_fn(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def softmax_top_k(logits: np.ndarray, labels: List[str],
                  k: int = 1) -> List[Dict]:
    if logits.ndim > 1:
        logits = logits.flatten()
    probs  = softmax_fn(logits)
    top_k  = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i] if i < len(labels) else str(i),
             "score": float(probs[i])} for i in top_k]


def waveform_reconstruct(spectrogram: np.ndarray, sr: int,
                         n_fft: int, hop: int) -> bytes:
    import librosa
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]
    waveform = librosa.griffinlim(spectrogram, n_iter=32,
                                   hop_length=hop, win_length=n_fft)
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


# ── Image postprocessing helpers ──────────────────────────────────────────────

def load_labels(labels_file: str) -> List[str]:
    with open(labels_file, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def bbox_decode(output: np.ndarray, box_format: str) -> np.ndarray:
    """Normalize box representation to xyxy."""
    if box_format == "xyxy":
        return output
    if box_format == "xywh":
        boxes = output.copy()
        boxes[..., 2] = output[..., 0] + output[..., 2]
        boxes[..., 3] = output[..., 1] + output[..., 3]
        return boxes
    if box_format == "cxcywh":
        boxes = np.zeros_like(output)
        boxes[..., 0] = output[..., 0] - output[..., 2] / 2
        boxes[..., 1] = output[..., 1] - output[..., 3] / 2
        boxes[..., 2] = output[..., 0] + output[..., 2] / 2
        boxes[..., 3] = output[..., 1] + output[..., 3] / 2
        return boxes
    raise ValueError(f"Unknown box_format '{box_format}'. Expected xyxy, xywh, cxcywh.")


def nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
        iou_threshold: float, score_threshold: float,
        max_det: int = 300):
    mask = scores >= score_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    if len(scores) == 0:
        return boxes, scores, class_ids

    order = np.argsort(scores)[::-1]
    keep  = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou <= iou_threshold]

    keep = keep[:max_det]
    return boxes[keep], scores[keep], class_ids[keep]


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def extract_keypoints(heatmaps: np.ndarray, num_kp: int,
                      threshold: float = 0.1):
    # heatmaps: [batch, num_kp, H, W]
    if heatmaps.ndim == 3:
        heatmaps = heatmaps[np.newaxis]
    B, K, H, W = heatmaps.shape
    keypoints = np.zeros((B, K, 2), dtype=np.float32)
    scores    = np.zeros((B, K),    dtype=np.float32)
    for b in range(B):
        for k in range(K):
            hm  = heatmaps[b, k]
            idx = np.unravel_index(hm.argmax(), hm.shape)
            scores[b, k]       = float(hm[idx])
            keypoints[b, k, 0] = float(idx[1]) / W  # x (normalized)
            keypoints[b, k, 1] = float(idx[0]) / H  # y (normalized)
    return keypoints, scores


def _heatmap_nms(heat: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Max-pool NMS: zero out non-local-max peaks in a heatmap [B,C,H,W]."""
    from scipy.ndimage import maximum_filter
    hmax = maximum_filter(heat, size=(1, 1, kernel, kernel), mode="constant")
    return heat * (heat == hmax)


def centernet_pose_decode(
    hm: np.ndarray,         # [1, 1,  H, W]  person center heatmap (sigmoid applied)
    wh: np.ndarray,         # [1, 2,  H, W]  bbox w/h
    hps: np.ndarray,        # [1, 34, H, W]  keypoint offset from center (2*num_kp)
    reg: np.ndarray,        # [1, 2,  H, W]  center sub-pixel offset
    hm_hp: np.ndarray,      # [1, 17, H, W]  per-keypoint heatmap (sigmoid applied)
    hm_offset: np.ndarray,  # [1, 2,  H, W]  per-keypoint heatmap offset
    score_threshold: float = 0.1,
    max_dets: int = 100,
    num_kp: int = 17,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Decode CenterNet-Pose outputs into keypoints and scores.

    Returns
    -------
    keypoints : np.ndarray  [N, num_kp, 2]  in normalised [0,1] image coords
    scores    : np.ndarray  [N]             person detection scores
    """
    B, _, H, W = hm.shape

    # ── 1. NMS on center heatmap ────────────────────────────────────────
    hm_nms = _heatmap_nms(hm)           # [1, 1, H, W]
    scores_flat = hm_nms[0, 0].flatten()
    topk = min(max_dets, len(scores_flat))
    topk_inds = np.argsort(scores_flat)[::-1][:topk]
    topk_scores = scores_flat[topk_inds]

    # ── 2. Filter by threshold ──────────────────────────────────────────
    keep = topk_scores >= score_threshold
    if not keep.any():
        return np.zeros((0, num_kp, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    topk_inds   = topk_inds[keep]
    topk_scores = topk_scores[keep]

    # ── 3. Center coords (ys, xs) in heatmap space ──────────────────────
    ys = (topk_inds // W).astype(np.float32)
    xs = (topk_inds %  W).astype(np.float32)

    # Sub-pixel offset from reg
    reg0 = reg[0]  # [2, H, W]
    xs += reg0[0].flatten()[topk_inds]
    ys += reg0[1].flatten()[topk_inds]

    # ── 4. Keypoint positions via hps offsets from center ───────────────
    hps0 = hps[0]        # [34, H, W]
    # hps layout: [kp0_x_offset, kp0_y_offset, kp1_x_offset, ...]
    hps_flat = hps0.reshape(num_kp * 2, H * W)   # [34, H*W]

    N = len(topk_inds)
    kp_raw = hps_flat[:, topk_inds].T  # [N, 34]

    kp_x = kp_raw[:, 0::2]  # [N, 17]  x offset
    kp_y = kp_raw[:, 1::2]  # [N, 17]  y offset

    # Absolute keypoint locations in heatmap space
    kp_abs_x = xs[:, None] + kp_x   # [N, 17]
    kp_abs_y = ys[:, None] + kp_y   # [N, 17]

    # ── 5. Refine keypoints using hm_hp and hm_offset; collect per-kp scores ──
    hm_hp_nms = _heatmap_nms(hm_hp[0:1])   # [1, 17, H, W]
    hp_off = hm_offset[0]                   # [2, H, W]
    kp_scores = np.zeros((N, num_kp), dtype=np.float32)

    for n in range(N):
        for k in range(num_kp):
            kx = float(np.clip(np.round(kp_abs_x[n, k]), 0, W - 1))
            ky = float(np.clip(np.round(kp_abs_y[n, k]), 0, H - 1))
            # Find the peak in a local window (radius 4) around predicted location
            r = 4
            x0, x1 = int(max(0, kx - r)), int(min(W, kx + r + 1))
            y0, y1 = int(max(0, ky - r)), int(min(H, ky + r + 1))
            local_hm = hm_hp_nms[0, k, y0:y1, x0:x1]
            if local_hm.size > 0 and local_hm.max() > 0:
                peak = np.unravel_index(local_hm.argmax(), local_hm.shape)
                py, px = peak[0] + y0, peak[1] + x0
                kp_abs_x[n, k] = px + hp_off[0, py, px]
                kp_abs_y[n, k] = py + hp_off[1, py, px]
                kp_scores[n, k] = float(hm_hp[0, k, py, px])
            else:
                # No peak found — use raw hm_hp value at predicted location
                kp_scores[n, k] = float(
                    hm_hp[0, k, int(np.clip(ky, 0, H - 1)), int(np.clip(kx, 0, W - 1))]
                )

    # ── 6. Normalise to [0, 1] image coordinates ───────────────────────
    keypoints = np.stack([kp_abs_x / W, kp_abs_y / H], axis=-1)  # [N, 17, 2]
    keypoints = keypoints.clip(0.0, 1.0).astype(np.float32)

    # scores shape: [N, num_kp]  (per-keypoint confidence from hm_hp)
    return keypoints, kp_scores


def _craft_get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    """
    Port of easyocr.craft_utils.getDetBoxes (poly=False).
    Pure numpy + scipy.ndimage — no cv2 required.
    """
    import math
    from scipy.ndimage import label as _label, binary_dilation

    text_score = (textmap > low_text).astype(np.uint8)
    link_score = (linkmap > link_threshold).astype(np.uint8)
    combined   = np.clip(text_score + link_score, 0, 1)

    labeled, nLabels = _label(combined)

    det = []
    for k in range(1, nLabels + 1):
        region = labeled == k
        size   = int(region.sum())
        if size < 10:
            continue
        if textmap[region].max() < text_threshold:
            continue

        segmap = region.astype(np.uint8)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

        ys, xs = np.where(segmap != 0)
        if len(xs) == 0:
            continue

        stats_x, stats_y = int(xs.min()), int(ys.min())
        stats_w = int(xs.max()) - stats_x + 1
        stats_h = int(ys.max()) - stats_y + 1
        niter   = int(math.sqrt(size * min(stats_w, stats_h) / (stats_w * stats_h)) * 2)
        img_h, img_w = textmap.shape
        sx = max(0,     stats_x - niter)
        sy = max(0,     stats_y - niter)
        ex = min(img_w, stats_x + stats_w + niter + 1)
        ey = min(img_h, stats_y + stats_h + niter + 1)

        struct  = np.ones((1 + 2 * niter, 1 + 2 * niter), dtype=bool)
        patch   = segmap[sy:ey, sx:ex].astype(bool)
        dilated = binary_dilation(patch, structure=struct)
        segmap2 = np.zeros_like(segmap)
        segmap2[sy:ey, sx:ex] = dilated.astype(np.uint8)

        ys2, xs2 = np.where(segmap2 != 0)
        if len(xs2) == 0:
            continue
        pts = np.stack([xs2, ys2], axis=1).astype(np.float32)

        box = _min_area_rect_box(pts)
        if box is None:
            continue

        bw = float(np.linalg.norm(box[0] - box[1]))
        bh = float(np.linalg.norm(box[1] - box[2]))
        box_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            box = np.array([
                [xs2.min(), ys2.min()],
                [xs2.max(), ys2.min()],
                [xs2.max(), ys2.max()],
                [xs2.min(), ys2.max()],
            ], dtype=np.float32)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        det.append(box.astype(np.float32))

    return det


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Graham scan convex hull. Returns hull vertices in CCW order."""
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x then y
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float32)


def _min_area_rect_box(pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Minimum-area bounding rectangle via rotating calipers on convex hull.
    Returns [4, 2] float32 corners, or None if degenerate.
    """
    if len(pts) < 2:
        return None

    # Axis-aligned fallback for tiny point clouds
    if len(pts) < 4:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    hull = _convex_hull(pts)
    n = len(hull)
    if n < 2:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    best_area = float("inf")
    best_box  = None
    for i in range(n):
        edge  = hull[(i + 1) % n] - hull[i]
        angle = np.arctan2(edge[1], edge[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rot     = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = hull @ rot.T
        x_min, x_max = rotated[:, 0].min(), rotated[:, 0].max()
        y_min, y_max = rotated[:, 1].min(), rotated[:, 1].max()
        area = (x_max - x_min) * (y_max - y_min)
        if area < best_area:
            best_area = area
            corners   = np.array([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max],
            ])
            best_box = (corners @ rot).astype(np.float32)

    return best_box


def _craft_group_text_box(polys, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
                          width_ths=1.0, add_margin=0.05, sort_output=True):
    """Faithful port of easyocr.utils.group_text_box (numpy only, no cv2)."""
    import math
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up   = (poly[3] - poly[1]) / max(10, poly[2] - poly[0])
        slope_down = (poly[5] - poly[7]) / max(10, poly[4] - poly[6])
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max(poly[0], poly[2], poly[4], poly[6])
            x_min = min(poly[0], poly[2], poly[4], poly[6])
            y_max = max(poly[1], poly[3], poly[5], poly[7])
            y_min = min(poly[1], poly[3], poly[5], poly[7])
            horizontal_list.append([x_min, x_max, y_min, y_max,
                                     0.5 * (y_min + y_max), y_max - y_min])
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width  = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
            margin = int(1.44 * add_margin * min(width, height))
            theta13 = abs(math.atan2(poly[1] - poly[5], max(10, poly[0] - poly[4])))
            theta24 = abs(math.atan2(poly[3] - poly[7], max(10, poly[2] - poly[6])))
            x1 = poly[0] - math.cos(theta13) * margin
            y1 = poly[1] - math.sin(theta13) * margin
            x2 = poly[2] + math.cos(theta24) * margin
            y2 = poly[3] - math.sin(theta24) * margin
            x3 = poly[4] + math.cos(theta13) * margin
            y3 = poly[5] + math.sin(theta13) * margin
            x4 = poly[6] - math.cos(theta24) * margin
            y4 = poly[7] + math.sin(theta24) * margin
            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # Pass 1: cluster boxes into same-line groups by y-center proximity
    new_box = []
    b_height, b_ycenter = [], []
    for poly in horizontal_list:
        if not new_box:
            b_height  = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                combined_list.append(new_box)
                b_height  = [poly[5]]
                b_ycenter = [poly[4]]
                new_box = [poly]
    combined_list.append(new_box)

    # Pass 2: within each line cluster, merge horizontally adjacent boxes
    for boxes in combined_list:
        if len(boxes) == 1:
            box    = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([box[0] - margin, box[1] + margin,
                                 box[2] - margin, box[3] + margin])
        else:
            boxes = sorted(boxes, key=lambda item: item[0])
            merged_box, new_box = [], []
            x_max = None
            for box in boxes:
                if not new_box:
                    b_height = [box[5]]
                    x_max    = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)
                            and (box[0] - x_max) < width_ths * (box[3] - box[2])):
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        merged_box.append(new_box)
                        b_height = [box[5]]
                        x_max    = box[1]
                        new_box  = [box]
            if new_box:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:
                    x_min = min(b[0] for b in mbox)
                    x_max = max(b[1] for b in mbox)
                    y_min = min(b[2] for b in mbox)
                    y_max = max(b[3] for b in mbox)
                    margin = int(add_margin * min(x_max - x_min, y_max - y_min))
                    merged_list.append([x_min - margin, x_max + margin,
                                        y_min - margin, y_max + margin])
                else:
                    box    = mbox[0]
                    margin = int(add_margin * min(box[1] - box[0], box[3] - box[2]))
                    merged_list.append([box[0] - margin, box[1] + margin,
                                        box[2] - margin, box[3] + margin])

    return merged_list, free_list


def easyocr_detector_postprocess(
    raw_output: np.ndarray,
    input_h: int,
    input_w: int,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    slope_ths: float = 0.1,
    ycenter_ths: float = 0.5,
    height_ths: float = 0.5,
    width_ths: float = 0.5,
    add_margin: float = 0.1,
    min_size: int = 20,
) -> tuple:
    """
    Convert EasyOCR detector output to bounding boxes.

    raw_output : [1, H/2, W/2, 2]  (score_text, score_link in last dim)
    input_h/w  : original detector input image dimensions (H, W)

    Returns (horizontal_list, free_list).
    horizontal_list : list of [xmin, xmax, ymin, ymax] in input-image pixel coords
    free_list       : list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in input-image pixel coords
    """
    # raw_output is NHWC [1, H/2, W/2, 2]
    out = raw_output[0]  # [H/2, W/2, 2]
    score_text = out[:, :, 0]
    score_link = out[:, :, 1]

    boxes_raw = _craft_get_det_boxes(
        score_text, score_link,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
    )

    # getDetBoxes returns coords in score-map space (half resolution).
    # Scale up by 2 to get input-image pixel coords.
    detections = []
    for box in boxes_raw:
        scaled = np.array(box, dtype=np.float32) * 2
        scaled[:, 0] = np.clip(scaled[:, 0], 0, input_w - 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, input_h - 1)
        detections.append(scaled.astype(np.int32).reshape(-1))

    horizontal_list, free_list = _craft_group_text_box(
        detections,
        slope_ths=slope_ths,
        ycenter_ths=ycenter_ths,
        height_ths=height_ths,
        width_ths=width_ths,
        add_margin=add_margin,
    )

    if min_size:
        horizontal_list = [
            b for b in horizontal_list
            if max(b[1] - b[0], b[3] - b[2]) > min_size
        ]
        free_list = [
            b for b in free_list
            if max(
                max(c[0] for c in b) - min(c[0] for c in b),
                max(c[1] for c in b) - min(c[1] for c in b),
            ) > min_size
        ]

    return horizontal_list, free_list


def decode_mask(output: np.ndarray, threshold: float,
                output_size: Optional[tuple] = None) -> np.ndarray:
    # Multi-class logits (NHWC with C>1 or NCHW with C>1 at axis 1):
    # argmax over the channel axis → binary foreground mask.
    if output.ndim == 4 and output.shape[-1] > 1:
        # NHWC: channel is last dim
        mask = (np.argmax(output, axis=-1) > 0).astype(np.uint8)
    elif output.ndim == 4 and output.shape[1] > 1:
        # NCHW: channel is dim 1
        mask = (np.argmax(output, axis=1) > 0).astype(np.uint8)
    else:
        # Single-channel binary logit: sigmoid + threshold
        mask = 1.0 / (1.0 + np.exp(-output.astype(np.float64)))
        mask = (mask >= threshold).astype(np.uint8)
    if output_size is not None and mask.shape[-2:] != output_size:
        from PIL import Image
        # resize each mask in batch
        out = []
        for m in (mask if mask.ndim == 3 else mask[0:1]):
            pil_m = Image.fromarray(m * 255)
            pil_m = pil_m.resize((output_size[1], output_size[0]), Image.NEAREST)
            out.append(np.array(pil_m) // 255)
        mask = np.stack(out)
    return mask


def decode_denoising_color(denoised_y: np.ndarray, model_rgb: np.ndarray) -> bytes:
    """Reconstruct a full-colour PNG from a denoised Y channel + model-resolution RGB.

    denoised_y : model output, shape [1, H, W, 1] or [H, W, 1] or [H, W], float32 [0, 1]
    model_rgb  : resized RGB stored by _pipeline_denoising, shape [H, W, 3], float32 [0, 1]

    Algorithm:
      1. Compute original Y from model_rgb using BT.601 full-range coefficients.
      2. Compute residual: delta_Y = denoised_Y - orig_Y  (what the model corrected).
      3. Reconstruct: output_RGB = model_rgb + delta_Y[:,:,None] then clip to [0, 1].
    """
    import io as _io
    from PIL import Image

    out = denoised_y
    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]                           # [H, W]

    # Original Y at model resolution
    r, g, b = model_rgb[:, :, 0], model_rgb[:, :, 1], model_rgb[:, :, 2]
    orig_y = 0.299 * r + 0.587 * g + 0.114 * b      # [H, W] in [0, 1]

    # Apply residual to all channels
    delta_y = out - orig_y                           # [H, W]
    reconstructed = model_rgb + delta_y[:, :, np.newaxis]
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    out_uint8 = (reconstructed * 255.0).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(out_uint8).save(buf, format="PNG")
    return buf.getvalue()



def decode_colorization(ab_output: np.ndarray, model_rgb: np.ndarray) -> bytes:
    """Reconstruct a colorized PNG from predicted AB channels + original L.

    ab_output  : model output, shape [1, 2, H, W] float32 — predicted AB channels,
                 tanh-activated in DDColor, range [-1, 1].  Scaled to CIE-Lab AB
                 range [-128, 128] by multiplying by 128.
    model_rgb  : original resized RGB stored by _pipeline_colorization, [H, W, 3]
                 float32 [0, 1].  The L channel is recomputed from this.

    Algorithm:
      1. Recompute L from model_rgb (range [0, 100]).
      2. Scale model AB output from [-1,1] → [-128, 128].
      3. Assemble Lab [H,W,3] and convert to RGB uint8.
    """
    import io as _io
    from PIL import Image

    # Squeeze batch dim
    ab = ab_output[0] if ab_output.ndim == 4 else ab_output  # [2, H, W]
    # CHW → HWC
    ab = ab.transpose(1, 2, 0)                               # [H, W, 2]
    ab = (ab * 128.0).astype(np.float32)                     # scale to CIE Lab AB range

    # Recompute L from stored RGB
    r, g, b = model_rgb[:, :, 0], model_rgb[:, :, 1], model_rgb[:, :, 2]
    # BT.601 produces luminance in [0,1]; convert to CIE L* via _lab_l_from_rgb
    lab_l = _lab_l_from_rgb(model_rgb)                       # [H, W] in [0, 100]

    lab = np.stack([lab_l, ab[:, :, 0], ab[:, :, 1]], axis=-1)  # [H, W, 3]
    rgb_out = _lab_to_rgb(lab)                               # [H, W, 3] float32 [0,1]
    rgb_uint8 = np.clip(rgb_out * 255.0, 0, 255).astype(np.uint8)

    buf = _io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG")
    return buf.getvalue()


def _lab_l_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Compute CIE L* channel from float32 RGB [H,W,3] in [0,1]. Returns [H,W]."""
    mask = rgb > 0.04045
    lin  = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92).astype(np.float32)
    # Only Y component needed for L*
    y = 0.2126729 * lin[:, :, 0] + 0.7151522 * lin[:, :, 1] + 0.0721750 * lin[:, :, 2]
    eps, kappa = 0.008856, 903.3
    fy = np.where(y > eps, np.cbrt(y), (kappa * y + 16.0) / 116.0)
    return (116.0 * fy - 16.0).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE-Lab [H,W,3] → float32 RGB [H,W,3] [0,1] (D65, sRGB)."""
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps, kappa = 0.008856, 903.3
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # Rescale by D65 white point
    x *= 0.95047
    z *= 1.08883

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    # XYZ → linear sRGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    lin = xyz @ M_inv.T

    # Apply sRGB gamma
    lin = np.clip(lin, 0.0, None)
    rgb = np.where(lin > 0.0031308,
                   1.055 * (lin ** (1.0 / 2.4)) - 0.055,
                   12.92 * lin)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    log_sum_exp = np.log(np.exp(x).sum(axis=-1, keepdims=True) + 1e-9)
    return x - log_sum_exp


def _gpt2_decode(token_string: str) -> str:
    """Invert GPT-2 bytes_to_unicode encoding so joined tokens decode as UTF-8."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    try:
        return bytes(u2b[c] for c in token_string).decode("utf-8", errors="replace")
    except KeyError:
        return token_string


def whisper_attention_decode(
    decoder: Any,
    run_opts: Any,
    cross_kv: Dict,
    suppress_ids: np.ndarray,
    sot_id: int,
    eot_id: int,
    lang_id: int,
    transcribe_id: int,
    notimestamps_id: int,
    max_decode_len: int,
    id_to_token: Dict[int, str],
) -> str:
    """Stateful autoregressive decode for QNN Whisper (static KV-cache interface)."""
    dec_inputs    = decoder.get_inputs()
    dec_out_names = [o.name for o in decoder.get_outputs()]

    # Pre-allocate self-KV cache (zeros)
    self_kv = {
        inp.name: np.zeros(inp.shape, dtype=np.float16)
        for inp in dec_inputs
        if "k_cache_self" in inp.name or "v_cache_self" in inp.name
    }
    # cache_size = key-dim of self-KV (199); attention_mask width = cache_size + 1
    cache_size = next(
        inp.shape[-1]
        for inp in dec_inputs
        if "k_cache_self" in inp.name
    )
    mask_len = cache_size + 1  # 200
    # MASK_NEG must match the value used at export time (-100.0); see model.py MASK_NEG
    NEG_INF = np.float16(-100.0)

    def _step(token_id: int, position: int) -> np.ndarray:
        # RIGHT-aligned unmasking: last (position+1) slots are 0; rest are NEG_INF.
        mask = np.full((1, 1, 1, mask_len), NEG_INF, dtype=np.float16)
        mask[0, 0, 0, mask_len - position - 1 :] = np.float16(0.0)
        feed = {
            "input_ids":      np.array([[token_id]], dtype=np.int32),
            "attention_mask": mask,
            "position_ids":   np.array([position],  dtype=np.int32),
        }
        feed.update(self_kv)
        feed.update(cross_kv)
        outs     = decoder.run(None, feed, run_opts)
        out_dict = dict(zip(dec_out_names, outs))
        for k in list(self_kv):
            out_k = k.replace("_in", "_out")
            if out_k in out_dict:
                self_kv[k] = out_dict[out_k]
        raw = out_dict["logits"][0, :, 0, 0].astype(np.float32)
        raw[suppress_ids] = -1e9
        return raw  # [vocab]

    prompt = [sot_id, lang_id, transcribe_id, notimestamps_id]
    for pos, tok in enumerate(prompt[:-1]):
        _step(tok, pos)

    logits  = _step(prompt[-1], len(prompt) - 1)
    next_id = int(logits.argmax())

    max_steps    = min(max_decode_len, cache_size - len(prompt))
    out_tokens: list = []
    repeat_count = 0
    prev_id = -1
    for pos in range(len(prompt), len(prompt) + max_steps):
        if next_id == eot_id:
            break
        if next_id == prev_id:
            repeat_count += 1
            if repeat_count >= 30:
                break
        else:
            repeat_count = 0
        prev_id = next_id
        out_tokens.append(next_id)
        logits  = _step(next_id, pos)
        next_id = int(logits.argmax())

    return "".join(_gpt2_decode(id_to_token.get(i, "")) for i in out_tokens).strip()


def zipformer_transducer_decode(
    encoder_out: Any,           # [1, T', 512]
    decoder: Any,
    joiner: Any,
    dec_run_opts: Any,
    joi_run_opts: Any,
    id_to_token: Dict[int, str],
    blank_id: int,
    context_size: int,
) -> str:
    """Greedy RNN-T decode: step through encoder frames, maintain decoder context."""
    hyp = [blank_id] * context_size   # initial context (blank padding)
    T = encoder_out.shape[1]
    for t in range(T):
        enc_frame = encoder_out[:, t, :]            # [1, 512]
        while True:
            y = np.array([hyp[-context_size:]], dtype=np.int32)  # [1, context_size]
            dec_out = decoder.run(None, {"y": y}, dec_run_opts)[0]   # [1, 512]
            logit = joiner.run(
                None,
                {"encoder_out": enc_frame, "decoder_out": dec_out},
                joi_run_opts,
            )[0]                                                  # [1, vocab]
            token_id = int(logit[0].argmax())
            if token_id == blank_id:
                break
            hyp.append(token_id)

    tokens = [id_to_token.get(i, "") for i in hyp[context_size:] if i != blank_id]
    # SentencePiece tokens use ▁ as word-boundary marker
    text = "".join(tokens).replace("▁", " ").strip()
    return text


# ── Postprocessor ─────────────────────────────────────────────────────────────

class Postprocessor:
    def __init__(self, config: ModelConfig) -> None:
        self._config  = config
        self._custom: Dict[str, Callable] = {}
        self._vocab: Dict[int, str] = {}
        self._labels: List[str] = []

        if config.output_type in (
            "ctc_greedy", "ctc_beam", "attention_greedy", "attention_beam",
            "transducer_greedy",
        ) and config.vocab_file:
            if config.vocab_file.endswith(".txt"):
                self._vocab = load_vocab_txt(config.vocab_file)
            else:
                raw = load_vocab(config.vocab_file)
                self._vocab = _vocab_id_to_token(raw)

        if config.output_type in (
            "softmax_top_k", "detection", "classification", "multiclass_mask",
        ):
            if config.labels_file and os.path.exists(config.labels_file):
                self._labels = load_labels(config.labels_file)

    def register(self, output_type: str,
                 fn: Callable[[np.ndarray, ModelConfig], Any]) -> None:
        self._custom[output_type] = fn

    def process(self, model_output: Union[List[np.ndarray], np.ndarray]) -> Any:
        cfg = self._config

        if cfg.output_type in self._custom:
            output = model_output[0] if isinstance(model_output, list) else model_output
            return self._custom[cfg.output_type](output, cfg)

        # For detection and pose, pass the full list so the decoder can handle
        # models that return separate outputs (CenterNet-Pose: 6 tensors).
        if cfg.modality == "image" and cfg.output_type in ("detection", "pose"):
            return self._process_image(model_output)

        # Unwrap list → pick first output for all other types
        if isinstance(model_output, list):
            output = model_output[0]
        else:
            output = model_output

        if cfg.modality == "audio":
            return self._process_audio(output)
        if cfg.modality == "image":
            return self._process_image(output)
        raise ValueError(f"Unknown modality '{cfg.modality}'.")

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _process_audio(self, output: np.ndarray) -> Any:
        cfg = self._config
        ot  = cfg.output_type

        if ot == "ctc_greedy":
            return ctc_greedy_decode(output, self._vocab)
        if ot == "ctc_beam":
            return ctc_beam_decode(output, self._vocab, beam_width=cfg.beam_width)
        if ot == "attention_greedy":
            return attention_greedy_decode(output, self._vocab)
        if ot == "attention_beam":
            return attention_beam_decode(output, self._vocab, beam_width=cfg.beam_width)
        if ot == "softmax_top_k":
            return softmax_top_k(output, self._labels, k=cfg.top_k)
        if ot == "waveform_reconstruction":
            return waveform_reconstruct(output, cfg.sample_rate,
                                        cfg.n_fft, cfg.hop_length)
        if ot == "transducer_greedy":
            raise ValueError(
                "output_type='transducer_greedy' requires decoder and joiner sessions. "
                "Register a handler via Postprocessor.register('transducer_greedy', fn) "
                "in the plugin's load() method."
            )
        raise ValueError(
            f"Unknown audio output_type '{ot}'. "
            f"Supported: ctc_greedy, ctc_beam, attention_greedy, attention_beam, "
            f"softmax_top_k, waveform_reconstruction, transducer_greedy. "
            f"Use Postprocessor.register('{ot}', fn) for custom types."
        )

    # ── Image ─────────────────────────────────────────────────────────────────

    def _process_image(self, output: np.ndarray) -> Any:
        cfg = self._config
        ot  = cfg.output_type

        if ot == "detection":
            return self._decode_detection(output)
        if ot == "classification":
            return self._decode_classification(output)
        if ot == "pose":
            return self._decode_pose(output)
        if ot == "segmentation":
            return self._decode_segmentation(output)
        if ot == "mask_list":
            return self._decode_mask_list(output)
        if ot == "super_resolution":
            return self._decode_super_resolution(output)
        if ot == "inpainting":
            return self._decode_inpainting(output)
        if ot == "denoising":
            return self._decode_denoising(output)
        if ot == "colorization":
            return self._decode_colorization(output)
        if ot == "multiclass_mask":
            return self._decode_multiclass_mask(output)
        raise ValueError(
            f"Unknown image output_type '{ot}'. "
            f"Supported: detection, classification, pose, segmentation, "
            f"mask_list, super_resolution, inpainting, denoising, colorization, "
            f"multiclass_mask. "
            f"Use Postprocessor.register('{ot}', fn) for custom types."
        )

    def _decode_detection(self, output: Union[np.ndarray, List[np.ndarray]]) -> DetectionResult:
        cfg = self._config

        # Separate-output format: [boxes(N,4), scores(N,), class_ids(N,)]
        if isinstance(output, list) and len(output) == 3:
            boxes_raw  = np.squeeze(output[0], axis=0) if output[0].ndim == 3 else output[0]
            scores_raw = np.squeeze(output[1], axis=0) if output[1].ndim == 2 else output[1]
            class_ids  = np.squeeze(output[2], axis=0) if output[2].ndim == 2 else output[2]
            class_ids  = class_ids.astype(np.int32)
        else:
            # Combined YOLO-style [1, N, 5+C] or [N, 5+C]
            if isinstance(output, list):
                output = output[0]
            if output.ndim == 3:
                output = output[0]
            if output.shape[-1] >= 5:
                boxes_raw  = output[:, :4]
                scores_raw = output[:, 4]
                class_ids  = (output[:, 5:].argmax(axis=-1).astype(np.int32)
                              if output.shape[-1] > 5
                              else np.zeros(len(output), dtype=np.int32))
            else:
                raise ValueError(f"Detection output shape {output.shape} not supported.")

        boxes = bbox_decode(boxes_raw, cfg.box_format)
        boxes, scores, class_ids = nms(
            boxes, scores_raw, class_ids,
            cfg.iou_threshold, cfg.score_threshold,
        )
        labels = [self._labels[i] if i < len(self._labels) else str(i)
                  for i in class_ids]
        return DetectionResult(boxes=boxes, scores=scores,
                               class_ids=class_ids, labels=labels)

    def _decode_classification(self, output: np.ndarray) -> ClassificationResult:
        cfg = self._config
        if output.ndim > 1:
            output = output.flatten()
        if cfg.softmax_applied:
            probs  = output.astype(np.float32)
            top_k  = np.argsort(probs)[::-1][:cfg.top_k]
            result = [{"label": self._labels[i] if i < len(self._labels) else str(i),
                       "score": float(probs[i])} for i in top_k]
        else:
            result = softmax_top_k(output, self._labels, k=cfg.top_k)
        return ClassificationResult(
            labels=[r["label"] for r in result],
            scores=np.array([r["score"] for r in result], dtype=np.float32),
        )

    def _decode_pose(self, output: Union[np.ndarray, List[np.ndarray]]) -> PoseResult:
        cfg = self._config
        # CenterNet-Pose returns 6 separate outputs: hm, wh, hps, reg, hm_hp, hm_offset
        if isinstance(output, list) and len(output) >= 6:
            hm, wh, hps, reg, hm_hp, hm_offset = output[:6]
            keypoints, scores = centernet_pose_decode(
                hm, wh, hps, reg, hm_hp, hm_offset,
                score_threshold=cfg.score_threshold,
                num_kp=cfg.num_keypoints,
            )
        else:
            # Fallback: single heatmap tensor (generic heatmap-based pose model)
            heatmaps = output if not isinstance(output, list) else output[0]
            keypoints, scores = extract_keypoints(heatmaps, cfg.num_keypoints,
                                                  cfg.score_threshold)
        return PoseResult(keypoints=keypoints, scores=scores)

    def _decode_segmentation(self, output: np.ndarray) -> np.ndarray:
        cfg = self._config
        return decode_mask(output, cfg.score_threshold)

    def _decode_mask_list(self, output: np.ndarray) -> list:
        cfg = self._config
        mask = decode_mask(output, threshold=getattr(cfg, "mask_threshold", 0.3))
        return mask.tolist()

    def _decode_multiclass_mask(self, output: np.ndarray) -> np.ndarray:
        # Per-pixel class-id mask. Some ONNX graphs bake the channel-argmax in
        # (output is already [1, H, W] class ids); others still emit per-class
        # logits ([1, C, H, W] or [1, H, W, C]) that need an explicit argmax here.
        mask = output
        if mask.ndim == 4:
            if mask.shape[-1] > 1 and mask.shape[1] != mask.shape[-1]:
                mask = np.argmax(mask, axis=-1)
            elif mask.shape[1] > 1:
                mask = np.argmax(mask, axis=1)
            else:
                mask = np.squeeze(mask, axis=1 if mask.shape[1] == 1 else -1)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        return mask.astype(np.uint8)

    def _decode_super_resolution(self, output: np.ndarray) -> bytes:
        import io as _io
        from PIL import Image
        out = output
        if out.ndim == 4:
            out = out[0]
        if out.shape[0] in (1, 3):  # CHW → HWC
            out = out.transpose(1, 2, 0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        if out.ndim == 3 and out.shape[2] == 1:  # HWC grayscale → HW
            out = out[:, :, 0]
        buf = _io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()

    def _decode_denoising(self, output: np.ndarray) -> bytes:
        """Fallback: greyscale PNG when no Cb/Cr cache is available.

        In normal use the plugin registers a custom handler via Postprocessor.register()
        that calls decode_denoising_color() with the Cb/Cr stashed by _pipeline_denoising.
        This fallback exists only if someone calls Postprocessor directly without the plugin.
        """
        import io as _io
        from PIL import Image
        out = output
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        buf = _io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()

    def _decode_inpainting(self, output: np.ndarray) -> bytes:
        """Decode AOT-GAN painted_image [1,H,W,3] float32 [0,1] NHWC → PNG bytes."""
        import io as _io
        from PIL import Image
        out = output
        if out.ndim == 4:
            out = out[0]              # [H, W, 3]
        if out.shape[0] in (1, 3):   # CHW → HWC (guard; model outputs NHWC)
            out = out.transpose(1, 2, 0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        buf = _io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()

    def _decode_colorization(self, output: np.ndarray) -> bytes:
        """Fallback colorization decode when no RGB cache is available.

        In normal use the plugin registers a handler via Postprocessor.register()
        that calls decode_colorization() with the RGB stashed by _pipeline_colorization.
        """
        import io as _io
        from PIL import Image
        ab = output[0] if output.ndim == 4 else output    # [2, H, W]
        H, W = ab.shape[1], ab.shape[2]
        gray = np.full((H, W, 3), 128, dtype=np.uint8)
        buf = _io.BytesIO()
        Image.fromarray(gray).save(buf, format="PNG")
        return buf.getvalue()


def voc_color_map(num_classes: int = 256) -> np.ndarray:
    """Standard PASCAL VOC color palette, class index -> [R, G, B]."""
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


def colorize_class_mask(mask: np.ndarray) -> bytes:
    """Render a per-pixel class-id mask as a PNG using the VOC color palette."""
    from PIL import Image
    cmap = voc_color_map()
    rgb = cmap[mask.astype(np.int64) % len(cmap)]
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── Overlay-drawing helpers (response_format == "url") ────────────────────────

# Standard COCO 17-keypoint skeleton, 0-indexed pairs.
_COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


def draw_detection_boxes(image: np.ndarray, boxes: np.ndarray,
                         scores: np.ndarray, labels: List[str]) -> bytes:
    """Draw xyxy pixel-space detection boxes + "label score" text. Returns PNG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        text = f"{label} {float(score):.2f}"
        draw.text((x0, max(0, y0 - 10)), text, fill=(255, 0, 0), font=font)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def draw_pose_keypoints(image: np.ndarray, keypoints: np.ndarray,
                        scores: np.ndarray, score_threshold: float = 0.1) -> bytes:
    """Draw normalised [0,1] pose keypoints + COCO skeleton. Returns PNG bytes."""
    from PIL import Image, ImageDraw

    pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    h, w = image.shape[0], image.shape[1]

    for person_kp, person_scores in zip(keypoints, scores):
        pts = person_kp * np.array([w, h], dtype=np.float32)  # [num_kp, 2] pixel coords
        visible = person_scores >= score_threshold

        for i, j in _COCO_SKELETON:
            if i < len(visible) and j < len(visible) and visible[i] and visible[j]:
                draw.line([tuple(pts[i]), tuple(pts[j])], fill=(0, 255, 0), width=2)

        for k, (x, y) in enumerate(pts):
            if visible[k]:
                r = 3
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def draw_ocr_boxes(image: np.ndarray, results: list) -> bytes:
    """Draw EasyOCR box + text results onto image. Returns PNG bytes.

    Each result's "box" is either [xmin,xmax,ymin,ymax] (horizontal box) or
    [[x,y],[x,y],[x,y],[x,y]] (quad, free-form box).
    """
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for r in results:
        box = r["box"]
        text = r.get("text", "")
        if box and isinstance(box[0], (list, tuple)):
            # quad: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            poly = [tuple(pt) for pt in box]
            draw.polygon(poly, outline=(0, 128, 255))
            x0, y0 = poly[0]
        else:
            # horizontal: [xmin, xmax, ymin, ymax]
            xmin, xmax, ymin, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 128, 255), width=2)
            x0, y0 = xmin, ymin
        draw.text((x0, max(0, y0 - 10)), text, fill=(0, 128, 255), font=font)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
