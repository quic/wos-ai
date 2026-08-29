# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from __future__ import annotations

import io
from typing import Any, Callable, Dict, Union

import numpy as np

from .config import ModelConfig


# ── Audio helpers ─────────────────────────────────────────────────────────────

def decode_audio(data: bytes, target_sr: int):
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return audio, sr


def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    import math
    gcd = math.gcd(target_sr, source_sr)
    return resample_poly(audio, target_sr // gcd, source_sr // gcd).astype(np.float32)


def to_mono(audio: np.ndarray, channels: int = 1) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1).astype(np.float32)


def pre_emphasis(audio: np.ndarray, coeff: float) -> np.ndarray:
    if coeff == 0.0:
        return audio
    return np.concatenate([[audio[0]], audio[1:] - coeff * audio[:-1]]).astype(np.float32)


def pad_or_chunk(audio: np.ndarray, sr: int, duration_ms: int) -> np.ndarray:
    target = int(sr * duration_ms / 1000)
    if len(audio) >= target:
        return audio[:target]
    return np.pad(audio, (0, target - len(audio))).astype(np.float32)


def amplitude_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return (audio / peak).astype(np.float32)
    return audio


def mel_spectrogram(audio: np.ndarray, sr: int, n_fft: int,
                    hop: int, n_mels: int) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
    )
    return mel.astype(np.float32)


def whisper_log_mel(audio: np.ndarray, sr: int, n_fft: int,
                    hop: int, n_mels: int) -> np.ndarray:
    """
    Whisper-compatible log-mel using torch.stft + torchaudio mel filterbank.
    Matches openai/whisper audio.py exactly: Hann window, power spectrogram,
    log10, clamp to max-8, normalise to [-1, 1].
    """
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float()
    window   = torch.hann_window(n_fft)
    stft     = torch.stft(waveform, n_fft, hop, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2          # [n_freqs, T]

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=float(sr) / 2,
        n_mels=n_mels,
        sample_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )                                                # [n_freqs, n_mels]

    mel_spec = mel_fb.T @ magnitudes                 # [n_mels, T]
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)


def log_mel(mel: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    return np.log(mel + offset).astype(np.float32)


def mfcc(audio: np.ndarray, sr: int, n_mfcc: int,
         n_fft: int, hop: int) -> np.ndarray:
    import librosa
    coeffs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
    )
    return coeffs.astype(np.float32)


def kaldi_fbank_features(audio: np.ndarray, sr: int, n_mels: int,
                         high_freq: float = -400.0) -> np.ndarray:
    """
    Kaldi-style filterbank features using torchaudio.compliance.kaldi.fbank.
    Matches the reference app.py parameters: dither=0, snip_edges=False,
    samp_freq=sr, num_bins=n_mels, high_freq=-400.
    Returns shape [T, n_mels] (no batch dimension).
    """
    import torch
    from torchaudio.compliance import kaldi as taudio_kaldi
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # [1, N]
    frames = taudio_kaldi.fbank(
        waveform,
        dither=0.0,
        snip_edges=False,
        sample_frequency=float(sr),
        num_mel_bins=n_mels,
        high_freq=high_freq,
    )
    return frames.numpy().astype(np.float32)  # [T, n_mels]



def feature_normalize(feat: np.ndarray, method: str) -> np.ndarray:
    if method == "global_mvn":
        mean = feat.mean()
        std  = feat.std() + 1e-8
        return ((feat - mean) / std).astype(np.float32)
    if method == "per_feature":
        mean = feat.mean(axis=-1, keepdims=True)
        std  = feat.std(axis=-1, keepdims=True) + 1e-8
        return ((feat - mean) / std).astype(np.float32)
    return feat


def tokenize(text: str, vocab: Dict[str, int]) -> np.ndarray:
    ids = [vocab.get(ch, vocab.get("<unk>", 0)) for ch in text]
    return np.array(ids, dtype=np.int64)


# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def letterbox(image: np.ndarray, target_w: int, target_h: int,
              pad_value: float = 0.0, pad_mode: str = "constant") -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    from PIL import Image
    pil = Image.fromarray(image)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(pil)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right  = target_w - new_w - pad_left
    if pad_mode in ("reflect", "edge"):
        np_mode = "reflect" if pad_mode == "reflect" else "edge"
        return np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode=np_mode)
    out = np.full((target_h, target_w, 3), pad_value, dtype=np.float32)
    out[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    return out


def center_crop(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    from PIL import Image
    pil = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(pil)
    top  = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return arr[top:top+target_h, left:left+target_w]


def resize_image(image: np.ndarray, width: int, height: int,
                 interpolation: str = "bilinear") -> np.ndarray:
    from PIL import Image
    _interp = {
        "bilinear": Image.BILINEAR,
        "nearest":  Image.NEAREST,
        "bicubic":  Image.BICUBIC,
    }
    pil = Image.fromarray(image).resize(
        (width, height), _interp.get(interpolation, Image.BILINEAR)
    )
    return np.array(pil)


def channel_reorder(image: np.ndarray, color_format: str) -> np.ndarray:
    if color_format == "BGR":
        return image[:, :, ::-1].copy()
    if color_format == "GRAY":
        return np.mean(image, axis=2, keepdims=True).astype(np.float32)
    return image  # RGB (default)


def normalize(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    return ((img - mean_arr) / std_arr).astype(np.float32)


def dtype_cast(image: np.ndarray, dtype: str) -> np.ndarray:
    return image.astype(dtype)


def add_batch_dim(image: np.ndarray) -> np.ndarray:
    return np.expand_dims(image, 0)


def make_inpainting_mask(image: np.ndarray, box: "tuple[int,int,int,int]") -> np.ndarray:
    """Create a float32 binary mask [H, W, 1] with 1.0 inside box, 0.0 outside.

    box: (x0, y0, x1, y1) in pixel coords of the already-resized image.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1, 0] = 1.0
    return mask


def _decode_video_frames(data: bytes, max_frames: int) -> np.ndarray:
    """Decode video bytes → [T, H, W, 3] uint8 RGB array via imageio."""
    import imageio
    import tempfile, os
    # Detect format from magic bytes so the temp file gets the right extension
    if data[:6] in (b"GIF87a", b"GIF89a"):
        suffix = ".gif"
    elif data[:4] == b"\x00\x00\x00\x18" or data[4:8] in (b"ftyp", b"moov"):
        suffix = ".mp4"
    else:
        suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        reader = imageio.get_reader(tmp_path)
        frames = []
        for frame in reader:
            arr = np.array(frame, dtype=np.uint8)
            if arr.ndim == 2:                    # grayscale → RGB
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:             # RGBA → RGB
                arr = arr[:, :, :3]
            frames.append(arr)
            if len(frames) >= max_frames * 4:    # stop early once we have enough to sample
                break
        reader.close()
    finally:
        os.unlink(tmp_path)
    if not frames:
        raise ValueError("Could not decode any frames from video data.")
    return np.stack(frames, axis=0)  # [T, H, W, 3]


def _sample_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Sample exactly `num_frames` from `frames` [T, H, W, 3].

    When T >= num_frames: uniform stride (linspace).
    When T < num_frames (short clips / GIFs): tile the clip so every
    tubelet of `tubelet_size` consecutive frames sees real motion rather
    than the same frame repeated.
    """
    T = len(frames)
    if T == 0:
        raise ValueError("No frames to sample.")
    if T >= num_frames:
        indices = np.linspace(0, T - 1, num_frames).round().astype(int)
    else:
        # Tile: frame i maps to source frame (i % T), wrapping around
        indices = np.arange(num_frames) % T
    return frames[indices]


def _video_short_side_resize(frames: np.ndarray, short_side: int,
                              interpolation: str = "bilinear") -> np.ndarray:
    """Resize all frames so the shorter spatial dimension equals `short_side`."""
    from PIL import Image
    _interp = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST, "bicubic": Image.BICUBIC}
    interp  = _interp.get(interpolation, Image.BILINEAR)
    h, w = frames.shape[1], frames.shape[2]
    scale = short_side / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.stack([
        np.array(Image.fromarray(f).resize((new_w, new_h), interp))
        for f in frames
    ], axis=0)
    return resized


def _video_center_crop(frames: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    """Center-crop all frames to crop_h × crop_w."""
    h, w = frames.shape[1], frames.shape[2]
    top  = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return frames[:, top:top + crop_h, left:left + crop_w]


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] array [H,W,3] to CIE-Lab [H,W,3].

    Uses the D65 illuminant / 2° observer via the sRGB→XYZ matrix then
    XYZ→Lab, matching skimage/PIL Lab conventions.
    L in [0,100], a/b in approx [-128, 127].
    """
    # sRGB linearise (undo gamma)
    mask  = rgb > 0.04045
    lin   = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    lin   = lin.astype(np.float32)

    # sRGB → XYZ (D65 matrix)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T  # [H, W, 3]

    # Normalise by D65 white point
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 2] /= 1.08883

    # XYZ → Lab
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    def __init__(self, config: ModelConfig) -> None:
        self._config  = config
        self._custom: Dict[str, Callable] = {}
        self._vocab: Dict[str, int] = {}
        if config.input_type == "tokens" and config.vocab_file:
            self._vocab = _load_vocab(config.vocab_file)

    def register(self, input_type: str,
                 fn: Callable[[Any, ModelConfig], np.ndarray]) -> None:
        self._custom[input_type] = fn

    def process(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        if cfg.input_type in self._custom:
            result = self._custom[cfg.input_type](data, cfg)
        elif cfg.modality == "audio":
            result = self._process_audio(data)
        elif cfg.modality == "image":
            result = self._process_image(data)
        else:
            raise ValueError(
                f"Unknown modality '{cfg.modality}'. Expected 'audio' or 'image'."
            )
        if cfg.input_dtype and not isinstance(result, tuple):
            result = result.astype(cfg.input_dtype)
        return result

    # ── Audio pipelines ───────────────────────────────────────────────────────

    def _process_audio(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        it  = cfg.input_type

        if it == "log_mel":
            return self._pipeline_log_mel(data)
        if it == "kaldi_fbank":
            return self._pipeline_kaldi_fbank(data)
        if it == "mfcc":
            return self._pipeline_mfcc(data)
        if it == "waveform":
            return self._pipeline_waveform(data)
        if it == "tokens":
            return self._pipeline_tokens(data)
        raise ValueError(
            f"Unknown audio input_type '{it}'. "
            f"Supported: log_mel, kaldi_fbank, mfcc, waveform, tokens. "
            f"Use Preprocessor.register('{it}', fn) for custom types."
        )

    def _load_audio(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        if isinstance(data, np.ndarray):
            audio = data.astype(np.float32)
        else:
            audio, sr = decode_audio(data, cfg.sample_rate)
            audio = resample(audio, sr, cfg.sample_rate)
        audio = to_mono(audio, cfg.channels or 1)
        audio = pre_emphasis(audio, cfg.pre_emphasis)
        return audio

    def _pipeline_log_mel(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg   = self._config
        audio = self._load_audio(data)
        if cfg.chunk_duration_ms:
            audio = pad_or_chunk(audio, cfg.sample_rate, cfg.chunk_duration_ms)

        if cfg.feature_norm == "whisper":
            feat = whisper_log_mel(audio, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        else:
            mel  = mel_spectrogram(audio, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels)
            feat = log_mel(mel)
            feat = feature_normalize(feat, cfg.feature_norm)

        # feat is [n_mels, T] here.
        # 4-D models expect [batch, 1, T, n_mels] (e.g. YamNet); T is input_shape[-2].
        # 3-D BTC models expect [batch, T, n_mels];               T is input_shape[-2].
        # 3-D BCT models expect [batch, n_mels, T];               T is input_shape[-1].
        is_4d = cfg.input_shape and len(cfg.input_shape) == 4
        layout = cfg.feature_layout  # "BCT" | "BTC"

        # Trim or pad the time axis to match the model's expected number of frames.
        if cfg.input_shape and len(cfg.input_shape) >= 3:
            expected_frames = cfg.input_shape[-2] if (is_4d or layout == "BTC") else cfg.input_shape[-1]
            if isinstance(expected_frames, int):
                if feat.shape[-1] > expected_frames:
                    feat = feat[..., :expected_frames]
                elif feat.shape[-1] < expected_frames:
                    pad = expected_frames - feat.shape[-1]
                    feat = np.pad(feat, ((0, 0), (0, pad)))

        if is_4d:
            # [n_mels, T] → [T, n_mels] → [1, 1, T, n_mels]
            return feat.T[np.newaxis, np.newaxis]
        if layout == "BTC":
            # [n_mels, T] → [T, n_mels] → [1, T, n_mels]
            return feat.T[np.newaxis]
        # BCT (default): [n_mels, T] → [1, n_mels, T]
        return np.expand_dims(feat, 0)

    def _pipeline_kaldi_fbank(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        # Decode audio but do NOT apply pre_emphasis — Kaldi fbank computes its own
        # internal preemphasis via the mel filterbank pipeline.
        if isinstance(data, np.ndarray):
            audio = data.astype(np.float32)
        else:
            audio, sr = decode_audio(data, cfg.sample_rate)
            audio = resample(audio, sr, cfg.sample_rate)
        audio = to_mono(audio, cfg.channels or 1)

        high_freq = cfg.high_freq
        frames = kaldi_fbank_features(audio, cfg.sample_rate, cfg.n_mels, high_freq)
        # frames: [T, n_mels]; return [1, T, n_mels] (BTC layout for zipformer)
        return frames[np.newaxis]  # [1, T, n_mels]

    def _pipeline_mfcc(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg    = self._config
        audio  = self._load_audio(data)
        if cfg.chunk_duration_ms:
            audio = pad_or_chunk(audio, cfg.sample_rate, cfg.chunk_duration_ms)
        n_mfcc = cfg.n_mels or 40
        feat   = mfcc(audio, cfg.sample_rate, n_mfcc, cfg.n_fft, cfg.hop_length)
        feat   = feature_normalize(feat, cfg.feature_norm)
        return np.expand_dims(feat, 0)

    def _pipeline_waveform(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg   = self._config
        if isinstance(data, np.ndarray):
            audio = data.astype(np.float32)
        else:
            audio, sr = decode_audio(data, cfg.sample_rate)
            audio = resample(audio, sr, cfg.sample_rate)
        audio = to_mono(audio, cfg.channels or 1)
        if cfg.amplitude_norm:
            audio = amplitude_normalize(audio)
        if cfg.chunk_duration_ms:
            audio = pad_or_chunk(audio, cfg.sample_rate, cfg.chunk_duration_ms)
        return np.expand_dims(audio, 0)

    def _pipeline_tokens(self, data: Union[str, bytes]) -> np.ndarray:
        if isinstance(data, bytes):
            text = data.decode("utf-8")
        else:
            text = data
        ids = tokenize(text, self._vocab)
        return np.expand_dims(ids, 0)

    # ── Image pipelines ───────────────────────────────────────────────────────

    def _process_image(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        it  = cfg.input_type
        if it == "super_resolution":
            return self._pipeline_super_resolution(data)
        if it == "inpainting":
            return self._pipeline_inpainting(data)
        if it == "video_classification":
            return self._pipeline_video_classification(data)
        if it == "denoising":
            return self._pipeline_denoising(data)
        if it == "colorization":
            return self._pipeline_colorization(data)
        if it not in ("object_detection", "pose_estimation", "classification",
                      "segmentation", "ocr_detection"):
            raise ValueError(
                f"Unknown image input_type '{it}'. "
                f"Supported: object_detection, pose_estimation, classification, "
                f"segmentation, ocr_detection, super_resolution, inpainting, "
                f"video_classification, denoising, colorization. "
                f"Use Preprocessor.register('{it}', fn) for custom types."
            )
        return self._pipeline_image(data)

    def _pipeline_image(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        w, h = cfg.resize[0], cfg.resize[1]

        if isinstance(data, np.ndarray):
            image = data
        else:
            image = decode_image(data)

        if cfg.resize_mode == "contain":
            image = letterbox(image, w, h, cfg.pad_value, getattr(cfg, "pad_mode", "constant"))
        elif cfg.resize_mode == "crop":
            image = center_crop(image, w, h)
        else:  # stretch
            image = resize_image(image, w, h, cfg.interpolation)

        # letterbox's constant-padding branch returns float32; stash a uint8 RGB
        # copy (post-resize, pre-normalize) so plugins can draw overlays on it.
        self._last_rgb_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        image = channel_reorder(image, cfg.color_format)
        image = normalize(image, cfg.mean, cfg.std)
        image = dtype_cast(image, cfg.input_dtype)

        if cfg.input_layout == "NCHW":
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        return add_batch_dim(image)
        """Preprocess a video for VideoMAE-style models.

        Expects `data` to be either:
          - Raw bytes of a video file (.mp4/.avi/.mov) — decoded with imageio/cv2
          - A numpy array of shape [T, H, W, 3] uint8 RGB frames

        Produces tensor [1, num_segments, H, W, tubelet_size*3] float32.
        VideoMAE packs `tubelet_size` consecutive frames into the channel dim:
          shape [1, 16, 224, 224, 15]  →  16 segments × 5 frames × 3 channels
        """
        cfg = self._config
        target_w, target_h = cfg.resize        # e.g. [224, 224]
        num_segments  = cfg.num_frames         # 16 (temporal segments)
        tubelet       = cfg.tubelet_size       # 5 (frames per tubelet)
        total_frames  = num_segments * tubelet  # 80

        if isinstance(data, np.ndarray):
            frames = data  # [T, H, W, 3] uint8
        else:
            frames = _decode_video_frames(data, total_frames)

        # Uniformly sample exactly `total_frames` frames from the clip
        frames = _sample_frames(frames, total_frames)

        # Short-side resize to 256, then center-crop to target_h × target_w
        frames = _video_short_side_resize(frames, 256, cfg.interpolation)
        frames = _video_center_crop(frames, target_w, target_h)

        # Normalize: /255, subtract mean, divide std; then convert to float32
        mean = np.array(cfg.mean, dtype=np.float32)
        std  = np.array(cfg.std,  dtype=np.float32)
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - mean) / std          # [T, H, W, 3]

        frames = frames.astype(cfg.input_dtype)

        # Reshape [T, H, W, 3] → [num_segments, tubelet, H, W, 3]
        # then rearrange to [num_segments, H, W, tubelet*3]
        T, H, W, C = frames.shape
        frames = frames.reshape(num_segments, tubelet, H, W, C)  # [16, 5, 224, 224, 3]
        frames = frames.transpose(0, 2, 3, 1, 4)                 # [16, 224, 224, 5, 3]
        frames = frames.reshape(num_segments, H, W, tubelet * C) # [16, 224, 224, 15]

        return frames[np.newaxis]  # [1, 16, 224, 224, 15]

    def _pipeline_inpainting(self, data: Union[bytes, np.ndarray, "tuple"]) -> "tuple":
        """Return (image_tensor, mask_tensor) for inpainting models (e.g. AOT-GAN).

        Verified from qualcomm/ai-hub-models v0.56.0 repaint/utils.py:
          - image: PIL → uint8 → /255 → float32 [0,1], RGB, NHWC [1,H,W,3]
          - mask:  PIL → grayscale → /255 > 0 → binary float32 [0,1], NHWC [1,H,W,1]
          - No mean/std subtraction (model.forward() does its own 2x-1 normalisation
            internally, which is baked into the ONNX graph)

        data may be:
          - (image_bytes_or_ndarray, mask_bytes_or_ndarray_or_None) tuple
          - plain image bytes (mask auto-generated as a centre 50%×50% rectangle)
        """
        cfg = self._config
        w, h = cfg.resize[0], cfg.resize[1]

        if isinstance(data, (bytes, np.ndarray)):
            image_data, mask_data = data, None
        else:
            image_data, mask_data = data[0], data[1]

        # ── Image ──────────────────────────────────────────────────────────────
        if isinstance(image_data, np.ndarray):
            image = image_data
        else:
            image = decode_image(image_data)          # uint8 HWC RGB

        image = resize_image(image, w, h, cfg.interpolation)
        img_f = image.astype(np.float32) / 255.0      # [H, W, 3] float32 [0,1]
        img_f = np.expand_dims(img_f, 0)              # [1, H, W, 3]

        # ── Mask ───────────────────────────────────────────────────────────────
        if mask_data is None:
            # Default: centre 50%×50% square
            cx, cy = w // 2, h // 2
            hw, hh = w // 4, h // 4
            mask_f = np.zeros((h, w, 1), dtype=np.float32)
            mask_f[cy - hh:cy + hh, cx - hw:cx + hw, 0] = 1.0
        elif isinstance(mask_data, np.ndarray):
            if mask_data.ndim == 3:
                mask_gray = mask_data[:, :, 0]
            else:
                mask_gray = mask_data
            mask_gray = resize_image(mask_gray.astype(np.uint8), w, h, cfg.interpolation)
            mask_f = (mask_gray.astype(np.float32) / 255.0 > 0.0).astype(np.float32)
            mask_f = mask_f[:, :, np.newaxis]         # [H, W, 1]
        else:
            mask_raw = decode_image(mask_data)        # uint8 HWC RGB
            mask_gray = mask_raw[:, :, 0]             # take first channel
            mask_gray = resize_image(mask_gray.astype(np.uint8), w, h, cfg.interpolation)
            mask_f = (mask_gray.astype(np.float32) / 255.0 > 0.0).astype(np.float32)
            mask_f = mask_f[:, :, np.newaxis]         # [H, W, 1]

        mask_f = np.expand_dims(mask_f, 0)            # [1, H, W, 1]
        return img_f, mask_f

    def _pipeline_denoising(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Preprocess for luminance-channel denoising models (e.g. DnCNN).

        Algorithm:
          1. Decode to RGB, resize to model input resolution.
          2. Store the resized float32 RGB on self._last_model_rgb for delta
             reconstruction in the postprocessor.
          3. Compute Y (luminance) using BT.601 full-range coefficients:
               Y = 0.299*R + 0.587*G + 0.114*B  (all in [0, 1]).
             This avoids the studio-swing offset in PIL's YCbCr conversion.
          4. Return Y as float32 NHWC [1, H, W, 1].
        """
        from PIL import Image

        cfg = self._config
        target_size = cfg.resize  # [W, H]
        if target_size is None:
            raise ValueError("denoising input_type requires 'resize: [W, H]' in config.")
        model_w, model_h = target_size[0], target_size[1]

        if isinstance(data, np.ndarray):
            pil_rgb = Image.fromarray(data.astype(np.uint8) if data.dtype != np.uint8 else data)
        else:
            pil_rgb = Image.open(io.BytesIO(data)).convert("RGB")

        # Resize to model input resolution and store as float32 [0, 1]
        pil_model = pil_rgb.resize((model_w, model_h), Image.BILINEAR)
        rgb_f = np.array(pil_model, dtype=np.float32) / 255.0  # [H, W, 3]
        self._last_model_rgb = rgb_f

        # BT.601 full-range luminance — no studio-swing offset
        y = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])
        tensor = y.astype(cfg.input_dtype)[np.newaxis, :, :, np.newaxis]  # [1, H, W, 1]
        return tensor

    def _pipeline_colorization(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Preprocess for Lab-space colorization models (e.g. DDColor).

        Algorithm:
          1. Decode to RGB, resize to model input resolution.
          2. Store the resized float32 RGB on self._last_model_rgb so the
             postprocessor can merge the predicted AB channels back.
          3. Convert RGB [0,1] → CIE-Lab.  L [0,100] → L/100 ∈ [0,1].
             AI Hub metadata value_range [0,1] confirms normalization is baked
             into the QNN graph — we send raw L/100 without external norm.
          4. Replicate L 3× → [H,W,3], return [1,H,W,3] NHWC float32.
        """
        cfg = self._config
        w, h = cfg.resize[0], cfg.resize[1]

        if isinstance(data, np.ndarray):
            image = data.astype(np.uint8) if data.dtype != np.uint8 else data
        else:
            image = decode_image(data)                    # [H, W, 3] uint8 RGB

        image = resize_image(image, w, h, cfg.interpolation)  # [H, W, 3] uint8
        rgb_f = image.astype(np.float32) / 255.0         # [H, W, 3] float32 [0,1]
        self._last_model_rgb = rgb_f                     # stash for postprocessor

        lab = _rgb_to_lab(rgb_f)                         # [H, W, 3] float32 Lab
        l_norm = (lab[:, :, 0] / 100.0).astype(np.float32)   # [H, W] in [0,1]
        l_3ch  = np.stack([l_norm, l_norm, l_norm], axis=-1)  # [H, W, 3]
        return l_3ch.astype(cfg.input_dtype)[np.newaxis]  # [1, H, W, 3] NHWC

    def _pipeline_super_resolution(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        cfg = self._config
        w, h = cfg.resize[0], cfg.resize[1]

        if isinstance(data, np.ndarray):
            image = data
        else:
            image = decode_image(data)

        image = resize_image(image, w, h, cfg.interpolation)
        image = channel_reorder(image, cfg.color_format)
        image = normalize(image, cfg.mean, cfg.std)
        image = dtype_cast(image, cfg.input_dtype)

        if cfg.input_layout == "NCHW":
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))

        return add_batch_dim(image)


# ── Private helpers ───────────────────────────────────────────────────────────


def prepare_recognizer_crop(
    gray_image: np.ndarray,
    box,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Crop a text region from a grayscale image and resize it for the EasyOCR recognizer.

    gray_image : [H, W] uint8 grayscale
    box        : (xmin, xmax, ymin, ymax) or ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    target_h/w : recognizer input height/width (e.g. 64, 800)

    Returns [1, target_h, target_w, 1] float32 in [0, 1], left-aligned with top-left-pixel padding.
    """
    from PIL import Image as _Image

    if isinstance(box[0], (tuple, list)):
        rect = np.array(box, dtype="float32")
        tl, tr, br, bl = rect
        max_w = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        ))
        max_h = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        ))
        if max_w < 1 or max_h < 1:
            cutout = np.zeros((1, 1), dtype=np.uint8)
        else:
            # PIL transform: source coords (x1,y1,...,x4,y4) → destination corners
            src = rect.flatten().tolist()   # [x1,y1, x2,y2, x3,y3, x4,y4]
            # PIL PERSPECTIVE needs coefficients; use QUAD transform instead
            pil_gray = _Image.fromarray(gray_image)
            dst = [0, 0, max_w, 0, max_w, max_h, 0, max_h]  # destination quad
            cutout = np.array(
                pil_gray.transform(
                    (max_w, max_h),
                    _Image.QUAD,
                    src,
                    resample=_Image.BILINEAR,
                ),
                dtype=np.uint8,
            )
    else:
        # Horizontal box (xmin, xmax, ymin, ymax)
        xmin = max(0, int(box[0]))
        xmax = min(int(box[1]), gray_image.shape[1])
        ymin = max(0, int(box[2]))
        ymax = min(int(box[3]), gray_image.shape[0])
        cutout = gray_image[ymin:ymax, xmin:xmax]

    if cutout.size == 0:
        return np.zeros((1, target_h, target_w, 1), dtype=np.float32)

    # Resize preserving aspect ratio (width-constrained), pad on the right
    h, w = cutout.shape[:2]
    scale = target_h / h
    new_w = min(int(w * scale), target_w)
    pil = _Image.fromarray(cutout).resize((new_w, target_h), _Image.BILINEAR)
    resized = np.array(pil, dtype=np.float32) / 255.0

    # Use top-left pixel value as pad colour (matches EasyOCRApp.prepare_recognizer_input)
    pad_val = float(resized[0, 0]) if resized.size > 0 else 0.0
    canvas = np.full((target_h, target_w), pad_val, dtype=np.float32)
    canvas[:, :new_w] = resized

    # [H, W] → [1, H, W, 1]  (NHWC, grayscale)
    return canvas[np.newaxis, :, :, np.newaxis]


def _load_vocab(vocab_file: str) -> Dict[str, int]:
    import json
    if not vocab_file:
        return {}
    with open(vocab_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data
    # list format: [token, ...]
    return {tok: idx for idx, tok in enumerate(data)}
