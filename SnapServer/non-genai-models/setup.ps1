# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
<#
.SYNOPSIS
    Full setup script for AIHub on-device models.

.DESCRIPTION
    1. Writes a lean common requirements.txt (libraries shared by ALL models).
    2. Writes per-model requirements.txt files (only the extra deps each model adds).
    3. Finds Python 3.12 AMD64 (x86-64) on the machine.
    4. Creates a virtual environment (.venv) inside this repo.
    5. Activates it and installs everything.

.NOTES
    Run from the repo root:
        Set-ExecutionPolicy -Scope Process Bypass
        .\setup.ps1

    Re-run at any time to refresh the requirements files and reinstall.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot

# ─── Colour helpers ────────────────────────────────────────────────────────────
function Write-Step  { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan   }
function Write-OK    { param([string]$Msg) Write-Host "    [OK]  $Msg" -ForegroundColor Green  }
function Write-Info  { param([string]$Msg) Write-Host "    [--]  $Msg" -ForegroundColor Gray   }
function Write-Warn  { param([string]$Msg) Write-Host "    [!!]  $Msg" -ForegroundColor Yellow }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Common requirements.txt (libraries used by every model)
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Writing common requirements.txt"

$CommonReqs = @"
# requirements.txt — common libraries required by ALL models
#
# Install this first before any per-model requirements.txt:
#   pip install -r requirements.txt
#
# Libraries shared by every model:
#   numpy         — tensor operations in pipeline_core (pre/postprocessor)
#   Pillow        — image decode / resize / encode (decode_image, letterbox, etc.)
#   onnxruntime   — ONNX inference sessions (CPU execution provider)
#   onnxruntime-qnn — Qualcomm QNN execution provider (pass --qnn flag to enable)
#   PyYAML        — config.yaml and models_prepost.yaml loading
#   setuptools    — runtime dependency of several packages

numpy==2.4.6
Pillow==12.2.0
onnxruntime==1.24.4
onnxruntime-qnn==2.2.0
PyYAML==6.0.3
setuptools==81.0.0
"@

Set-Content -Path "$Root\requirements.txt" -Value $CommonReqs -Encoding UTF8
Write-OK "requirements.txt written"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Per-model requirements.txt files
#
# Only lists libraries that are NOT in the common requirements.txt above.
# Models that need nothing extra get an explicit "no additional requirements"
# file so the intent is documented.
#
# Dependency map (based on pipeline_core source analysis):
#
#  yamnet          — librosa (mel_spectrogram), soundfile, scipy (resample_poly)
#  whisper_base    — torch + torchaudio (whisper_log_mel uses torch.stft /
#                    torchaudio.functional.melscale_fbanks), soundfile, scipy
#  zipformer       — torch + torchaudio (kaldi_fbank uses
#                    torchaudio.compliance.kaldi.fbank), soundfile, scipy
#  centernet_pose  — scipy (scipy.ndimage.maximum_filter in _heatmap_nms)
#  easyocr         — scipy (scipy.ndimage.label + binary_dilation in CRAFT post)
#  All other image models (inception_v3, mobilenet_v2, yolov3,
#  unet_segmentation, aotgan) — no extra deps beyond common
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Writing per-model requirements.txt files"

# Helper: write a model-specific requirements.txt only when the folder exists
function Write-ModelReqs {
    param(
        [string]$ModelDir,   # subfolder name under $Root
        [string]$Content     # file content
    )
    $Dir = Join-Path $Root $ModelDir
    if (-not (Test-Path $Dir)) {
        Write-Warn "Folder '$ModelDir' not found — skipping"
        return
    }
    $File = Join-Path $Dir "requirements.txt"
    Set-Content -Path $File -Value $Content -Encoding UTF8
    Write-OK "$ModelDir\requirements.txt"
}

# ── YamNet ─────────────────────────────────────────────────────────────────────
Write-ModelReqs "yamnet" @"
# yamnet/requirements.txt — extra deps for YamNet (AudioSet event classification)
# Also install the common requirements.txt first:
#   pip install -r ../requirements.txt
#
# librosa  : mel_spectrogram() in pipeline_core/preprocessor.py
# soundfile: decode_audio() reads .wav/.flac via sf.read()
# scipy    : resample_poly() in preprocessor.resample()
#            (transitive librosa deps installed automatically by pip)

librosa==0.11.0
soundfile==0.14.0
scipy==1.17.1
"@

# ── Whisper Base ───────────────────────────────────────────────────────────────
Write-ModelReqs "whisper_base" @"
# whisper_base/requirements.txt — extra deps for Whisper Base (ASR)
# Also install the common requirements.txt first:
#   pip install -r ../requirements.txt
#
# torch      : whisper_log_mel() uses torch.stft + torch.hann_window
# torchaudio : whisper_log_mel() uses torchaudio.functional.melscale_fbanks
# soundfile  : decode_audio() reads .wav/.flac via sf.read()
# scipy      : resample_poly() in preprocessor.resample()

torch==2.12.0
torchaudio==2.11.0
soundfile==0.14.0
scipy==1.17.1
"@

# ── Zipformer ──────────────────────────────────────────────────────────────────
Write-ModelReqs "zipformer" @"
# zipformer/requirements.txt — extra deps for Zipformer (streaming RNN-T ASR)
# Also install the common requirements.txt first:
#   pip install -r ../requirements.txt
#
# torch      : kaldi_fbank_features() uses torch.from_numpy / torch.Tensor
# torchaudio : kaldi_fbank_features() uses torchaudio.compliance.kaldi.fbank
# soundfile  : decode_audio() reads .wav/.flac via sf.read()
# scipy      : resample_poly() in preprocessor.resample()

torch==2.12.0
torchaudio==2.11.0
soundfile==0.14.0
scipy==1.17.1
"@

# ── Inception V3 ───────────────────────────────────────────────────────────────
Write-ModelReqs "inception_v3" @"
# inception_v3/requirements.txt — extra deps for Inception V3
# No additional requirements beyond the common requirements.txt.
# Install common deps only:
#   pip install -r ../requirements.txt
"@

# ── MobileNet V2 ───────────────────────────────────────────────────────────────
Write-ModelReqs "mobilenet_v2" @"
# mobilenet_v2/requirements.txt — extra deps for MobileNet V2
# No additional requirements beyond the common requirements.txt.
# Install common deps only:
#   pip install -r ../requirements.txt
"@

# ── YOLOv3 ─────────────────────────────────────────────────────────────────────
Write-ModelReqs "yolov3" @"
# yolov3/requirements.txt — extra deps for YOLOv3
# No additional requirements beyond the common requirements.txt.
# Install common deps only:
#   pip install -r ../requirements.txt
"@

# ── CenterNet Pose ─────────────────────────────────────────────────────────────
Write-ModelReqs "centernet_pose" @"
# centernet_pose/requirements.txt — extra deps for CenterNet-Pose
# Also install the common requirements.txt first:
#   pip install -r ../requirements.txt
#
# scipy : _heatmap_nms() uses scipy.ndimage.maximum_filter
#         to suppress non-local-maximum peaks in person-center heatmap

scipy==1.17.1
"@

# ── UNet Segmentation ──────────────────────────────────────────────────────────
Write-ModelReqs "unet_segmentation" @"
# unet_segmentation/requirements.txt — extra deps for UNet Segmentation
# No additional requirements beyond the common requirements.txt.
# Install common deps only:
#   pip install -r ../requirements.txt
"@

# ── EasyOCR ────────────────────────────────────────────────────────────────────
Write-ModelReqs "easyocr" @"
# easyocr/requirements.txt — extra deps for EasyOCR (text detection + recognition)
# Also install the common requirements.txt first:
#   pip install -r ../requirements.txt
#
# scipy : _craft_get_det_boxes() uses scipy.ndimage.label (connected components)
#         and scipy.ndimage.binary_dilation (region dilation for CRAFT detector)

scipy==1.17.1
"@

# ── AOT-GAN ────────────────────────────────────────────────────────────────────
Write-ModelReqs "aotgan" @"
# aotgan/requirements.txt — extra deps for AOT-GAN (image inpainting)
# No additional requirements beyond the common requirements.txt.
# Install common deps only:
#   pip install -r ../requirements.txt
"@


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Find Python 3.12 AMD64 (x86-64)
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Locating Python 3.12 AMD64"

$Python = $null

# 1. Try Python Launcher with explicit 64-bit x86-64 selector
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $ver = & py -3.12-64 --version 2>&1
    if ($LASTEXITCODE -eq 0 -and $ver -match "3\.12") {
        $Python = "py -3.12-64"
        Write-OK "Found via Python Launcher: $ver (AMD64)"
    }
    if (-not $Python) {
        # Fall back to any 3.12 via the launcher
        $ver = & py -3.12 --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $ver -match "3\.12") {
            $Python = "py -3.12"
            Write-OK "Found via Python Launcher: $ver"
        }
    }
}

# 2. Scan common AMD64 installation directories
if (-not $Python) {
    $Candidates = @(
        "C:\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:PROGRAMFILES\Python312\python.exe",
        "$env:PROGRAMFILES\Python\Python312\python.exe"
    )
    foreach ($Path in $Candidates) {
        if (Test-Path $Path) {
            $ver = & $Path --version 2>&1
            if ($ver -match "3\.12") {
                $Python = $Path
                Write-OK "Found at: $Path — $ver"
                break
            }
        }
    }
}

if (-not $Python) {
    Write-Host ""
    Write-Host "ERROR: Python 3.12 AMD64 not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Download and install from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/release/python-3129/" -ForegroundColor Yellow
    Write-Host "  Select: Windows installer (64-bit)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Then re-run this script." -ForegroundColor Yellow
    exit 1
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Create virtual environment
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Creating virtual environment (.venv)"

$VenvDir = Join-Path $Root ".venv"

if (Test-Path $VenvDir) {
    Write-Info ".venv already exists — skipping creation"
} else {
    if ($Python -match "^py ") {
        # Python Launcher invocation (e.g. "py -3.12-64")
        $pyArgs = ($Python -replace "^py ","").Split(" ") + @("-m", "venv", $VenvDir)
        & py @pyArgs
    } else {
        & $Python -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: venv creation failed." -ForegroundColor Red
        exit 1
    }
    Write-OK "Virtual environment created at $VenvDir"
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Activate virtual environment
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Activating virtual environment"

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "ERROR: Activate.ps1 not found at $ActivateScript" -ForegroundColor Red
    exit 1
}

. $ActivateScript
Write-OK "Activated — python now resolves to: $(Get-Command python | Select-Object -ExpandProperty Source)"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Upgrade pip, wheel, setuptools
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Upgrading pip / wheel / setuptools"

python -m pip install --quiet --upgrade pip wheel
python -m pip install --quiet "setuptools==81.0.0"
Write-OK "pip $(python -m pip --version)"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Install common requirements
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Installing common requirements (requirements.txt)"

python -m pip install -r "$Root\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: common requirements install failed." -ForegroundColor Red
    exit 1
}
Write-OK "Common requirements installed"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Install per-model requirements
# Installs only if the model folder exists AND its requirements.txt has at least
# one non-comment, non-empty line.
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Installing per-model requirements"

$ModelDirs = @(
    "yamnet",
    "whisper_base",
    "zipformer",
    "inception_v3",
    "mobilenet_v2",
    "yolov3",
    "centernet_pose",
    "unet_segmentation",
    "easyocr",
    "aotgan"
)

foreach ($Model in $ModelDirs) {
    $ReqFile = Join-Path $Root "$Model\requirements.txt"
    if (-not (Test-Path $ReqFile)) {
        Write-Warn "$Model — requirements.txt missing, skipping"
        continue
    }

    # Check if any non-comment, non-blank lines exist (i.e., actual packages)
    $HasPackages = @(Get-Content $ReqFile |
        Where-Object { $_ -notmatch "^\s*#" -and $_ -match "\S" }).Count -gt 0

    if (-not $HasPackages) {
        Write-Info "$Model — no extra packages (common deps only)"
        continue
    }

    Write-Info "Installing $Model extras..."
    python -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: install failed for $Model" -ForegroundColor Yellow
    } else {
        Write-OK "$Model extras installed"
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Smoke-test: verify key imports
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Verifying key imports"

$Checks = @(
    @{ Label="numpy";          Code="import numpy; print(numpy.__version__)" },
    @{ Label="Pillow";         Code="import PIL; print(PIL.__version__)" },
    @{ Label="onnxruntime";    Code="import onnxruntime; print(onnxruntime.__version__)" },
    @{ Label="PyYAML";         Code="import yaml; print(yaml.__version__)" },
    @{ Label="scipy";          Code="import scipy; print(scipy.__version__)" },
    @{ Label="soundfile";      Code="import soundfile; print(soundfile.__version__)" },
    @{ Label="torch";          Code="import torch; print(torch.__version__)" },
    @{ Label="torchaudio";     Code="import torchaudio; print(torchaudio.__version__)" },
    @{ Label="librosa";        Code="import librosa; print(librosa.__version__)" }
)

foreach ($Check in $Checks) {
    $Result = python -c $Check.Code 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "$($Check.Label) $Result"
    } else {
        Write-Warn "$($Check.Label) not available (only needed by some models)"
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "To activate this environment in a new shell:" -ForegroundColor Cyan
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Quick-start examples:" -ForegroundColor Cyan
Write-Host "    python run_model.py --model yamnet              --input audio.wav" -ForegroundColor White
Write-Host "    python run_model.py --model whisper-base        --input audio.wav" -ForegroundColor White
Write-Host "    python run_model.py --model inception-v3        --input photo.jpg" -ForegroundColor White
Write-Host "    python run_model.py --model yolov3              --input photo.jpg" -ForegroundColor White
Write-Host "    python run_model.py --model aotgan              --input photo.jpg --mask mask.png" -ForegroundColor White
Write-Host ""
Write-Host "Add --qnn to any command to run on the Snapdragon X Elite NPU." -ForegroundColor Yellow
Write-Host ""
