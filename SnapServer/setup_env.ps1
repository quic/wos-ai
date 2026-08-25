#Requires -Version 5.1
<#
.SYNOPSIS
    Environment setup for OpenAI-Compatible Server.
    Checks for Python 3.10+, creates .venv, installs requirements.txt

.USAGE
    powershell -ExecutionPolicy Bypass -File setup_env.ps1
    (or double-click setup_env.bat)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param($msg) Write-Host "" ; Write-Host "[STEP] $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "[INFO] $msg" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Info "Working directory: $ScriptDir"

# =============================================================================
# STEP 1 - Find Python 3.10+
# =============================================================================
Write-Step "Checking for Python 3.10+"

function Find-Python {
    $candidates = @("python", "python3", "py")
    foreach ($cmd in $candidates) {
        $exe = Get-Command $cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $ver = & $exe.Source -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>$null
            if ($ver -match "^(\d+)\.(\d+)$") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 10) {
                    return $exe.Source
                }
            }
        } catch { continue }
    }

    # Check common install paths
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            try {
                $ver = & $p -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>$null
                if ($ver -match "^(\d+)\.(\d+)$") {
                    $major = [int]$Matches[1]
                    $minor = [int]$Matches[2]
                    if ($major -ge 3 -and $minor -ge 10) { return $p }
                }
            } catch { continue }
        }
    }
    return $null
}

$PythonExe = Find-Python

if ($PythonExe) {
    $ver = & $PythonExe -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor) + '.' + str(sys.version_info.micro))" 2>$null
    Write-OK "Found Python $ver at: $PythonExe"
} else {
    Write-Warn "Python 3.10+ not found."
    Write-Host ""
    Write-Host "  Options:" -ForegroundColor White
    Write-Host "    [1] Auto-download Python 3.12.9 (x64) from python.org" -ForegroundColor White
    Write-Host "    [2] Open python.org in browser" -ForegroundColor White
    Write-Host "    [3] Exit (install Python manually then re-run)" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "Enter choice (1/2/3)"

    if ($choice.Trim() -eq "1") {
        Write-Step "Downloading Python 3.12.9 (x64)"
        $PyVersion = "3.12.9"
        $PyUrl = "https://www.python.org/ftp/python/" + $PyVersion + "/python-" + $PyVersion + "-amd64.exe"
        $PyInstaller = Join-Path $env:TEMP ("python-" + $PyVersion + "-amd64.exe")

        Write-Info "Downloading: $PyUrl"
        try {
            $curlExe = Get-Command curl.exe -ErrorAction SilentlyContinue
            if ($curlExe) {
                & curl.exe -L --progress-bar -o $PyInstaller $PyUrl
            } else {
                $ProgressPreference = "SilentlyContinue"
                Invoke-WebRequest -Uri $PyUrl -OutFile $PyInstaller -UseBasicParsing
            }
            Write-OK "Downloaded to: $PyInstaller"
        } catch {
            Write-Host "[FAIL] Download failed: $_" -ForegroundColor Red
            Write-Host "       Install Python manually from https://python.org" -ForegroundColor Yellow
            exit 1
        }

        Write-Step "Installing Python 3.12.9 (x64) silently"
        $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0 Include_launcher=1"
        Start-Process -FilePath $PyInstaller -ArgumentList $installArgs -Wait -NoNewWindow
        Remove-Item $PyInstaller -Force -ErrorAction SilentlyContinue

        # Refresh PATH for this session using enum (avoids smart-quote issues)
        $machinePath = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::Machine)
        $userPath    = [System.Environment]::GetEnvironmentVariable("PATH", [System.EnvironmentVariableTarget]::User)
        $env:PATH = $machinePath + ";" + $userPath

        $PythonExe = Find-Python
        if (-not $PythonExe) {
            Write-Warn "Python installed but not yet in PATH."
            Write-Warn "Close this terminal, open a new one, and re-run setup_env.bat"
            pause
            exit 0
        }
        $ver = & $PythonExe -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor) + '.' + str(sys.version_info.micro))" 2>$null
        Write-OK "Python $ver installed at: $PythonExe"

    } elseif ($choice.Trim() -eq "2") {
        Write-Info "Opening https://www.python.org/downloads/ ..."
        Start-Process "https://www.python.org/downloads/"
        Write-Host ""
        Write-Host "  After installing Python, re-run: setup_env.bat" -ForegroundColor Yellow
        pause
        exit 0
    } else {
        Write-Host ""
        Write-Host "  Install Python 3.10+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Then re-run: setup_env.bat" -ForegroundColor Yellow
        pause
        exit 0
    }
}

# =============================================================================
# STEP 2 - Create virtual environment
# =============================================================================
Write-Step "Setting up virtual environment"

$VenvDir = Join-Path $ScriptDir ".venv"
$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"

if (Test-Path $VenvActivate) {
    Write-Info "Virtual environment already exists at .venv"
} else {
    Write-Info "Creating .venv ..."
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-OK "Virtual environment created at .venv"
}

. $VenvActivate
Write-OK "Activated: $env:VIRTUAL_ENV"

# =============================================================================
# STEP 3 - Upgrade pip
# =============================================================================
Write-Step "Upgrading pip"
& python -m pip install --upgrade pip --quiet
Write-OK "pip upgraded"

# =============================================================================
# STEP 4 - Install requirements
# =============================================================================
Write-Step "Installing packages from requirements.txt"

$ReqFile = Join-Path $ScriptDir "requirements.txt"
if (-not (Test-Path $ReqFile)) {
    Write-Host "[FAIL] requirements.txt not found in $ScriptDir" -ForegroundColor Red
    exit 1
}

& pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Some packages failed. Check output above."
} else {
    Write-OK "All packages installed"
}

# =============================================================================
# STEP 5 - Verify
# =============================================================================
Write-Step "Verifying key packages"

$checks = @(
    @{ Name = "fastapi";      Cmd = "import fastapi; print(fastapi.__version__)" },
    @{ Name = "uvicorn";      Cmd = "import uvicorn; print(uvicorn.__version__)" },
    @{ Name = "openai";       Cmd = "import openai; print(openai.__version__)" },
    @{ Name = "pydantic";     Cmd = "import pydantic; print(pydantic.__version__)" },
    @{ Name = "tiktoken";     Cmd = "import tiktoken; print(tiktoken.__version__)" },
    @{ Name = "numpy";        Cmd = "import numpy; print(numpy.__version__)" },
    @{ Name = "streamlit";    Cmd = "import streamlit; print(streamlit.__version__)" }
)

foreach ($c in $checks) {
    try {
        $v = & python -c $c.Cmd 2>$null
        if ($v) {
            Write-OK "$($c.Name) $v"
        } else {
            Write-Warn "$($c.Name) - installed (version unknown)"
        }
    } catch {
        Write-Warn "$($c.Name) - not installed"
    }
}

# =============================================================================
# DONE
# =============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " To start the server:"
Write-Host "   .venv\Scripts\Activate.ps1"
Write-Host "   python server.py"
Write-Host ""
Write-Host " Edit config\models.yaml to configure your models."
Write-Host ""
