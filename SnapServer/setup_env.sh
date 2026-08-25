#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Environment Setup (Linux / macOS)
#
# What this does:
#   1. Checks if Python 3.10+ is installed
#   2. If not found, offers install instructions (apt / brew / manual)
#   3. Creates a .venv virtual environment
#   4. Installs all packages from requirements.txt
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}[STEP]${NC} $1"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
info()  { echo -e "[INFO] $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Find Python 3.10+
# ═══════════════════════════════════════════════════════════════════════════════
step "Checking for Python 3.10+"

find_python() {
    for cmd in python3 python python3.12 python3.11 python3.10; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_EXE=$(find_python || true)

if [ -n "$PYTHON_EXE" ]; then
    VER=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
    ok "Found Python $VER at: $(command -v $PYTHON_EXE)"
else
    warn "Python 3.10+ not found on this system."
    echo ""
    echo "  Install Python 3.10+ using one of these methods:"
    echo ""

    # Detect OS and show relevant instructions
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS (Homebrew):"
        echo "    brew install python@3.12"
        echo ""
        echo "  macOS (direct download):"
        echo "    https://www.python.org/ftp/python/3.12.9/python-3.12.9-macos11.pkg"
    elif command -v apt-get &>/dev/null; then
        echo "  Ubuntu / Debian:"
        echo "    sudo apt-get update"
        echo "    sudo apt-get install -y python3.12 python3.12-venv python3-pip"
    elif command -v dnf &>/dev/null; then
        echo "  Fedora / RHEL:"
        echo "    sudo dnf install -y python3.12"
    elif command -v pacman &>/dev/null; then
        echo "  Arch Linux:"
        echo "    sudo pacman -S python"
    else
        echo "  Download from: https://www.python.org/downloads/"
    fi

    echo ""
    echo "  After installing Python, re-run this script:"
    echo "    ./setup_env.sh"
    echo ""

    # Offer to try auto-install on Linux with apt
    if command -v apt-get &>/dev/null; then
        read -rp "  Try to install Python 3.12 via apt now? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            echo ""
            info "Running: sudo apt-get install -y python3.12 python3.12-venv python3-pip"
            sudo apt-get update -qq
            sudo apt-get install -y python3.12 python3.12-venv python3-pip
            PYTHON_EXE=$(find_python || true)
            if [ -z "$PYTHON_EXE" ]; then
                fail "Python still not found after install. Please install manually."
            fi
            VER=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
            ok "Python $VER installed"
        else
            exit 0
        fi
    elif [[ "$OSTYPE" == "darwin"* ]] && command -v brew &>/dev/null; then
        read -rp "  Try to install Python 3.12 via Homebrew now? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            echo ""
            info "Running: brew install python@3.12"
            brew install python@3.12
            PYTHON_EXE=$(find_python || true)
            if [ -z "$PYTHON_EXE" ]; then
                fail "Python still not found after install. Please install manually."
            fi
            VER=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
            ok "Python $VER installed"
        else
            exit 0
        fi
    else
        exit 0
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Create virtual environment
# ═══════════════════════════════════════════════════════════════════════════════
step "Setting up virtual environment"

VENV_DIR="$SCRIPT_DIR/.venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    info "Virtual environment already exists at .venv"
else
    info "Creating .venv with $PYTHON_EXE ..."
    "$PYTHON_EXE" -m venv "$VENV_DIR"
    ok "Virtual environment created at .venv"
fi

# Activate
source "$VENV_DIR/bin/activate"
ok "Activated: $VIRTUAL_ENV"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Upgrade pip
# ═══════════════════════════════════════════════════════════════════════════════
step "Upgrading pip"
pip install --upgrade pip --quiet
ok "pip upgraded"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Install requirements
# ═══════════════════════════════════════════════════════════════════════════════
step "Installing packages from requirements.txt"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    fail "requirements.txt not found in $SCRIPT_DIR"
fi

pip install -r requirements.txt
ok "All packages installed"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Verify
# ═══════════════════════════════════════════════════════════════════════════════
step "Verifying key packages"

for pkg in fastapi uvicorn openai pydantic tiktoken transformers numpy streamlit; do
    ver=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null || true)
    if [ -n "$ver" ]; then
        ok "$pkg $ver"
    else
        warn "$pkg — not installed or import failed"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo " To start the server:"
echo "   source .venv/bin/activate"
echo "   python server.py"
echo ""
echo " Edit config/models.yaml to configure your models."
echo ""
