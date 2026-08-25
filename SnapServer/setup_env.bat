@echo off
REM ============================================================
REM  setup_env.bat — Environment Setup
REM
REM  What this does:
REM    1. Checks if Python 3.10+ is installed
REM    2. If not found, offers to auto-download Python 3.12 (x64)
REM    3. Creates a .venv virtual environment
REM    4. Installs all packages from requirements.txt
REM
REM  Usage:
REM    Double-click this file, or run from cmd:
REM      setup_env.bat
REM ============================================================

setlocal

REM Check PowerShell is available
where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell not found. Please install PowerShell and retry.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Environment Setup
echo ============================================================
echo.

REM Run the PowerShell setup script
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0setup_env.ps1"

if errorlevel 1 (
    echo.
    echo [ERROR] Setup encountered an error. See output above.
    echo.
    echo If you see an "execution policy" error, run manually:
    echo   powershell -ExecutionPolicy Bypass -File setup_env.ps1
    echo.
    pause
    exit /b 1
)

echo.
pause
endlocal