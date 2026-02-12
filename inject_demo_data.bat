@echo off
REM ============================================================
REM  Public PitchBook Observer - Inject Demo Data
REM ============================================================

echo.
echo  ======================================================
echo   Public PitchBook Observer - Demo Data Injection
echo  ======================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%~dp0"

python scripts/inject_demo_data.py

pause
