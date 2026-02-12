@echo off
REM ============================================================
REM  Public PitchBook Observer - Start API Server
REM ============================================================

echo.
echo  ======================================================
echo   Public PitchBook Observer - API Server
echo  ======================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12+ and try again.
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%~dp0"

echo Starting API server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
