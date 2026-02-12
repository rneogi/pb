@echo off
REM ============================================================
REM  Public PitchBook Observer - Setup / Install Dependencies
REM ============================================================

echo.
echo  ======================================================
echo   Public PitchBook Observer - Setup
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

echo Python found:
python --version
echo.

REM Change to script directory
cd /d "%~dp0"

echo Installing dependencies...
echo.

pip install pyyaml httpx beautifulsoup4 lxml fastapi uvicorn pydantic numpy scikit-learn

echo.
echo Checking for optional dependencies...

REM Try to install sentence-transformers (may fail without torch)
echo.
echo Installing sentence-transformers (for embeddings)...
pip install sentence-transformers 2>nul
if errorlevel 1 (
    echo WARNING: sentence-transformers install failed - will use TF-IDF fallback
)

REM Try trafilatura
echo.
echo Installing trafilatura (for content extraction)...
pip install trafilatura 2>nul

echo.
echo ======================================================
echo   Setup Complete!
echo ======================================================
echo.
echo Next steps:
echo   1. Run 'run_pipeline.bat' to crawl and index data
echo   2. Run 'run_chat.bat' for interactive chat
echo   3. Run 'run_batch.bat' to test with demo questions
echo.

pause
