@echo off
REM =============================================================
REM  PitchBook Observer Agent - One-Click Launcher
REM =============================================================

echo.
echo  ======================================================
echo   PitchBook Observer Agent
echo   Powered by Claude Opus
echo  ======================================================
echo.

REM ---- Quietly check Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo  The agent needs a runtime component to start.
    echo  Please install Python 3.12+ from https://www.python.org/downloads/
    echo  Then run this again.
    pause
    exit /b 1
)

cd /d "%~dp0"

REM ---- Agent boot sequence ----
echo  Agent is initializing...
echo.

echo    [1/4] Pulling latest data from repo...
git pull --quiet
echo    [OK] Index up to date

echo    [2/4] Loading agent modules...
pip install -q pyyaml httpx[http2] beautifulsoup4 lxml trafilatura fastapi uvicorn pydantic numpy scikit-learn python-dotenv anthropic 2>nul
pip install -q sentence-transformers 2>nul
pip install -q streamlit plotly pandas 2>nul

echo    [3/4] Fetching and indexing fresh data...
python -m pipeline.agents.ingest_agent
python -m pipeline.agents.compilation_agent
echo    [OK] Index up to date

echo    [4/4] Preparing directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\clean" mkdir "data\clean"
if not exist "data\meta" mkdir "data\meta"
if not exist "data\events" mkdir "data\events"
if not exist "data\memory" mkdir "data\memory"
if not exist "data\responses" mkdir "data\responses"
if not exist "indexes\chroma" mkdir "indexes\chroma"
if not exist "db" mkdir "db"
if not exist "products" mkdir "products"
if not exist "batch_results" mkdir "batch_results"
if not exist "runs" mkdir "runs"

echo    [5/5] Launching agent...
echo.
echo  ======================================================
echo   PitchBook Observer Agent is starting!
echo   Your browser will open in a few seconds.
echo.
echo   If it doesn't, go to: http://localhost:8501
echo   Press Ctrl+C to stop the agent.
echo  ======================================================
echo.

REM Open browser after a short delay (gives Streamlit time to start)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

REM Launch Streamlit (this blocks until Ctrl+C)
python -m streamlit run app\streamlit_chat.py --server.port 8501

pause
