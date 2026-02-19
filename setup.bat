@echo off
setlocal

echo.
echo  ======================================================
echo   PitchBook Observer -- SA Setup
echo  ======================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.12+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER%

:: Install dependencies
echo.
echo  [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo  [OK] Dependencies installed

:: API key setup
echo.
echo  [2/3] API Key Setup
if exist .env (
    findstr /i "ANTHROPIC_API_KEY=sk-ant-" .env >nul 2>&1
    if not errorlevel 1 (
        echo  [OK] API key already configured in .env
        goto :verify
    )
)

echo.
echo  Get your key at: https://console.anthropic.com/
echo.
set /p SA_KEY="  Enter your Anthropic API key (sk-ant-...): "
if "%SA_KEY%"=="" (
    echo  [WARN] No key entered. Running in template mode ^(no LLM synthesis^).
    goto :verify
)

(
echo # PitchBook Observer - Local Environment
echo # This file is gitignored and will NOT be committed.
echo.
echo ANTHROPIC_API_KEY=%SA_KEY%
echo CLAUDE_MODEL=claude-opus-4-6
) > .env
echo  [OK] API key saved to .env

:verify
echo.
echo  [3/3] Verifying setup...
python -c "from pipeline.agents.runtime_agent import RuntimeAgent; print('  [OK] Agent imports OK')" 2>nul
if errorlevel 1 echo  [WARN] Import check failed - re-run setup or check Python version

python -c "import pathlib; idx=pathlib.Path('indexes/chroma'); files=[f for f in idx.glob('*') if f.is_file()] if idx.exists() else []; print(f'  [OK] Index ready: {len(files)} files')" 2>nul

echo.
echo  ======================================================
echo   Setup complete! How to use:
echo.
echo   Web UI ^(recommended^):   runit.bat
echo   CLI:                     run_chat.bat
echo   Claude Code:             /pb what funding rounds were announced?
echo.
echo   To refresh data ^(admin only^):  /pipeline
echo  ======================================================
echo.
pause
endlocal
