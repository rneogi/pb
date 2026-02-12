@echo off
REM ============================================================
REM  PitchBook Observer Agent - CLI Chat
REM ============================================================

echo.
echo  ======================================================
echo   PitchBook Observer Agent - Chat
echo  ======================================================
echo.

REM Check runtime quietly
python --version >nul 2>&1
if errorlevel 1 (
    echo  The agent needs a runtime component to start.
    echo  Please install Python 3.12+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

cd /d "%~dp0"

REM Load API key from .env file if it exists
if exist "%~dp0.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
        if "%%A"=="ANTHROPIC_API_KEY" set ANTHROPIC_API_KEY=%%B
    )
)

REM If key already loaded from .env, skip the prompt
if defined ANTHROPIC_API_KEY goto :has_key

REM Prompt for API key
echo  The agent works best with Claude Opus.
echo  (Get a key at https://console.anthropic.com)
echo.
set /p USER_KEY="  Enter your API key (or press Enter to skip): "
if "%USER_KEY%"=="" goto :no_key
set ANTHROPIC_API_KEY=%USER_KEY%

:has_key
echo.
echo  Claude Opus is online.
echo.
python -m app.chat_interface --llm
goto :done

:no_key
echo.
echo  Running in offline mode (no API key).
echo  For the full experience, re-run with a key.
echo.
python -m app.chat_interface

:done
pause
