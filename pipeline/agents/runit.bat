@echo off
REM =============================================================
REM  PitchBook Observer Agent - Launch from Agent Hub
REM =============================================================

echo.
echo  ======================================================
echo   PitchBook Observer Agent - Starting...
echo  ======================================================
echo.

REM Navigate to project root (two levels up)
cd /d "%~dp0..\.."

REM Delegate to the main launcher
if exist runit.bat (
    call runit.bat
) else (
    echo  Agent could not locate its modules.
    echo  Make sure the full project structure is intact.
    pause
    exit /b 1
)
