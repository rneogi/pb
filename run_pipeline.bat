@echo off
REM ============================================================
REM  Public PitchBook Observer - Run Pipeline
REM ============================================================

echo.
echo  ======================================================
echo   Public PitchBook Observer - Pipeline Runner
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

REM Get current week
for /f "tokens=*" %%a in ('python -c "from datetime import datetime; print(datetime.utcnow().strftime('%%Y-W%%W'))"') do set WEEK=%%a

echo Current week: %WEEK%
echo.

REM Parse arguments
set STAGE=
set CUSTOM_WEEK=

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--week" (
    set CUSTOM_WEEK=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--only" (
    set STAGE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--help" (
    goto help
)
shift
goto parse_args

:help
echo Usage: run_pipeline.bat [options]
echo.
echo Options:
echo   --week ^<YYYY-WW^>   Target week (default: current week)
echo   --only ^<stage^>     Run only specific stage
echo   --help             Show this help
echo.
echo Stages: crawl, clean, delta, evaluate, schedule, index, pargv_batch
echo.
echo Examples:
echo   run_pipeline.bat                     Run full pipeline for current week
echo   run_pipeline.bat --week 2026-W05     Run for specific week
echo   run_pipeline.bat --only crawl        Run only crawl stage
echo.
pause
exit /b 0

:run
if not "%CUSTOM_WEEK%"=="" set WEEK=%CUSTOM_WEEK%

echo Running pipeline for week: %WEEK%
if not "%STAGE%"=="" echo Stage: %STAGE%
echo.

if "%STAGE%"=="" (
    python -m pipeline.run --week %WEEK%
) else (
    python -m pipeline.run --week %WEEK% --only %STAGE%
)

echo.
echo Pipeline run complete.
pause
