@echo off
REM ============================================================
REM  Public PitchBook Observer - Batch Demo Runner
REM ============================================================

echo.
echo  ======================================================
echo   Public PitchBook Observer - Batch Demo Runner
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

REM Parse arguments
set CATEGORY=all
set LIMIT=0
set MODE=hybrid
set TOPK=8

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--category" (
    set CATEGORY=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--limit" (
    set LIMIT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--mode" (
    set MODE=%~2
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
echo Usage: run_batch.bat [options]
echo.
echo Options:
echo   --category ^<cat^>   Question category (deals_funding, deals_ma, investor, company, trend, general, all)
echo   --limit ^<n^>        Limit number of questions (0 = all)
echo   --mode ^<mode^>      Retrieval mode (hybrid, vector, keyword)
echo   --help             Show this help
echo.
echo Examples:
echo   run_batch.bat                           Run all questions
echo   run_batch.bat --category deals_funding  Run only deals_funding questions
echo   run_batch.bat --limit 20                Run first 20 questions
echo   run_batch.bat --mode vector             Use vector-only retrieval
echo.
pause
exit /b 0

:run
echo Running batch with:
echo   Category: %CATEGORY%
echo   Limit: %LIMIT% (0 = all)
echo   Mode: %MODE%
echo   Top-K: %TOPK%
echo.

python -m app.batch_runner --category %CATEGORY% --limit %LIMIT% --mode %MODE% --top-k %TOPK%

echo.
echo Batch run complete. Results saved to batch_results/
pause
