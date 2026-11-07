@echo off
rem Fundamental Qualitative launcher (run from this folder; uv manages the environment).
rem   run.bat                              menu: 1) tickers you type  2) portfolio  3) global benchmark,
rem                                        then 1) auto (default)  2) everything  3) prices only
rem   run.bat -u portfolio                 portfolio.txt (auto: prices, full data where a report is due)
rem   run.bat -u benchmark                 the global benchmark (S&P 500 + IBEX 35 + DAX + others)
rem   run.bat -u "AAPL,MSFT,SAP.DE"        a list of tickers
rem   run.bat -u "portfolio,KO"            mixed
rem   run.bat -u portfolio --offline       reuse the cache, no downloads
rem   run.bat -u portfolio --full          re-fetch all fundamentals (slow)
rem   run.bat -u portfolio --prices-only   reuse saved fundamentals, refresh prices only (fast)
rem   run.bat -u portfolio --benchmark none   rank the portfolio on its own (default: scored vs the global benchmark)
rem   run.bat test                         the test suite
rem   run.bat fetch|rank|report|clean ...  single steps (same options)

setlocal
cd /d "%~dp0"
rem One render process unless set: two ran this 8 GB machine out of memory on a full re-render (Nov 7, 2026).
if not defined FQ_RENDER_WORKERS set "FQ_RENDER_WORKERS=1"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it from https://docs.astral.sh/uv/ and try again.
    exit /b 1
)

if /i "%~1"=="test" (
    uv run pytest -q
    exit /b %ERRORLEVEL%
)

set "FIRST=%~1"
if "%FIRST%"=="" (
    uv run fq run --open
) else if "%FIRST:~0,1%"=="-" (
    uv run fq run --open %*
) else (
    uv run fq %*
)
set "CODE=%ERRORLEVEL%"
endlocal & exit /b %CODE%
