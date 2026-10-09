@echo off
rem ============================================================
rem  MyFLOW: ???? ???? ? TUI ??? ????????? ??????? (??????),
rem  ?????????? ????? ???????? ? ????? ????????.
rem  ???????: whisper :8081 + llama :8082, ???? ? data\*.log
rem ============================================================
title MyFLOW
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem --- 0. esli MyFLOW uzhe zapushchen (inache dva PTT na odnu knopku) ---
powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*myflow*' }; if ($p) { exit 2 } else { exit 0 }" >nul 2>&1
if errorlevel 2 (
  echo [MyFLOW] uzhe zapushchen. Zakroy staroe okno MyFLOW.
  pause
  exit /b 0
)

rem --- 1. proverim okruzhenie ---
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [MyFLOW] net .venv - sozday okruzhenie:
  echo    py -3.13 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

rem --- 2. TUI (servery podnimaet sam, otdel'nyh okon bolshe net) ---
"%~dp0.venv\Scripts\python.exe" -X utf8 -m myflow.tui
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [MyFLOW] TUI zavershilsya s kodom %RC%. Logi: data\main.log, data\asr.log.err, data\llm.log.err
  pause
)
exit /b %RC%