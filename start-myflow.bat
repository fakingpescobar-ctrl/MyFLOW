@echo off
rem ============================================================
rem  MyFLOW: odno okno - TUI (panel, mikrofon, servera).
rem  Servers (llama/whisper) podnimaet sam i GASIT PRI VYHODE:
rem  nichego ne ostayotsya v pamyati/VRAM (pri ige eto vazhno).
rem  Logi: data\*.log
rem ============================================================
title MyFLOW
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem --- okno-interfeys s upravleniem (sm. ui.bat) ---
set "MYFLOW_WEBUI=1"

rem --- 0. esli MyFLOW uzhe zapushchen (inache dva PTT na odnu knopku) ---
rem     proveryaem imenno prilozhenie (modul tui): server interfeysa
rem     (myflow.webui) rabotaet ryadom i meshat ne dolzhen
powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*myflow.tui*' }; if ($p) { exit 2 } else { exit 0 }" >nul 2>&1
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

rem --- 2. ubiraem hvosty ot proshlogo zapuska (posle krasha/ubijstva processa) ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-servers.ps1" -Quiet >nul 2>&1

rem --- 3. razmer okna iz nastroek: klassicheskaya konsol (ConsoleHost) ne
rem  pomnit razmer mezhdu zapuskami i otkryvaetsya 120x30 - stavim svoi.
rem  Menyat razmer mozhno rastyanuv okno myshyu (zapomnitsya) ili strokoy
rem  "Razmer okna" v panele.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\apply-window-size.ps1" >nul 2>&1

rem --- 4. TUI (servery podnimaet sam, otdel'nyh okon bolshe net) ---
"%~dp0.venv\Scripts\python.exe" -X utf8 -m myflow.tui
set "RC=%ERRORLEVEL%"

rem --- 5. GASIM SERVERY POSLE VYHODA: osvobozhdaem pamyat i VRAM ---
rem  TUI gasit ih sam pri normal'nom vyhode, a etot shag dobitvaet ostavshiesya
rem  (zakrytie okna krestikom, krash, Stop-Process). Trogayutsya TOLKO
rem  processy iz papki proekta - chuzhie sborki llama.cpp ne zadety.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-servers.ps1"
if not "%RC%"=="0" (
  echo.
  echo [MyFLOW] TUI zavershilsya s kodom %RC%. Logi: data\main.log, data\llm.log
  pause
)
exit /b %RC%

