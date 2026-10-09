@echo off
rem ============================================================
rem  MyFLOW launcher: servers (whisper :8081 + llama :8082) + main
rem  Puti beryutsya ot raspolozheniya fayla (%~dp0), ne hardkodyatsya.
rem  Tekst v ASCII: cmd.exe nenadezhno chitaet UTF-8 v .bat
rem ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem --- 0. esli main uzhe zapushchen (inache dva PTT na odnu knopku) ---
powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*myflow*' }; if ($p) { exit 2 } else { exit 0 }" >nul 2>&1
if errorlevel 2 (
  echo [MyFLOW] main uzhe zapushchen. Zakroy staroe okno "MyFLOW MAIN".
  pause
  exit /b 0
)

rem --- 1. servery (fonovo, vidimoe okno-pult) ---
set "PS=pwsh"
where pwsh >nul 2>&1 || set "PS=powershell"
echo [MyFLOW] zapuskayu servery...
start "MyFLOW servers" %PS% -NoExit -ExecutionPolicy Bypass -File "%~dp0scripts\start-all.ps1"

rem --- 2. zhdem ASR :8081 (do 90s) ---
set /a tries=0
:wait_asr
curl.exe -s --noproxy "*" --max-time 2 -o NUL http://127.0.0.1:8081/health
if errorlevel 1 (
  set /a tries+=1
  if !tries! geq 45 (
    echo [MyFLOW] ASR :8081 ne podnyalsya za 90s - smotri okno "MyFLOW servers".
    goto fail
  )
  timeout /t 2 /nobreak >nul
  goto wait_asr
)
echo [MyFLOW] ASR  :8081 OK

rem --- 3. zhdem LLM :8082 (do 90s, ne kritichno) ---
set /a tries=0
:wait_llm
curl.exe -s --noproxy "*" --max-time 2 -o NUL http://127.0.0.1:8082/v1/models
if errorlevel 1 (
  set /a tries+=1
  if !tries! geq 45 (
    echo [MyFLOW] WARNING: LLM :8082 ne podnyalsya - rabota bez polirovki.
    goto run_main
  )
  timeout /t 2 /nobreak >nul
  goto wait_llm
)
echo [MyFLOW] LLM  :8082 OK

:run_main
rem --- 4. main v otdelnom okne ---
echo [MyFLOW] zapuskayu main...
start "MyFLOW MAIN" /D "%~dp0" cmd /k ""%~dp0.venv\Scripts\python.exe" -m myflow.main"
echo [MyFLOW] vse zapushcheno.
echo [MyFLOW] F8 = PTT (derzhi), F9 = otmena, F7 = stil.
echo.
exit /b 0

:fail
echo [MyFLOW] prover logi v data\ (asr.log.err / llm.log.err).
pause
exit /b 1