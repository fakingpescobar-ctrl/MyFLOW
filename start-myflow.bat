@echo off
setlocal EnableDelayedExpansion
rem ============================================================
rem  MyFLOW - правильный запуск одной командой (двойной клик).
rem  1) серверы (whisper :8081 + llama :8082) - отдельное окно
rem  2) main (хоткеи F7/F8/F9)                  - отдельное окно
rem  Если main уже работает - второй НЕ запускается.
rem ============================================================
cd /d C:\Projects\MyFLOW

rem --- 0. защита от дублей main (иначе два слушателя F8) ---
powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*myflow*' }; if ($p) { 'RUNNING' ; exit 2 } else { exit 0 }" >nul 2>&1
if errorlevel 2 (
  echo [MyFLOW] main уже запущен. Закрой его окно и запусти снова.
  pause
  exit /b 0
)

rem --- 1. серверы (окно откроется само, здорово) ---
echo [MyFLOW] поднимаю серверы...
start "MyFLOW servers" pwsh -NoExit -ExecutionPolicy Bypass -File "scripts\start-all.ps1" -NoMain

rem --- 2. ждём ASR :8081 (до 90с) ---
set /a tries=0
:wait_asr
curl.exe -s --noproxy "*" --max-time 2 -o NUL http://127.0.0.1:8081/health
if errorlevel 1 (
  set /a tries+=1
  if !tries! geq 45 (
    echo [MyFLOW] ASR :8081 не поднялся за 90с - смотри окно "MyFLOW servers".
    goto fail
  )
  timeout /t 2 /nobreak >nul
  goto wait_asr
)
echo [MyFLOW] ASR  :8081 OK

rem --- 3. ждём LLM :8082 (до 90с, не критично) ---
set /a tries=0
:wait_llm
curl.exe -s --noproxy "*" --max-time 2 -o NUL http://127.0.0.1:8082/v1/models
if errorlevel 1 (
  set /a tries+=1
  if !tries! geq 45 (
    echo [MyFLOW] WARNING: LLM :8082 не поднялся - полировка не будет работать
    goto run_main
  )
  timeout /t 2 /nobreak >nul
  goto wait_llm
)
echo [MyFLOW] LLM  :8082 OK

:run_main
rem --- 4. main в отдельном окне ---
echo [MyFLOW] запускаю main...
start "MyFLOW MAIN" /D "C:\Projects\MyFLOW" cmd /k "".venv\Scripts\python.exe" -m myflow.main"
echo [MyFLOW] ВСЁ ПОДНЯТО.
echo [MyFLOW] В окне "MyFLOW MAIN" сейчас будет ВЫБОР МИКРОФОНА (только живые).
echo [MyFLOW] F8 = диктовка, F9 = отмена, F7 = стиль.
echo.
pause
exit /b 0

:fail
pause
exit /b 1