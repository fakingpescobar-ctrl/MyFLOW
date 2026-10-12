@echo off
rem ============================================================
rem  MyFLOW: otkryt interfeys OTDELNYM OKNOM (ne vkladka brauzera).
rem
rem  Ispolzuem rezhim prilozheniya u Chromium (Edge/Chrome):
rem  --app=... daet okno bez vkladok i adresnoy stroki, so svoey
rem  ikonkoy v panele zadach - po vidu i povedeniyu otdelnoe prilozhenie.
rem
rem  Potom etu zhe stranicu mozhno zavernut v pywebview (natvnoe okno
rem  na sistemnom WebView2) ili v Electron - interfeys perenesetsya
rem  bez perepisyvaniya.
rem
rem  VNIMANIE: vnutri blokov if nelzya skobki i && - cmd ih ne perevarivaet.
rem ============================================================
title MyFLOW - interfeys
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=8790"
set "URL=http://127.0.0.1:%PORT%/"

if not exist "%~dp0.venv\Scripts\python.exe" goto no_venv

rem --- server uzhe rabotaet? togda tolko otkryvaem okno ---
netstat -ano | findstr /C:":%PORT% " | findstr /C:"LISTENING" >nul 2>nul
if not errorlevel 1 goto open_window

echo [ui] zapuskayu server interfeysa na portu %PORT%...
start "MyFLOW UI" /min "%~dp0.venv\Scripts\python.exe" -X utf8 -m myflow.webui --port %PORT% --no-open
rem --- daem serveru podnyatsya ---
powershell -NoProfile -Command "Start-Sleep -Milliseconds 1200" >nul 2>nul

:open_window
echo [ui] otkryvayu otdelnoe okno...
rem --- snachala probuem po imeni: start naydet brauzer cherez reestr ---
start "" chrome --app="%URL%" --window-size=1040,760
if not errorlevel 1 exit /b 0
start "" msedge --app="%URL%" --window-size=1040,760
if not errorlevel 1 exit /b 0
rem --- potom po obychnym mestam ustanovki ---
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" goto chrome_files
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" goto chrome_files_x86
if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" goto edge_files
goto fallback_browser

:chrome_files
start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --app="%URL%" --window-size=1040,760
exit /b 0

:chrome_files_x86
start "" "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" --app="%URL%" --window-size=1040,760
exit /b 0

:edge_files
start "" "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" --app="%URL%" --window-size=1040,760
exit /b 0

:fallback_browser
echo [ui] Edge/Chrome ne nayden - otkryvayu obychnym brauzerom.
start "" "%URL%"
exit /b 0

:no_venv
echo [oshibka] net okruzheniya .venv - snachala zapusti install-myflow.bat
pause
exit /b 1
