@echo off
rem ============================================================
rem  MyFLOW: ustanovka dlya novogo polzovatelya.
rem  1) proveryaet Python  2) sozdaet .venv  3) stavit zavisimosti
rem  4) zapuskaet doktora - on skazhet, chego eshe ne hvataet.
rem
rem  Modeli (GigaAM, whisper, postprocessor) vesyat gigaBAJTY i
rem  skachivayutsya otdelno, smotri README. Doktor perechislit,
rem  chto imenno nuzhno.
rem
rem  VNIMANIE: vnutri blokov if nelzya skobki i && - cmd ih ne perevarivaet.
rem ============================================================
title MyFLOW - ustanovka
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo [1/4] proveryayu Python...
where py >nul 2>nul
if errorlevel 1 goto no_python

if exist "%~dp0.venv\Scripts\python.exe" goto have_venv
echo [2/4] sozdayu okruzhenie .venv - eto zaymet minutu
py -3.13 -m venv "%~dp0.venv"
if not exist "%~dp0.venv\Scripts\python.exe" goto venv_failed
goto deps

:have_venv
echo [2/4] okruzhenie .venv uzhe est

:deps
echo [3/4] ustanavlivayu zavisimosti
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto deps_failed

echo [4/4] proveryayu, chto gotovo
echo.
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0tools\doctor.py"
echo.
echo Esli vyshe vse "ok" - zapuskay start-myflow.bat
pause
exit /b 0

:no_python
echo [oshibka] ne nayden Python. Ustanovi Python 3.13 s python.org
echo i zapusti etot fail snova.
pause
exit /b 1

:venv_failed
echo [oshibka] ne udalos sozdat .venv
pause
exit /b 1

:deps_failed
echo [oshibka] ne ustanovilis zavisimosti - smotri soobshchenie vyshe
pause
exit /b 1
