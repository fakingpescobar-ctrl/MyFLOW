@echo off
rem ============================================================
rem  MyFLOW: okno dlya CHTENIYA GLAVY vsluh - teleprompter.
rem
rem  Otkryvaet tekst krupno, krasit slova po mere chteniya i daet knopki:
rem    seryy   - prochitano
rem    krasnyy - GigaAM uslyshal ne to, zhivoy WER
rem    zheltyy - tekushchee slovo
rem  Knopki: Zanova R | Otchet W | Propustit S | Stop Esc
rem
rem  V MyFLOW vstavka dolzhna byt "aktivnoe okno Ctrl+V",
rem  chitat po predlozheniyu s pauzoy. Zakryt - Esc.
rem
rem  Tekst beretsya iz data\bench\private\ (v git ne popadaet):
rem    esli est klinok-ch1.txt - beretsya on, inache pervyy .txt v papke.
rem  Zapusk:  read-chapter.bat                  - sam naydet tekst
rem           read-chapter.bat put\glava.txt   - svoy fail
rem           read-chapter.bat --check         - proverka bez okna
rem
rem  VNIMANIE: vnutri blokov if nelzya skobki i && - cmd ih ne perevarivaet.
rem ============================================================
title MyFLOW - chtenie glavy
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto no_venv

set "MODE="
set "TEXT="
set "PLAIN="
set "HILITE="
if /i "%~1"=="--check" set "MODE=--check"
if /i "%~1"=="--notepad" set "PLAIN=1"
if /i "%~1"=="--highlight" set "HILITE=--highlight"
if defined MODE goto resolve
if defined PLAIN goto resolve
set "TEXT=%~1"
if defined TEXT goto resolve

:resolve
if defined TEXT if exist "%TEXT%" goto run
rem --- snachala imenno glava, potom lyuboy drugoy tekst (krom kopiy .orig) ---
if exist "%~dp0data\bench\private\klinok-ch1.txt" set "FOUND=%~dp0data\bench\private\klinok-ch1.txt"
for %%F in ("%~dp0data\bench\private\*.txt") do if not defined FOUND if not "%%~xF"==".orig" set "FOUND=%%F"
if defined FOUND set "TEXT=%FOUND%"
if not defined TEXT goto no_text
if not exist "%TEXT%" goto no_text
goto run

:no_venv
echo [chtenie] net .venv - sozday okruzhenie:
echo    py -3.13 -m venv .venv
echo    .venv\Scripts\pip install -r requirements.txt
pause
exit /b 1

:no_text
echo [chtenie] net teksta. Polozhi glavu v papku data\bench\private\ v UTF-8
echo ili ukazhi svoy put:  read-chapter.bat put\glava.txt
echo.
dir /b "%~dp0data\bench\private\*.txt" 2>nul
pause
exit /b 1

:run
echo [chtenie] tekst: %TEXT%
if defined PLAIN goto run_notepad
echo [chtenie] otkryvayu tekst dlya naCHITKI (bez podsvetki).
echo [chtenie] v MyFLOW vstavka "nikuda (tolko istoriya)"; zakryt - Esc.
"%~dp0.venv\Scripts\python.exe" -X utf8 -m myflow.readwindow --text "%TEXT%" --font 24 %MODE% %HILITE%
if errorlevel 1 goto failed
exit /b 0

:run_notepad
echo [chtenie] otkryvayu tekst v Blokmote (bez podsvetki).
start "" notepad.exe "%TEXT%"
exit /b 0

:run_python
"%~dp0.venv\Scripts\python.exe" -X utf8 -m myflow.readwindow --text "%TEXT%" --font 24 %MODE%
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo [chtenie] okno zakrylos s oshibkoy - smotri soobshchenie vyshe.
pause
exit /b 1
