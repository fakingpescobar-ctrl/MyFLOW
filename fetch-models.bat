@echo off
rem ============================================================
rem  MyFLOW: skachat modeli (whisper ~547 MB).
rem  GigaAM (~214 MB) skachaetsya sam pri pervom raspoznavanii.
rem  LLM-mudel' i postprocessor v git ne lezhat - sm. README.md.
rem
rem  Kachka s dokachkoy: esli obryvaetsya - zapusti snova,
rem  prodolzhit s togo zhe mesta.
rem
rem  VNIMANIE: vnutri blokov if nelzya skobki i && - cmd ih ne perevarivaet.
rem ============================================================
title MyFLOW - modeli
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" goto no_venv

echo [modeli] proveryayu i kachayu chego net...
echo.
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0tools\fetch_models.py"
echo.
echo Posle etogo: python tools\doctor.py - on skazhet, vsyo li gotovo.
pause
exit /b 0

:no_venv
echo [oshibka] net okruzheniya .venv - snachala zapusti install-myflow.bat
pause
exit /b 1
