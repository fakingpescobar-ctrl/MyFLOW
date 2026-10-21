@echo off
rem ============================================================
rem  MyFLOW: okruzhenie dlya doobucheniya postprocessora.
rem
rem  Zachem: svoy postprocessor (LoRA na Qwen3-0.6B) uchitsya na
rem  TVOIH pravkah - i poluchaetsya luchshe obshchey Qwen3-4B:
rem  WER 0.139 protiv 0.152, 64 ms protiv 72 ms, i ~0.6 GB VRAM
rem  vmesto 5-6 GB. Etot put - samyy dohodchiviy v proekte.
rem
rem  Eto OTDELNOE okruzhenie .venv-train: tam CUDA-torch (~2.5 GB),
rem  v boevoe .venv ego tяnut ne nado.
rem
rem  1) sozdaet .venv-train (Python 3.13)
rem  2) stavit zavisimosti iz requirements-train.txt
rem  3) proveryaet, chto torch/transformers/peft na meste
rem
rem  VAZHNO: torch beretsya s indeksa PyTorch (tam CUDA-sborka 2.5 GB);
rem  s PyPI idet CPU-koleso za 194 MB - i obucheniye upoyet v CPU.
rem
rem  Esli pip zaviset (byvaet v etoy seti) - skript podtyanivet
rem  kolesa cherez curl s dokachkoy i postavit oflayn.
rem
rem  VNIMANIE: vnutri blokov if nelzya skobki i && - cmd ih ne perevarivaet.
rem ============================================================
title MyFLOW - okruzhenie obucheniya
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "TRAIN_PY=%~dp0.venv-train\Scripts\python.exe"
set "RUNTIME_PY=%~dp0.venv\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "WHEELS=%~dp0data\train\wheels"

if exist "%TRAIN_PY%" goto have_venv

echo [1/3] sozdayu okruzhenie .venv-train...
where py >nul 2>nul
if errorlevel 1 goto no_python
py -3.13 -m venv "%~dp0.venv-train"
if not exist "%TRAIN_PY%" goto venv_failed
goto deps

:have_venv
echo [1/3] okruzhenie .venv-train uzhe est

:deps
echo [2/3] stavlyu zavisimosti (torch s CUDA - ~2.5 GB, dolgo)...
echo       (esli pip zamriyot - sam peredydy na oflayn-kolesa)
"%TRAIN_PY%" -m pip install --upgrade pip
powershell -NoProfile -Command "$p = Start-Process -FilePath '%TRAIN_PY%' -ArgumentList '-m','pip','install','--extra-index-url','%TORCH_INDEX%','-r','%~dp0requirements-train.txt' -PassThru -NoNewWindow; if (-not $p.WaitForExit(1200000)) { $p.Kill(); Write-Host 'pip ZAVES - idem po oflayn-puti'; exit 1 } else { exit $p.ExitCode }"
if errorlevel 1 goto pip_failed
goto check

rem ------------------------------------------------------------ pip ne spravilsya
:pip_failed
echo.
echo [2/3] oflayn-put: kolesa cherez curl s dokachkoy...
if not exist "%RUNTIME_PY%" goto no_runtime
if not exist "%WHEELS%" mkdir "%WHEELS%"

rem torch s CUDA berem s indeksa PyTorch - s PyPI lezhit CPU-koleso.
if not exist "%WHEELS%\torch-2.6.0+cu124-cp313-cp313-win_amd64.whl" (
  echo   kachayu torch 2.6.0+cu124 - 2.5 GB, dokat s momenta...
  curl.exe -L -C - --retry 40 --retry-delay 5 --speed-limit 50000 --speed-time 60 ^
    -o "%WHEELS%\torch-2.6.0+cu124-cp313-cp313-win_amd64.whl" ^
    "%TORCH_INDEX%/torch-2.6.0%%2Bcu124-cp313-cp313-win_amd64.whl"
  if errorlevel 1 goto wheels_failed
)

echo   kachayu ostalnye kolesa...
"%RUNTIME_PY%" -X utf8 "%~dp0tools\fetch_wheels.py" ^
  accelerate annotated-doc anyio certifi charset-normalizer click colorama ^
  datasets dill filelock fsspec gguf h11 hf-xet httpcore2 httpx2 huggingface_hub ^
  idna Jinja2 markdown-it-py MarkupSafe mdurl mpmath multiprocess networkx ^
  numpy packaging pandas peft py-cpuinfo pyarrow Pygments python-dateutil PyYAML ^
  regex requests rich safetensors sentencepiece shellingham six ^
  "sympy==1.13.1" tokenizers torch tqdm transformers truststore typer ^
  typing_extensions tzdata urllib3 xxhash
if errorlevel 1 goto wheels_failed

echo   stavlyu oflayn...
"%TRAIN_PY%" -m pip install --no-index --find-links "%WHEELS%" --no-cache-dir ^
  --progress-bar off torch transformers peft accelerate datasets gguf pyarrow sympy numpy
if errorlevel 1 goto wheels_failed

:check
echo.
echo [3/3] proverka...
"%TRAIN_PY%" -c "import torch, transformers, peft; print('torch', torch.__version__, '| CUDA dostupen:', torch.cuda.is_available())"
if errorlevel 1 goto check_failed

echo.
echo GOTOVO. Teper mozhno obuchit svoy postprocessor:
echo   .venv\Scripts\python -X utf8 tools\retrain_postproc.py --dry-run
echo.
echo Dalshe v pulte: stroka "Doobuchenie" (Enter) zapustit polnyi cikl.
pause
exit /b 0

:wheels_failed
echo.
echo [oshibka] ne udalos postavit kolesa oflayn.
echo           Posmotri data\train\wheels - tam skachannye kolesa.
pause
exit /b 1

:check_failed
echo.
echo [oshibka] torch/transformers/peft ne importiruyutsya.
echo           Esli pishet pro CUDA - znachit postavilos CPU-koleso.
pause
exit /b 1

:venv_failed
echo.
echo [oshibka] ne udalos sozdat .venv-train.
pause
exit /b 1

:no_python
echo.
echo [oshibka] ne nayden Python. Ustanovi Python 3.13 s python.org
pause
exit /b 1

:no_runtime
echo.
echo [oshibka] net boevogo okruzheniya .venv - snachala zapusti install-myflow.bat
pause
exit /b 1
