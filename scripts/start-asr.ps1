# start-asr.ps1 — локальный whisper-server (CUDA) в видимом окне.
# Модель кладётся в $env:MYFLOW_MODELS_DIR (по умолчанию X:\llm\models).
#
# Тюнинг 10.10.2026 (замеры на RTX 3080):
#   -l ru   без авто-детекта языка: он стоит лишний проход и иногда промахивается
#           (в логе был случай, когда русская фраза распозналась как французская);
#   -bs 5   beam search: качество на именах/числах выше (пример: «меня зовут
#           Максим» вместо «меня тут Максим»), запас по скорости 10-кратный —
#           RTF остаётся 0.01-0.02 (8 с аудио ≈ 200-450 мс);
#   -sns    suppress non-speech tokens — меньше галлюцинаций на шуме/музыке;
#   -t 4    потоков CPU хватает (flash attention включён по умолчанию).
param()
$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent $PSScriptRoot
$whisper = Join-Path $root "bin\whisper\Release\whisper-server.exe"
$models  = if ($env:MYFLOW_MODELS_DIR) { $env:MYFLOW_MODELS_DIR } else { "X:\llm\models" }
$model   = Join-Path $models "ggml-large-v3-turbo-q5_0.bin"

if (-not (Test-Path $whisper)) { throw "нет whisper-server.exe: $whisper" }
if (-not (Test-Path $model))   { throw "нет модели: $model (задай MYFLOW_MODELS_DIR)" }

Write-Host "MyFLOW ASR: whisper-server (large-v3-turbo) :8081  [ru beam5 sns]" -ForegroundColor Cyan
& $whisper -m $model --host 127.0.0.1 --port 8081 -l ru -bs 5 -sns -t 4
