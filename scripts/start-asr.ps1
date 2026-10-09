# start-asr.ps1 — локальный whisper-server (CUDA) в видимом окне.
# Модель кладётся в $env:MYFLOW_MODELS_DIR (по умолчанию X:\llm\models).
param()
$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent $PSScriptRoot
$whisper = Join-Path $root "bin\whisper\Release\whisper-server.exe"
$models  = if ($env:MYFLOW_MODELS_DIR) { $env:MYFLOW_MODELS_DIR } else { "X:\llm\models" }
$model   = Join-Path $models "ggml-large-v3-turbo-q5_0.bin"

if (-not (Test-Path $whisper)) { throw "нет whisper-server.exe: $whisper" }
if (-not (Test-Path $model))   { throw "нет модели: $model (задай MYFLOW_MODELS_DIR)" }

Write-Host "MyFLOW ASR: whisper-server (large-v3-turbo) :8081" -ForegroundColor Cyan
& $whisper -m $model --host 127.0.0.1 --port 8081 -l auto -t 8
