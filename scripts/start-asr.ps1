# start-asr.ps1 — локальный whisper-server (CUDA) в видимом окне.
# Модель: ggml-large-v3-turbo-q5_0.bin (уже есть в X:\llm\models).
$whisper = "bin\whisper\Release\whisper-server.exe"
$model   = "X:\llm\models\ggml-large-v3-turbo-q5_0.bin"
if (-not (Test-Path $whisper)) { throw "нет whisper-server.exe: $whisper" }
if (-not (Test-Path $model))   { throw "нет модели: $model" }

Write-Host "MyFLOW ASR: whisper-server (large-v3-turbo) :8081" -ForegroundColor Cyan
& $whisper -m $model --host 127.0.0.1 --port 8081 -l auto -t 8