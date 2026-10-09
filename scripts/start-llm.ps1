# start-llm.ps1 — локальная LLM для полировки (llama-server) в видимом окне.
# Модель: Qwen3-4B-Instruct-2507 Q8 — быстрый форматтер (~0.5-1.5с/диктовку,
# 16x быстрее gemma-4-12B). Как у Wispr: маленькая nano-модель для polish.
# Контекст 8192: системный промпт (стиль + правила чисел/омофонов + словарь)
# занимает ~1.5-2к токенов, с -c 2048 приходили реальные 400 exceed_context
# (data/main.log), а KV-кэш 4B-модели на 8192 стоит ~30-40 МБ VRAM.
# -ngl 35 — VRAM-компромисс, чтобы рядом влез whisper-server GPU (10 ГБ).
param()
$ErrorActionPreference = "Stop"

$llamaDir = if ($env:MYFLOW_LLAMA_DIR) { $env:MYFLOW_LLAMA_DIR } else { "X:\llm\llama-cpp" }
$models   = if ($env:MYFLOW_MODELS_DIR) { $env:MYFLOW_MODELS_DIR } else { "X:\llm\models" }
$llama = Join-Path $llamaDir "llama-server.exe"
$model = Join-Path $models "Qwen3-4B-Instruct-2507-Q8_0.gguf"

if (-not (Test-Path $llama)) { throw "нет llama-server.exe: $llama (задай MYFLOW_LLAMA_DIR)" }
if (-not (Test-Path $model)) { throw "нет модели: $model (задай MYFLOW_MODELS_DIR)" }

Write-Host "MyFLOW LLM: llama-server (qwen3-4b-q8) :8082 (VRAM-save: c8192 ngl35)" -ForegroundColor Green
& $llama -m $model --host 127.0.0.1 --port 8082 -c 8192 -ngl 35 --jinja
