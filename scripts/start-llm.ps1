# start-llm.ps1 — локальная LLM для полировки (llama-server) в видимом окне.
# Модель: Qwen3-4B-Instruct-2507 Q8 — быстрый форматтер (~0.5-1.5с/диктовку,
# 16x быстрее gemma-4-12B). Как у Wispr: маленькая nano-модель для polish.
# VRAM-компромисс: -c 4096 + -ngl 35 — чтобы рядом влез whisper-server GPU
# (10GB карта; полный ngl+4096 даёт CUDA OOM для whisper).
$llama  = "X:\llm\llama-cpp\llama-server.exe"
$model  = "X:\llm\models\Qwen3-4B-Instruct-2507-Q8_0.gguf"
if (-not (Test-Path $llama)) { throw "нет llama-server.exe: $llama" }
if (-not (Test-Path $model)) { throw "нет модели: $model" }

Write-Host "MyFLOW LLM: llama-server (qwen3-4b-q8) :8082 (VRAM-save: c4096 ngl35)" -ForegroundColor Green
& $llama -m $model --host 127.0.0.1 --port 8082 -c 4096 -ngl 35 --jinja