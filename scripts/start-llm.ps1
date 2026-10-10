# start-llm.ps1 — локальная LLM для полировки (llama-server) в видимом окне.
# Модель: Qwen3-4B-Instruct-2507 Q8 — быстрый форматтер (~0.2с/диктовку).
#
# Тюнинг 10.10.2026 (замеры tools/bench_llm.py на RTX 3080):
#   -np 1 -kvu      один слот (было 4): контекст целиком нашему запросу,
#                   кэш системного промпта всегда попадает (cache_n ~2200);
#   -ngl 99         ВСЕ 36 слоёв на GPU: при -ngl 35 последний слой вместе с
#                   выходной проекцией (389 МБ Q8) считался на CPU и ронял
#                   генерацию до 80 tok/s -> стало 105-115 tok/s
#                   (полировка p50 205 мс вместо 260-330);
#   -ctk/-ctv q8_0  KV-кэш вдвое компактнее (~1.2 ГБ вместо 2.4) — освобождает
#                   VRAM под -ngl 99; итого сервер держит 6.2 ГБ;
#   -fa on          flash attention явно.
# VRAM 10 ГБ: llama 6.2 ГБ + whisper ~1 ГБ — влезает с запасом.
param()
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$models = if ($env:MYFLOW_MODELS_DIR) { $env:MYFLOW_MODELS_DIR }
          elseif (Test-Path (Join-Path $repoRoot "models")) { Join-Path $repoRoot "models" }
          else { Join-Path $repoRoot "models" }
$llamaDir = if ($env:MYFLOW_LLAMA_DIR) { $env:MYFLOW_LLAMA_DIR }
            elseif (Test-Path (Join-Path $repoRoot "bin\llama-cpp")) { Join-Path $repoRoot "bin\llama-cpp" }
            else { Join-Path $repoRoot "bin\llama-cpp" }
$llama = Join-Path $llamaDir "llama-server.exe"
$model = Join-Path $models "Qwen3-4B-Instruct-2507-Q8_0.gguf"

if (-not (Test-Path $llama)) { throw "нет llama-server.exe: $llama (задай MYFLOW_LLAMA_DIR)" }
if (-not (Test-Path $model)) { throw "нет модели: $model (задай MYFLOW_MODELS_DIR)" }

Write-Host "MyFLOW LLM: llama-server (qwen3-4b-q8) :8082  [np1 kvu ngl99 fa q8-KV]" -ForegroundColor Green
& $llama -m $model --host 127.0.0.1 --port 8082 -c 8192 -np 1 -kvu -ngl 99 -fa on `
    -ctk q8_0 -ctv q8_0 --jinja
