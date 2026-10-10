# start-llm-9b.ps1 — «режим качества»: Qwen3.5-9B вместо 4B.
#
# A/B на своей истории (tools/eval_polish.py, 40 фраз, RTX 3080):
#   4B Q8   : p50 174 мс · 110 tok/s · исправляет морфологию ХУЖЕ («батника»),
#             но чище убирает слова-паразиты; откатов guard 15 %
#   9B Q4   : p50 219 мс ·  92 tok/s · «перезапусти батник» верно, пунктуация
#             лучше («Брат, …»), откатов guard 5 %, но филлеры оставляет
#
# VRAM: 9B ≈ 7.9 ГБ вместе с whisper (свободно ~2 ГБ) — влезает, но запаса мало,
# поэтому по умолчанию используется 4B (scripts/start-llm.ps1).
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
$model = Join-Path $models "Qwen3.5-9B-UD-Q4_K_XL.gguf"

if (-not (Test-Path $llama)) { throw "нет llama-server.exe: $llama" }
if (-not (Test-Path $model)) { throw "нет модели: $model (положи Qwen3.5-9B-UD-Q4_K_XL.gguf)" }

Write-Host "MyFLOW LLM: llama-server (qwen3.5-9b-q4) :8082  [np1 kvu ngl99 fa q8-KV]" -ForegroundColor Magenta
& $llama -m $model --host 127.0.0.1 --port 8082 -c 8192 -np 1 -kvu -ngl 99 -fa on `
    -ctk q8_0 -ctv q8_0 --jinja
