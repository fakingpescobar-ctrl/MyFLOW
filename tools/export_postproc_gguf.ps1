# Собрать GGUF из дообученного постпроцессора MyFLOW.
#
# Нужно, потому что боевой рантайм — llama.cpp: он принимает только GGUF.
# Скрипт:
#   1. скачивает convert_hf_to_gguf.py (один файл из репозитория llama.cpp),
#      если его ещё нет рядом со сборкой;
#   2. конвертирует слитую HF-модель в GGUF (f16);
#   3. квантует нашим llama-quantize.exe (по умолчанию Q8_0 — как у Qwen3-4B);
#   4. кладёт результат в models/.
#
# Запуск:
#   powershell -File tools\export_postproc_gguf.ps1 -Merged data\train\postproc_merged
param(
    [string]$Merged = "data\train\postproc_merged",
    [string]$Quant = "Q8_0",
    [string]$Ref = "master"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$mergedPath = Join-Path $root $Merged
$llamaDir = Join-Path $root "bin\llama-cpp"
$modelsDir = Join-Path $root "models"
$work = Join-Path $root "data\train\gguf"
New-Item -ItemType Directory -Force -Path $work, $modelsDir | Out-Null

if (-not (Test-Path $mergedPath)) {
    throw "нет слитой модели: $mergedPath (сначала tools\merge_postproc_lora.py)"
}
$quantize = Join-Path $llamaDir "llama-quantize.exe"
if (-not (Test-Path $quantize)) { throw "нет квантователя: $quantize" }

$converter = Join-Path $work "convert_hf_to_gguf.py"
if (-not (Test-Path $converter)) {
    Write-Host "скачиваю convert_hf_to_gguf.py ($Ref)..."
    $url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/$Ref/convert_hf_to_gguf.py"
    Invoke-WebRequest -Uri $url -OutFile $converter -UseBasicParsing
}

# В учебном окружении есть transformers/gguf; конвертация идёт на CPU (быстро для 0.6B)
$python = Join-Path $root ".venv-train\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = Join-Path $root ".venv\Scripts\python.exe"
    Write-Host "учебного окружения нет — использую боевое (нужны transformers и gguf)"
}

$f16 = Join-Path $work "postproc-f16.gguf"
Write-Host "конвертирую в GGUF (f16)..."
& $python -X utf8 $converter $mergedPath --outfile $f16 --outtype f16
if ($LASTEXITCODE -ne 0) { throw "конвертация не удалась" }

$name = "myflow-postproc-qwen3-06b-$($Quant.ToLower()).gguf"
$target = Join-Path $modelsDir $name
Write-Host "квантую в $Quant..."
& $quantize $f16 $target $Quant
if ($LASTEXITCODE -ne 0) { throw "квантование не удалось" }

$size = [math]::Round((Get-Item $target).Length / 1MB, 1)
Write-Host "готово: models\$name ($size МБ)"
Write-Host ""
Write-Host "Проверить модель отдельно (порт 8083):"
Write-Host "  bin\llama-cpp\llama-server.exe -m `"$target`" --host 127.0.0.1 --port 8083 -c 8192 -np 1 -ngl 99 -fa on --jinja"
Write-Host "Затем замер:"
Write-Host "  .venv\Scripts\python.exe -X utf8 tools\eval_postproc.py --llm --url http://127.0.0.1:8083/v1"
