# start-all.ps1 — подъём серверов MyFLOW (пульт, вызывается из start-myflow.bat).
#   whisper-server (:8081) и llama-server (:8082) — ФОНОВЫЕ, логи в data\*.log.
#   Это окно — пульт: показывает статус и живёт, пока его не закроют.
#   Закрытие пульта НЕ убивает серверы (они отдельные процессы).
#   Если порт занят — сервер не перезапускается (скип).
# Пути: $root вычисляется от расположения скрипта; модели — $env:MYFLOW_MODELS_DIR.
param()
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$models = if ($env:MYFLOW_MODELS_DIR) { $env:MYFLOW_MODELS_DIR } else { "X:\llm\models" }
$llamaDir = if ($env:MYFLOW_LLAMA_DIR) { $env:MYFLOW_LLAMA_DIR } else { "X:\llm\llama-cpp" }
$okColor = "Green"; $infoColor = "Cyan"; $errColor = "Red"

function Test-Port($port) {
    try { return [bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction Stop) }
    catch { return $false }
}

function Wait-Port($port, $path, $seconds = 45) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        curl.exe -s --noproxy "*" -o NUL -w "%{http_code}" "http://127.0.0.1:$port$path" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Start-Server($exe, $arguments, $logName) {
    $out = "$root\data\$logName.log"
    $err = "$root\data\$logName.log.err"
    Start-Process -FilePath $exe -ArgumentList $arguments -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
}

Write-Host "`n=== MyFLOW: серверы (фоновые, логи в data\) ===" -ForegroundColor $infoColor

# --- 1. ASR whisper-server :8081 ---
if (Test-Port 8081) {
    Write-Host "ASR  :8081 — уже работает, скип" -ForegroundColor $okColor
} else {
    $whisper = Join-Path $root "bin\whisper\Release\whisper-server.exe"
    $wModel  = Join-Path $models "ggml-large-v3-turbo-q5_0.bin"
    if (-not (Test-Path $whisper)) { Write-Host "нет whisper-server.exe: $whisper" -ForegroundColor $errColor; exit 1 }
    if (-not (Test-Path $wModel))   { Write-Host "нет модели: $wModel (MYFLOW_MODELS_DIR)" -ForegroundColor $errColor; exit 1 }
    Write-Host "ASR  :8081 — поднимаю whisper-server (фон, лог data\asr.log)..." -ForegroundColor $infoColor
    Start-Server $whisper @('-m', $wModel, '--host', '127.0.0.1', '--port', '8081', '-l', 'auto', '-t', '8') "asr"
    if (Wait-Port 8081 "/health") { Write-Host "ASR  :8081 — OK" -ForegroundColor $okColor }
    else { Write-Host "ASR  :8081 — НЕ поднялся за 45с (лог: data\asr.log.err)" -ForegroundColor $errColor }
}

# --- 2. LLM llama-server :8082 (Qwen3-4B, VRAM-save: c8192 ngl35) ---
if (Test-Port 8082) {
    Write-Host "LLM  :8082 — уже работает, скип" -ForegroundColor $okColor
} else {
    $llama  = Join-Path $llamaDir "llama-server.exe"
    $lModel = Join-Path $models "Qwen3-4B-Instruct-2507-Q8_0.gguf"
    if (-not (Test-Path $llama))  { Write-Host "нет llama-server.exe: $llama" -ForegroundColor $errColor; exit 1 }
    if (-not (Test-Path $lModel)) { Write-Host "нет модели: $lModel" -ForegroundColor $errColor; exit 1 }
    Write-Host "LLM  :8082 — поднимаю llama-server (фон, лог data\llm.log)..." -ForegroundColor $infoColor
    Start-Server $llama @('-m', $lModel, '--host', '127.0.0.1', '--port', '8082', '-c', '8192', '-ngl', '35', '--jinja') "llm"
    if (Wait-Port 8082 "/v1/models") { Write-Host "LLM  :8082 — OK" -ForegroundColor $okColor }
    else { Write-Host "LLM  :8082 — НЕ поднялся за 45с (лог: data\llm.log.err)" -ForegroundColor $errColor }
}

Write-Host "`n=== Серверы готовы ===`nЭто окно можно закрыть — серверы продолжат работать." -ForegroundColor $okColor
Write-Host "Логи: $root\data\asr.log, $root\data\llm.log (ошибки: *.log.err)" -ForegroundColor $infoColor
Write-Host "Пульт закрыть — Ctrl+C. Серверы при этом НЕ останавливаются.`n" -ForegroundColor $infoColor

# Живём, пока не закроют окно (заодно держим консоль открытой для status-обновлений).
while ($true) { Start-Sleep -Seconds 30 }
