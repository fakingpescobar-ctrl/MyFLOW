# stop-servers.ps1 — погасить серверы MyFLOW и освободить память/VRAM.
#
# Трогает ТОЛЬКО процессы из папки этого проекта (bin\llama-cpp, bin\whisper):
# чужие сборки llama.cpp (например, из другого каталога) не задевает.
#
# Использование:
#   powershell -ExecutionPolicy Bypass -File scripts\stop-servers.ps1
#   powershell ... -File scripts\stop-servers.ps1 -Quiet
#
# Учти: если окно MyFLOW открыто, оно поднимет LLM заново при следующей фразе
# (это автолечение). Чтобы VRAM осталась свободной — сначала закрой MyFLOW
# либо выключи LLM в панели (Enter на строке «LLM»).
param([switch]$Quiet)

$ErrorActionPreference = "SilentlyContinue"
# Русский текст в консоли: иначе в cp866/cp1251 получается мусор.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$root = Split-Path -Parent $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

function Stop-ProjectServers {
    param([string]$ProjectRoot)
    $killed = @()
    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe' or Name='whisper-server.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($ProjectRoot, 'OrdinalIgnoreCase') } |
        ForEach-Object {
            $killed += "$($_.Name) (PID $($_.ProcessId))"
            Stop-Process -Id $_.ProcessId -Force
        }
    return $killed
}

$stopped = Stop-ProjectServers -ProjectRoot $root
if (-not $Quiet) {
    if ($stopped.Count -gt 0) {
        Write-Host "MyFLOW: остановлены серверы проекта:" -ForegroundColor Green
        $stopped | ForEach-Object { Write-Host "  - $_" }
    } else {
        Write-Host "MyFLOW: серверов проекта в памяти нет." -ForegroundColor DarkGray
    }
    # Что осталось чужого — чтобы было видно, что мы ничего не тронули.
    $foreign = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe' or Name='whisper-server.exe'" |
        Where-Object { $_.ExecutablePath -and -not $_.ExecutablePath.StartsWith($root, 'OrdinalIgnoreCase') }
    if ($foreign) {
        Write-Host "Чужие экземпляры (не проекта, не тронуты):" -ForegroundColor DarkYellow
        $foreign | ForEach-Object { Write-Host "  - $($_.ExecutablePath) (PID $($_.ProcessId))" }
    }
}
