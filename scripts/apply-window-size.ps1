# Применить сохранённый размер окна MyFLOW ДО запуска TUI.
#
# Зачем: классическая консоль (ConsoleHost) не запоминает размер между запусками
# и всегда открывается 120x30 — отсюда «окно опять мелкое». Скрипт читает размер
# из data/settings.json (секция tui) и ставит его окну.
#
# Запускается из start-myflow.bat. Ключ -Check только печатает, что сделал бы.
param([switch]$Check)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$settings = Join-Path $root "data\settings.json"

$cols = 0
$rows = 0
if (Test-Path $settings) {
    try {
        $json = Get-Content $settings -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($json.tui) {
            $cols = [int]$json.tui.cols
            $rows = [int]$json.tui.rows
        }
    } catch {
        $cols = 0
        $rows = 0
    }
}

if ($cols -lt 60 -or $rows -lt 20) {
    if ($Check) { Write-Host "[размер] в настройках нет размера (tui.cols/rows) — пропускаю" }
    exit 0
}

if ($Check) {
    Write-Host "[размер] поставил бы ${cols}x${rows}"
    exit 0
}

try {
    # Буфер делаем ровно под окно: тогда консоль не может «уехать» прокруткой.
    [console]::SetBufferSize($cols, $rows)
    [console]::SetWindowSize($cols, $rows)
} catch {
    # Если .NET-путь не сработал (или буфер уже меньше) — пробуем mode con.
    try { mode con: cols=$cols lines=$rows | Out-Null } catch { }
}
exit 0
