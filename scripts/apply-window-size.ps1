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

# В Windows Terminal окно принадлежит терминалу: менять размер бесполезно и
# вредно (conpty станет больше окна, и пульт нарисует лишние строки -> «куча»).
Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class ConHost {
    [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern int GetClassNameW(IntPtr h, StringBuilder s, int n);
    public static string ClassName() {
        var h = GetConsoleWindow();
        if (h == IntPtr.Zero) return "";
        var sb = new StringBuilder(256);
        GetClassNameW(h, sb, sb.Capacity);
        return sb.ToString();
    }
}
"@
$pseudo = ([ConHost]::ClassName() -eq "PseudoConsoleWindow")
if ($env:WT_SESSION -or $pseudo) {
    if ($Check) { Write-Host "[размер] терминал управляет окном (WT/conpty [$([ConHost]::ClassName())]) — пропускаю" }
    exit 0
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
    # Буфер ВЫШЕ окна (запас на логи): если сделать его равным окну, любая
    # посторонняя строка вывода прокручивает экран и кадр TUI «съезжает».
    $bufferRows = [Math]::Max($rows + 300, $rows)
    [console]::SetBufferSize($cols, $bufferRows)
    [console]::SetWindowSize($cols, $rows)
} catch {
    # Если .NET-путь не сработал (или буфер уже меньше) — пробуем mode con.
    try { mode con: cols=$cols lines=$rows | Out-Null } catch { }
}
exit 0
