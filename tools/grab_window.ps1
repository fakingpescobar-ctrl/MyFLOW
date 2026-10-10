# Сфотографировать окно по части заголовка (например, окно MyFLOW).
#
# Нужно для диагностики отрисовки: картинка показывает ровно то, что видит
# пользователь — переносы строк, наложения, обрезанные края.
param(
    [string]$Match = "MyFLOW",
    [string]$Out = ""
)

if (-not $Out) {
    # Путь по умолчанию — внутри проекта (никаких абсолютных путей в коде).
    $root = Split-Path -Parent $PSScriptRoot
    $Out = Join-Path $root "data\_window.png"
}

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public class WinGrab {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
    public delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);

    public static List<string> Titles() {
        var list = new List<string>();
        EnumWindows((h, p) => {
            if (!IsWindowVisible(h)) return true;
            var sb = new StringBuilder(512);
            GetWindowTextW(h, sb, sb.Capacity);
            if (sb.Length > 0) list.Add(h.ToInt64() + "|" + sb.ToString());
            return true;
        }, IntPtr.Zero);
        return list;
    }
}
"@

$target = $null
$title = ""
foreach ($item in [WinGrab]::Titles()) {
    $parts = $item.Split("|", 2)
    if ($parts[1] -like "*$Match*") {
        $target = [IntPtr][int64]$parts[0]
        $title = $parts[1]
        break
    }
}
if ($null -eq $target) {
    Write-Host "окно с '$Match' в заголовке не найдено"
    exit 1
}

[void][WinGrab]::ShowWindow($target, 9)
[void][WinGrab]::SetForegroundWindow($target)
Start-Sleep -Milliseconds 800

$rect = New-Object WinGrab+RECT
[void][WinGrab]::GetWindowRect($target, [ref]$rect)
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top
Write-Host "окно '$title': позиция $($rect.Left),$($rect.Top), размер ${w}x${h}"

$bmp = New-Object System.Drawing.Bitmap($w, $h)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$gfx.Dispose()
$bmp.Dispose()
Write-Host "сохранил: $Out"
