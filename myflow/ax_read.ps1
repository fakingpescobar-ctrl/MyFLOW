# MyFLOW AX reader: reads the focused text field via UI Automation.
#
# Runs as a persistent helper process (like Wispr's native "Wispr Flow Helper.exe"):
# Add-Type costs ~1s, so the process starts once and serves requests over stdin/stdout.
#
# Protocol (one line per request):
#   stdin : read
#   stdout: {"ok":true,"source":"value|text|legacy|none|error","pid":N,
#            "window":"...","text_b64":"<UTF-8 text, base64>"}
#   stdin : quit  -> exit
#
# source:
#   value  - ValuePattern.Current.Value (modern fields: Electron, Chrome, UWP, WPF, WinForms)
#   legacy - LegacyIAccessible.Current.Value (old Win32/Tk fields)
#   text   - TextPattern.DocumentRange (documents; for empty fields may return the
#            placeholder, so it is tried last and reported as unreliable)
#   none   - no text available
#   error  - exception while reading
#
# NOTE: this file must stay ASCII-only. Windows PowerShell 5.1 parses .ps1 without a
# BOM as ANSI, so non-ASCII literals/comments would be mangled.

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'SilentlyContinue'

function Get-Pattern {
    param($element, $pattern)
    try {
        $obj = $null
        if ($element.TryGetCurrentPattern($pattern, [ref]$obj)) { return $obj }
    } catch { }
    return $null
}

function Get-FocusedText {
    $result = [ordered]@{ ok = $true; source = 'none'; pid = 0; window = ''; text_b64 = '' }
    try {
        $el = [System.Windows.Automation.AutomationElement]::FocusedElement
        if (-not $el) { return $result }
        $result.pid = [int]$el.Current.ProcessId

        try {
            $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
            $node = $el
            for ($i = 0; $i -lt 12 -and $node; $i++) {
                $parent = $walker.GetParent($node)
                if (-not $parent) { break }
                $node = $parent
                if ($node.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window) { break }
            }
            if ($node) { $result.window = [string]$node.Current.Name }
        } catch { }

        $text = $null
        $source = 'none'
        $reliable = $true

        $vp = Get-Pattern $el ([System.Windows.Automation.ValuePattern]::Pattern)
        if ($vp) {
            $text = [string]$vp.Current.Value
            $source = 'value'
        }

        if ($null -eq $text) {
            # LegacyIAccessiblePattern есть не во всех сборках .NET: тип резолвится
            # в рантайме, поэтому ветку изолируем — иначе падает вся функция.
            try {
                $legacyPattern = [System.Windows.Automation.LegacyIAccessiblePattern]::Pattern
                $lp = Get-Pattern $el $legacyPattern
                if ($lp) {
                    $text = [string]$lp.Current.Value
                    $source = 'legacy'
                }
            } catch { }
        }

        if ($null -eq $text) {
            $tp = Get-Pattern $el ([System.Windows.Automation.TextPattern]::Pattern)
            if ($tp) {
                $text = [string]$tp.DocumentRange.GetText(-1)
                $source = 'text'
                $reliable = $false
            }
        }

        if ($null -eq $text) { $text = '' }
        $result.source = $source
        $result.ok = $reliable
        $result.text_b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($text))
    } catch {
        $result.ok = $false
        $result.source = 'error'
        try { $result | Add-Member -NotePropertyName err -NotePropertyValue ([string]$_.Exception.Message) } catch { }
    }
    return $result
}

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    $cmd = $line.Trim().ToLower()
    if ($cmd -eq 'quit') { break }
    if ($cmd -ne '' -and $cmd -ne 'read') { continue }
    $obj = Get-FocusedText
    [Console]::Out.WriteLine(($obj | ConvertTo-Json -Compress))
    [Console]::Out.Flush()
}
