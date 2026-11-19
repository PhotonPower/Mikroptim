<#
.SYNOPSIS
    cmake --build mit Schutz gegen Quellen, die sich während des Builds ändern (Windows).

.DESCRIPTION
    MSBuild (Visual-Studio-Generator) entscheidet über Neuübersetzung nur nach Zeitstempeln:
    Eine Objektdatei gilt als aktuell, wenn sie jünger ist als Quelle und Header. Ändert sich ein
    Header, während eine Datei, die ihn einbindet, gerade übersetzt wird (z. B. git rebase oder
    git checkout während eines laufenden Builds), ist die Objektdatei danach jünger als der Header,
    enthält aber den alten Stand. Folgebuilds übersetzen sie nie wieder; Strukturen haben dann in
    verschiedenen Objektdateien verschiedene Layouts (Testfehler, "Stack around the variable was
    corrupted"). Bei .cu-Dateien ist das Fenster groß, weil nvcc je Datei mehrere Architekturen
    übersetzt und MSBuild die .cu-Dateien eines Ziels nacheinander abarbeitet.

    Dieses Skript merkt sich den Startzeitpunkt, ruft cmake --build auf und sucht danach im Repo
    (git ls-files, auch unversionierte nicht ignorierte Dateien) nach Dateien, die nach dem Start
    geändert wurden. Solche Dateien bekommen den aktuellen Zeitstempel ("touch"), damit alle
    abhängigen Objektdateien wieder als veraltet gelten, und der Build wird einmal wiederholt
    (abschaltbar mit -KeineWiederholung; dann Exit-Code 3). docs/hardware.md, Abschnitt
    "Bauen unter Windows".

.PARAMETER Preset
    Build-Preset (Standard: release-cuda). Ignoriert, wenn -BuildDir gesetzt ist.

.PARAMETER BuildDir
    Build-Verzeichnis statt eines Presets (z. B. build/gcc).

.PARAMETER Config
    Konfiguration für Multi-Config-Generatoren zusammen mit -BuildDir (Standard: Release).

.PARAMETER KeineWiederholung
    Nach einer Änderung während des Builds nicht neu bauen, sondern mit Exit-Code 3 enden.

.PARAMETER BuildArgs
    Weitere Argumente für cmake --build (z. B. --parallel 8 --target mikroptim_tests).

.EXAMPLE
    tools/bauen.ps1 -Preset release-cuda --parallel 8
.EXAMPLE
    tools/bauen.ps1 -BuildDir build/gcc
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Preset = 'release-cuda',
    [string]$BuildDir = '',
    [string]$Config = 'Release',
    [switch]$KeineWiederholung,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BuildArgs = @()
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Invoke-CMakeBuild {
    # Out-Host: die Ausgabe von cmake nicht als Rückgabewert der Funktion einsammeln
    if ($BuildDir) {
        & cmake --build $BuildDir --config $Config @BuildArgs | Out-Host
    } else {
        & cmake --build --preset $Preset @BuildArgs | Out-Host
    }
    return $LASTEXITCODE
}

# Alle Dateien des Repos außer Ignoriertem (Build-Verzeichnisse, data/glass/private, ...)
function Get-RepoFiles {
    $names = & git -C $repo ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) { throw 'git ls-files fehlgeschlagen' }
    foreach ($n in $names) {
        $p = Join-Path $repo $n
        if (Test-Path -LiteralPath $p -PathType Leaf) { Get-Item -LiteralPath $p }
    }
}

for ($lauf = 1; $lauf -le 2; $lauf++) {
    # Vor dem Start geänderte Dateien sieht der Build; die eigenen Zeitstempel aus dem ersten Lauf
    # liegen vor dem Start des zweiten
    $start = Get-Date
    $code = Invoke-CMakeBuild
    $geaendert = @(Get-RepoFiles | Where-Object { $_.LastWriteTime -gt $start })
    if ($geaendert.Count -eq 0) {
        exit $code
    }
    Write-Warning (("bauen.ps1: {0} Datei(en) wurden während des Builds geändert; Objektdateien, " +
        "die sie einbinden, können veraltet sein:") -f $geaendert.Count)
    $jetzt = Get-Date
    foreach ($f in $geaendert) {
        Write-Warning ('  ' + $f.FullName.Substring($repo.Length + 1))
        # kurz gesperrte Dateien (Compiler, Virenscanner) erneut versuchen
        for ($i = 0; ; $i++) {
            try { $f.LastWriteTime = $jetzt; break }
            catch { if ($i -ge 50) { throw }; Start-Sleep -Milliseconds 100 }
        }
    }
    if ($KeineWiederholung -or $lauf -eq 2) {
        Write-Warning 'bauen.ps1: Zeitstempel erneuert; bitte erneut bauen (nicht während eines Builds rebasen).'
        exit 3
    }
    Write-Warning 'bauen.ps1: Zeitstempel erneuert, baue erneut.'
}
