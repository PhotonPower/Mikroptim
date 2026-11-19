<#
.SYNOPSIS
    Prüft, ob ein inkrementeller Build nach Header-Änderungen alle abhängigen Objekte neu übersetzt.

.DESCRIPTION
    Zwei Prüfungen auf einem vorhandenen Build-Verzeichnis (Visual Studio/MSBuild oder Ninja):

    A "abhaengigkeiten": Hängt einen Kommentar an einen Header an, baut, und prüft, dass jede
       Übersetzungseinheit (.cpp/.cu), die den Header direkt oder indirekt einbindet, neu
       übersetzt wurde (Zeitstempel der Objektdatei). Das Orakel ist ein eigener Scan der
       #include "..."-Direktiven (Obermenge: bedingte Includes zählen mit), unabhängig vom
       Buildsystem.
    B "waehrend": Ändert den Header, während der Compiler eine einbindende Datei übersetzt (wie ein
       git rebase während eines laufenden Builds), und zwar so, dass er nicht mehr übersetzbar ist
       (#error). Der Folgebuild muss diesen Fehler melden; baut er erfolgreich, ist eine veraltete
       Objektdatei liegen geblieben.

    Mit -MitSchutz laufen alle Builds über tools/bauen.ps1 statt direkt über cmake --build.
    Der Header wird am Ende mit Inhalt und Zeitstempel wiederhergestellt; ein abschließender
    Build bringt das Build-Verzeichnis wieder in einen konsistenten Stand.
    Exit-Code: 0 alle gewählten Prüfungen bestanden, 1 mindestens eine nicht bestanden,
    2 Aufruf- oder Baufehler. docs/hardware.md, Abschnitt "Bauen unter Windows".

.EXAMPLE
    tools/pruefe_abhaengigkeiten.ps1 -BuildDir build/release-cuda
.EXAMPLE
    tools/pruefe_abhaengigkeiten.ps1 -BuildDir build/release-cuda -MitSchutz
.EXAMPLE
    tools/pruefe_abhaengigkeiten.ps1 -BuildDir build/gcc -Teil abhaengigkeiten
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$BuildDir = 'build/release-cuda',
    [string]$Config = 'Release',
    # Header relativ zum Repo; Standard: mit CUDA include/mikroptim/cuda/smoke_trace.hpp (eine .cu-
    # und eine .cpp-Datei), sonst include/mikroptim/eval/fft.hpp
    [string]$Header = '',
    # Übersetzungseinheit, während deren Übersetzung Prüfung B den Header ändert
    [string]$Quelle = '',
    # Ziel, das Prüfung B baut (Standard: mikroptim_cuda bzw. mikroptim_tests)
    [string]$Ziel = '',
    [ValidateSet('alle', 'abhaengigkeiten', 'waehrend')]
    [string]$Teil = 'alle',
    [switch]$MitSchutz,
    [int]$Parallel = 8,
    # Höchstwartezeit in Sekunden, bis der Compiler die Quelle aus Prüfung B übersetzt
    [int]$Wartezeit = 900
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
if (-not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) {
    Write-Host "Kein konfiguriertes Build-Verzeichnis: $BuildDir"
    exit 2
}
$buildAbs = (Resolve-Path $BuildDir).Path
$cache = Get-Content (Join-Path $buildAbs 'CMakeCache.txt')
$mitCuda = [bool]($cache | Select-String -Pattern '^MIKROPTIM_ENABLE_CUDA:BOOL=(ON|TRUE|1)$')
$generator = (($cache | Select-String -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$').Matches[0].Groups[1].Value)
if (-not $Header) {
    $Header = if ($mitCuda) { 'include/mikroptim/cuda/smoke_trace.hpp' } else { 'include/mikroptim/eval/fft.hpp' }
}
if (-not $Quelle) {
    $Quelle = if ($mitCuda) { 'src/cuda/smoke_trace.cu' } else { 'tests/test_psf.cpp' }
}
if (-not $Ziel) {
    $Ziel = if ($mitCuda) { 'mikroptim_cuda' } else { 'mikroptim_tests' }
}
$headerAbs = (Resolve-Path $Header).Path
Write-Host "Build-Verzeichnis: $buildAbs ($generator, CUDA: $mitCuda), Header: $Header"
Write-Host ("Builds über: " + $(if ($MitSchutz) { 'tools/bauen.ps1' } else { 'cmake --build' }))

function Get-BuildArgv([string]$Ziel) {
    $a = @('--parallel', "$Parallel")
    if ($Ziel) { $a += @('--target', $Ziel) }
    if ($MitSchutz) {
        return @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'bauen.ps1'),
            '-BuildDir', $buildAbs, '-Config', $Config) + $a
    }
    return @('--build', $buildAbs, '--config', $Config) + $a
}

function Invoke-Build([string]$Log, [string]$Ziel = '') {
    $argv = Get-BuildArgv $Ziel
    if ($MitSchutz) { & powershell @argv *> $Log } else { & cmake @argv *> $Log }
    return $LASTEXITCODE
}

function Start-Build([string]$Log, [string]$Ziel = '') {
    $exe = if ($MitSchutz) { 'powershell' } else { 'cmake' }
    $p = Start-Process -FilePath $exe -ArgumentList (Get-BuildArgv $Ziel) -RedirectStandardOutput $Log `
        -RedirectStandardError "$Log.err" -PassThru -NoNewWindow
    $null = $p.Handle # sonst liefert ExitCode unter Windows PowerShell nichts
    return $p
}

# --- Orakel: transitive #include "..."-Hülle einer Datei ---------------------------------------
$includeRe = [regex]'(?m)^\s*#\s*include\s*"([^"]+)"'
$suchpfade = @('include', 'src', 'tests') | ForEach-Object { Join-Path $repo $_ }
$huellenCache = @{}
function Resolve-Include([string]$Name, [string]$Von) {
    foreach ($d in @((Split-Path $Von -Parent)) + $suchpfade) {
        $p = [IO.Path]::GetFullPath((Join-Path $d $Name))
        if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
    }
    return $null
}
function Get-Huelle([string]$Datei) {
    $gesehen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $stapel = New-Object System.Collections.Stack
    $stapel.Push($Datei)
    while ($stapel.Count -gt 0) {
        $c = $stapel.Pop()
        if (-not $huellenCache.ContainsKey($c)) {
            $text = [IO.File]::ReadAllText($c)
            $huellenCache[$c] = @($includeRe.Matches($text) | ForEach-Object { Resolve-Include $_.Groups[1].Value $c } |
                Where-Object { $_ })
        }
        foreach ($p in $huellenCache[$c]) {
            if ($gesehen.Add($p)) { $stapel.Push($p) }
        }
    }
    return , $gesehen # Komma: die Menge nicht in die Pipeline entrollen
}

# --- Übersetzungseinheiten und ihre Objektdateien ---------------------------------------------
function Get-Uebersetzungseinheiten {
    $liste = @()
    $cc = Join-Path $buildAbs 'compile_commands.json'
    if ($generator -notlike 'Visual Studio*' -and (Test-Path $cc)) {
        foreach ($e in (Get-Content $cc -Raw | ConvertFrom-Json)) {
            $out = if ([IO.Path]::IsPathRooted($e.output)) { $e.output } else { Join-Path $e.directory $e.output }
            $liste += [pscustomobject]@{ Quelle = [IO.Path]::GetFullPath($e.file); Objekt = [IO.Path]::GetFullPath($out) }
        }
        return $liste
    }
    foreach ($proj in Get-ChildItem $buildAbs -Recurse -Filter *.vcxproj) {
        [xml]$x = Get-Content $proj.FullName -Raw
        $ns = New-Object Xml.XmlNamespaceManager $x.NameTable
        $ns.AddNamespace('m', $x.DocumentElement.NamespaceURI)
        $intDir = $x.SelectNodes('//m:IntDir', $ns) | Where-Object { $_.Condition -like "*'$Config|*" } |
            Select-Object -First 1
        if (-not $intDir) { continue }
        # Combine statt Join-Path: Pfade in .vcxproj können absolut sein
        $int = [IO.Path]::Combine($proj.DirectoryName, $intDir.InnerText)
        foreach ($item in $x.SelectNodes('//m:ClCompile[@Include] | //m:CudaCompile[@Include]', $ns)) {
            $q = [IO.Path]::GetFullPath([IO.Path]::Combine($proj.DirectoryName, $item.Include))
            if (-not $q.StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)) { continue }
            $obj = Join-Path $int ([IO.Path]::GetFileNameWithoutExtension($q) + '.obj')
            $eigen = $item.SelectNodes('m:ObjectFileName', $ns) | Where-Object { $_.Condition -like "*'$Config|*" -or -not $_.Condition } |
                Select-Object -First 1
            if ($eigen -and $eigen.InnerText -notmatch '^\$\(IntDir\)$') {
                $obj = [IO.Path]::Combine($proj.DirectoryName, ($eigen.InnerText -replace '\$\(IntDir\)', $intDir.InnerText))
            }
            $liste += [pscustomobject]@{ Quelle = $q; Objekt = [IO.Path]::GetFullPath($obj) }
        }
    }
    return $liste
}

# Schreibzugriffe auf den Header wiederholen: MSBuild, Compiler oder Virenscanner halten ihn
# gelegentlich kurz offen
function Invoke-Wiederholt([scriptblock]$Aktion) {
    for ($i = 0; ; $i++) {
        try { & $Aktion; return }
        catch { if ($i -ge 100) { throw }; Start-Sleep -Milliseconds 100 }
    }
}

$original = [IO.File]::ReadAllBytes($headerAbs)
$originalZeit = (Get-Item $headerAbs).LastWriteTime
function Restore-Header {
    Invoke-Wiederholt { [IO.File]::WriteAllBytes($headerAbs, $original) }
    Invoke-Wiederholt { (Get-Item $headerAbs).LastWriteTime = $originalZeit }
}

$ergebnis = 0
$logDir = Join-Path ([IO.Path]::GetTempPath()) 'mikroptim_pruefe_abhaengigkeiten'
New-Item -ItemType Directory -Force $logDir | Out-Null

try {
    Write-Host 'Vorbereitung: Build auf aktuellen Stand bringen ...'
    if ((Invoke-Build (Join-Path $logDir 'vorbereitung.log')) -ne 0) {
        Write-Host "Baufehler in der Vorbereitung, siehe $logDir\vorbereitung.log"
        exit 2
    }

    if ($Teil -in 'alle', 'abhaengigkeiten') {
        $tus = @(Get-Uebersetzungseinheiten | Where-Object { (Get-Huelle $_.Quelle).Contains($headerAbs) })
        Write-Host ("A: {0} Übersetzungseinheit(en) binden {1} ein ({2} .cu)" -f $tus.Count, $Header,
            @($tus | Where-Object { $_.Quelle -like '*.cu' }).Count)
        $start = Get-Date
        Start-Sleep -Milliseconds 1100 # Zeitstempelauflösung
        Invoke-Wiederholt { [IO.File]::AppendAllText($headerAbs, "`n// pruefe_abhaengigkeiten.ps1`n") }
        if ((Invoke-Build (Join-Path $logDir 'teil_a.log')) -ne 0) {
            Write-Host "Baufehler in Prüfung A, siehe $logDir\teil_a.log"
            exit 2
        }
        $veraltet = @($tus | Where-Object { -not (Test-Path $_.Objekt) -or (Get-Item $_.Objekt).LastWriteTime -lt $start })
        foreach ($t in $tus) {
            $neu = $veraltet -notcontains $t
            Write-Host ("   {0}  {1}" -f $(if ($neu) { 'neu   ' } else { 'ALT   ' }), $t.Quelle.Substring($repo.Length + 1))
        }
        if ($veraltet.Count -eq 0 -and $tus.Count -gt 0) {
            Write-Host 'A abhaengigkeiten: BESTANDEN'
        } else {
            Write-Host ("A abhaengigkeiten: NICHT BESTANDEN ({0} veraltet)" -f $veraltet.Count)
            $ergebnis = 1
        }
        Restore-Header
    }

    if ($Teil -in 'alle', 'waehrend') {
        $quelleAbs = (Resolve-Path $Quelle).Path
        $quelleName = Split-Path $Quelle -Leaf
        $objekt = (Get-Uebersetzungseinheiten | Where-Object { $_.Quelle -eq $quelleAbs } | Select-Object -First 1).Objekt
        if (-not $objekt -or -not (Get-Huelle $quelleAbs).Contains($headerAbs)) {
            Write-Host "B waehrend: $Quelle ist keine Übersetzungseinheit dieses Builds oder bindet $Header nicht ein"
            exit 2
        }
        # Aussagekräftig ist ein Versuch nur, wenn der Compiler den Header schon gelesen hatte, als er
        # sich änderte, und die Objektdatei erst danach schrieb. Das Zeitfenster hängt von Quelle und
        # Rechner ab (nvcc liest die Header für jede Architektur erneut); daher sucht das Skript die
        # Verzögerung nach dem Compilerstart per Bisektion: Schlägt die Übersetzung fehl (#error
        # gelesen), war die Änderung zu früh; ist die Objektdatei vor der Änderung fertig, zu spät.
        $bewertet = $false
        $unten = 0.0
        $oben = -1.0
        $verzoegerung = 1.0
        for ($versuch = 1; $versuch -le 8 -and -not $bewertet; $versuch++) {
            # Header ändern, damit die Quelle im nächsten Build neu übersetzt wird
            Invoke-Wiederholt { [IO.File]::WriteAllBytes($headerAbs, $original) }
            Start-Sleep -Milliseconds 1100
            Invoke-Wiederholt { [IO.File]::AppendAllText($headerAbs, "`n// pruefe_abhaengigkeiten.ps1`n") }
            $log1 = Join-Path $logDir 'teil_b_build1.log'
            $start1 = Get-Date
            $p = Start-Build $log1 $Ziel
            $gesehen = $false
            $ende = (Get-Date).AddSeconds($Wartezeit)
            while (-not $p.HasExited -and (Get-Date) -lt $ende) {
                $c = Get-CimInstance Win32_Process -Filter "Name='nvcc.exe' OR Name='cl.exe' OR Name='cc1plus.exe'" |
                    Where-Object { $_.CommandLine -like "*$quelleName*" }
                if ($c) { $gesehen = $true; break }
                Start-Sleep -Milliseconds 100
            }
            if (-not $gesehen) {
                $p.WaitForExit()
                Write-Host "B: Übersetzung von $quelleName nicht beobachtet"
                break
            }
            Start-Sleep -Milliseconds ([int]($verzoegerung * 1000))
            $aenderung = Get-Date
            Invoke-Wiederholt { [IO.File]::AppendAllText($headerAbs, "#error MKO_PRUEFE_ABHAENGIGKEITEN`n") }
            # Während des Builds beobachten (mit -MitSchutz baut tools/bauen.ps1 danach selbst neu,
            # und ein fehlschlagender Compilerlauf kann die Objektdatei wieder löschen)
            $neuesteZeit = [datetime]::MinValue
            while ($true) {
                $o = Get-Item $objekt -ErrorAction SilentlyContinue
                if ($o -and $o.LastWriteTime -gt $neuesteZeit) { $neuesteZeit = $o.LastWriteTime }
                if ($p.HasExited) { break }
                Start-Sleep -Milliseconds 100
            }
            $p.WaitForExit()
            if ($neuesteZeit -le $start1) {
                Write-Host ("B: Versuch {0}: Änderung nach {1:0.###} s zu früh (Übersetzung sah den #error)" -f
                    $versuch, $verzoegerung)
                $unten = $verzoegerung
                $verzoegerung = if ($oben -lt 0) { 2 * $verzoegerung } else { ($unten + $oben) / 2 }
                continue
            }
            if ($neuesteZeit -le $aenderung) {
                Write-Host ("B: Versuch {0}: Änderung nach {1:0.###} s zu spät (Objektdatei schon geschrieben)" -f
                    $versuch, $verzoegerung)
                $oben = $verzoegerung
                $verzoegerung = ($unten + $oben) / 2
                continue
            }
            Write-Host ("B: Header {0:0.###} s nach dem Start von {1} geändert; Objektdatei danach mit altem Stand geschrieben" -f
                $verzoegerung, $quelleName)
            $log2 = Join-Path $logDir 'teil_b_build2.log'
            $code2 = Invoke-Build $log2 $Ziel
            $fehlerGemeldet = [bool](Select-String -Path $log2 -Pattern 'MKO_PRUEFE_ABHAENGIGKEITEN' -SimpleMatch -Quiet)
            # Die Quelle selbst muss im Folgebuild übersetzt werden (andere Einheiten, die den
            # Header erst nach der Änderung lasen, melden den #error ohnehin)
            $quelleUebersetzt = [bool](Select-String -Path $log2 -Pattern $quelleName -SimpleMatch -Quiet)
            Write-Host (("   Folgebuild: Exit {0}, #error gemeldet: {1}, {2} erneut übersetzt: {3}") -f
                $code2, $fehlerGemeldet, $quelleName, $quelleUebersetzt)
            if ($code2 -ne 0 -and $fehlerGemeldet -and $quelleUebersetzt) {
                Write-Host 'B waehrend: BESTANDEN'
            } else {
                Write-Host 'B waehrend: NICHT BESTANDEN (veraltete Objektdatei bleibt liegen)'
                $ergebnis = 1
            }
            $bewertet = $true
        }
        if (-not $bewertet) {
            Write-Host 'B waehrend: kein aussagekräftiger Versuch (andere Quelle mit -Quelle wählen)'
            $ergebnis = [Math]::Max($ergebnis, 2)
        }
    }
} finally {
    Restore-Header
    Write-Host 'Abschluss: Header wiederhergestellt, baue ...'
    # Quellen- und Objektstand angleichen: der Header bekommt einen neuen Zeitstempel, damit
    # auch nach Prüfung B alles aus dem Originalinhalt übersetzt wird
    Invoke-Wiederholt { (Get-Item $headerAbs).LastWriteTime = Get-Date }
    if ((Invoke-Build (Join-Path $logDir 'abschluss.log')) -ne 0) {
        Write-Host "Baufehler beim Abschluss, siehe $logDir\abschluss.log"
        $ergebnis = 2
    }
}
exit $ergebnis
