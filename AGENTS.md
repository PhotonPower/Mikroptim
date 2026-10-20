# AGENTS.md – Arbeitsanweisung für Programmieragenten

Diese Datei ist verbindlich für alle Agenten, die an Mikroptim arbeiten. Bei Widerspruch zwischen dieser Datei und anderen Dokumenten gilt: ADRs in `docs/adr/` > AGENTS.md > übrige Doku. Unklarheiten, die eine der Entscheidungen unter „Nicht selbst entscheiden“ berühren, werden als Frage an den Maintainer im PR formuliert, nicht still aufgelöst.

## 1. Worum es geht

Mikroptim ist ein CUDA-beschleunigter sequenzieller Raytracer mit Optimierer für Mikroskopobjektive aus sphärischen Linsen. Erstes Ziel: ein Plan-Apochromat 50x/0,95 für Zeiss ICS (Tubuslinse 164,5 mm). Das Besondere ist die **Systemsynthese**: Das System wächst von einer Linse aus, gesteuert durch Geisterplatten (versteckte, teuer bestrafte Freiheitsgrade) und Homotopie über NA, Feld und Spektrum.

## 2. Lesereihenfolge vor jeder Aufgabe

1. `README.md`
2. diese Datei
3. `docs/plan/meilensteine.md` – deine Aufgabe, ihre Abhängigkeiten und Abnahmekriterien
4. die in der Aufgabe genannten Fachdokumente, mindestens:
   - `docs/numerik.md` (Konventionen, Formeln, Genauigkeit) – **immer** bei Arbeit am Tracer
   - `docs/architektur.md` (Schichten, Datenlayout, Kernel-Modi)
   - `docs/optimierung.md` (Merit, LM, Geisterplatten, Homotopie)
   - `docs/hardware.md` (Zielhardware, CUDA-Regeln)
5. alle ADRs in `docs/adr/` (kurz, bitte vollständig lesen)

## 3. Arbeitsweise

- **Eine Aufgabe = ein Branch = ein PR.** Branchname `m<M>/t<N>-<kurzname>`. Aufgaben nur beginnen, wenn ihre Abhängigkeiten gemergt sind.
- **Tests zuerst denken.** Jede Aufgabe hat Abnahmekriterien; jedes Kriterium wird durch einen automatisierten Test oder ein reproduzierbares Skript belegt. Analytische Orakel sind Vergleichen mit dem eigenen Code vorzuziehen.
- **Klein bleiben.** Keine Arbeit an Folgeaufgaben „nebenbei“. Gefundene Probleme außerhalb der Aufgabe als Notiz im PR auflisten.
- **Doku mitziehen.** Wenn du eine Formel, ein Datenlayout oder eine API festlegst, gehört sie in das passende Dokument unter `docs/`. `CHANGELOG.md` unter „Unreleased“ ergänzen.
- **PR-Vorlage vollständig ausfüllen**, jedes Abnahmekriterium mit Ergebnis (Zahl, Testname) auflisten.

## 4. Bauen und Testen

```bash
cmake --preset debug-cpu && cmake --build --preset debug-cpu && ctest --preset debug-cpu
cmake --preset release-cuda && cmake --build --preset release-cuda && ctest --preset release-cuda
./build/release-cuda/mikroptim --devices
```

Die CI kompiliert CUDA, hat aber **keine GPU**. GPU-Tests müssen lokal laufen: auf der RTX 3090 (sm_86) und der Tesla P100 (sm_60). Tests, die eine GPU brauchen, prüfen `mko::cuda::device_count()` und rufen sonst `GTEST_SKIP()` auf.

## 5. Code-Konventionen

- C++20, Namensraum `mko` (CUDA-Host-APIs in `mko::cuda`). Typen `PascalCase`, Funktionen und Variablen `snake_case`, Konstanten `kPascalCase`, Makros `MKO_…`.
- Header `.hpp`, Quellen `.cpp`/`.cu`. Öffentliche Header unter `include/mikroptim/`, interne Header neben den Quellen.
- Formatierung nach `.clang-format`.
- Code, der auf Host **und** Device läuft, ist header-only, als `MKO_HD` markiert und über den Zahlentyp `R` templatisiert. Er enthält keine Ausnahmen, keine dynamische Allokation, keine `std::`-Container und keine virtuellen Funktionen.
- Öffentliche Header unter `include/mikroptim/cuda/` sind **reines C++** ohne CUDA-Typen, damit Host-Code ohne nvcc sie einbinden kann.
- Host-Code: Fehler beim Einlesen und Konfigurieren als Ausnahmen; im Tracepfad Statuscodes (`TraceStatus`).
- Jeder CUDA-API-Aufruf wird geprüft (`MKO_CUDA_TRY` oder gleichwertig). Nach jedem Kernelstart `cudaGetLastError()`.

## 6. Numerik-Regeln (nicht verhandelbar ohne ADR)

1. **Kein FP32 im OPD-Pfad.** Optische Weglängen, Positionen und Richtungen im Merit- und Auswertungspfad sind FP64 (später alternativ df64). FP32 nur in ausdrücklich als „Vorfilterung“ gekennzeichneten Pfaden.
2. **Keine fast-math-Optionen** (`-ffast-math`, `/fp:fast`, `--use_fast_math`, `-prec-div=false`, `-prec-sqrt=false`).
3. **Deterministische Reduktionen:** keine `atomicAdd` auf Gleitkommazahlen für Ergebnisse, die in die Merit oder Jacobi-Matrix eingehen. Baumreduktion im Block, danach feste Reihenfolge. Zwei Läufe mit gleichen Eingaben liefern bitgleiche Ergebnisse auf derselben GPU.
4. Konventionen (Koordinaten, Vorzeichen, Einheiten, Tracerichtung) aus `docs/numerik.md` exakt einhalten. Längen in mm, Wellenlängen in den Dateien in nm, in Dispersionsformeln in µm.
5. Toleranzen in Tests kommen aus `tests/tolerances.hpp`. Wer eine Toleranz lockern muss, begründet das im PR.

## 7. CUDA-Regeln

- Toolkit **CUDA 12.x** (Empfehlung 12.9). **Nicht CUDA 13**: Es unterstützt die Tesla P100 nicht (ADR-0002). CMake bricht bei 13 ab; das nicht umgehen.
- Architekturen `60-real;86-real;86-virtual`. Nicht ändern ohne ADR.
- Warp-Operationen nur mit `_sync`-Varianten (`__shfl_down_sync`, `__syncwarp`, `__ballot_sync`). Keine Annahmen über implizit synchrone Warps.
- Shared Memory pro Block ≤ 48 KB (kein Opt-in nötig, läuft auf beiden Karten).
- Keine Tensor Cores, keine Features oberhalb sm_60 ohne Fallback.
- Batchgrößen zur Laufzeit an `cudaMemGetInfo` ausrichten, nie fest verdrahten (P100: 16 GB, RTX 3090: 24 GB).
- Parallelisierung: ein Block pro (Variante, Feld, Wellenlänge), Threads über Pupillenstrahlen (ADR-0004, `docs/architektur.md`). Strahlen, die versagen, laufen maskiert weiter statt früh auszusteigen.

## 8. Daten und Sicherheit

- **Niemals** Tokens, Passwörter oder andere Geheimnisse in Code, Konfiguration, CI oder Commit-Nachrichten.
- Herstellerkataloge für Gläser (z. B. `.agf`) nur committen, wenn ihre Lizenz das erlaubt. Sonst nach `data/glass/private/` (ignoriert) und nur das Ladeformat dokumentieren.
- Patentdaten von Startsystemen trägt der Maintainer ein. Agenten erfinden keine Linsendaten für reale Objektive.

## 9. Nicht selbst entscheiden – Maintainer fragen

- Änderungen an Toolchain, CUDA-Version, Zielarchitekturen oder Lizenz
- Änderungen an Koordinaten- und Vorzeichenkonventionen oder an der Tracerichtung
- Inkompatible Änderungen am Dateiformat (Hauptversion von `mikroptim.system/…`)
- Abweichungen von den Numerik-Regeln in Abschnitt 6
- Externe Abhängigkeiten über GoogleTest, CUDA-Bibliotheken (cuBLAS, cuSOLVER, cuFFT), nlohmann/json, nanobind und (nur zum Bauen der Python-Wheels) scikit-build-core hinaus

## 10. Definition of Done

Eine Aufgabe ist fertig, wenn

- alle Abnahmekriterien belegt sind,
- CI grün ist (CPU auf Linux und Windows, CUDA kompiliert),
- bei CUDA-Code die GPU-Tests lokal auf beiden Karten grün sind,
- Doku, Beispiele und `CHANGELOG.md` aktualisiert sind,
- keine neuen Compiler-Warnungen entstehen (`MIKROPTIM_WARNINGS_AS_ERRORS=ON`).
