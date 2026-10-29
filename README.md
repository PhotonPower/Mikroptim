# Mikroptim

GPU-beschleunigtes Raytracing und automatische Systemsynthese für hochaperturige Mikroskopobjektive.

Mikroptim optimiert sequenzielle Linsensysteme aus sphärischen Flächen auf CUDA-GPUs. Die Optimierung beginnt mit sehr einfachen Systemen und lässt sie wachsen: Versteckte Freiheitsgrade („Geisterplatten“) zeigen an, *wo* und *welche* Korrektur dem realen System fehlt, und neue Elemente werden über Einfüge-Operatoren ohne Sprung in der Meritfunktion eingeschaltet. Die nötige Rechenleistung entsteht dadurch, dass sehr viele Systemvarianten gleichzeitig getract werden (Multistart, Homotopie-Zweige, Jacobi-Spalten, Tolerierung).

## Status

M0 und M1 abgeschlossen: Systeme und Glaskataloge aus JSON laden, sequenzieller CPU-Referenztracer, Paraxialoptik, Pupillenquadratur, OPD und RMS-Wellenfront, Kommandozeile `mikroptim trace` ([docs/cli.md](docs/cli.md)). M2 (CUDA-Vorwärtskernel) umgesetzt: SystemBatch, Kernel `RESIDUALS` und `MERIT` mit deterministischer Reduktion, `TraceEngine` mit Tranchen, `mikroptim trace --gpu`, Benchmark ([docs/benchmarks.md](docs/benchmarks.md)). M3 (gebündelter Levenberg-Marquardt, `mikroptim optimize`) und M4 (Merit in Wellen, weiches Strahlversagen, mechanische, Petzval- und paraxiale Nebenbedingungen, Geisterplatten mit L1-Strafe und Diagnosebericht) abgeschlossen. Aus M6 umgesetzt: Glas als Variable mit konvexer Hülle, Einfüge-Operatoren, Homotopie-Controller und Strahlsuche über die Kandidaten (`mikroptim synth --beam`), Glas-Einrasten (`mikroptim snap`); aus M7: df64, FP32-Vorfilter, mehrere GPUs; aus M8: Modus EVAL, skalare und vektorielle PSF (`mikroptim eval`), Dezentrierung und Verkippung einzelner Flächen (ADR-0011). In Arbeit: M5 (adjungierte Ableitungen) und M6 (Synthese). Offene Maintainer-Entscheidungen stehen in den PRs. Die Implementierung erfolgt nach dem [Meilensteinplan](docs/plan/meilensteine.md) durch Programmieragenten. Regeln für Agenten stehen in [AGENTS.md](AGENTS.md).

## Erstes Zielsystem

| Größe | Wert |
|---|---|
| Typ | Plan-Apochromat für Auflicht (Epi), trocken, ohne Deckglas |
| Vergrößerung / NA | 50x / 0,95 |
| Tubuslinse | Zeiss ICS, f = 164,5 mm |
| Feldzahl | 25 |
| Abgleichlänge | 45 mm |
| Spektralbereich | voll korrigiert 436–656 nm, 405 und 700 nm schwach gewichtet |

Details und abgeleitete Kenngrößen: [docs/spezifikation.md](docs/spezifikation.md).

## Kernideen

- **Batch-first:** Parallelisiert wird primär über Systemvarianten, sekundär über Strahlen ([ADR-0004](docs/adr/0004-batch-ueber-varianten.md)).
- **FP64 als Standard:** Der Tracekern ist über den Zahlentyp templatisiert (FP64, später df64, FP32 nur zur Vorfilterung) ([ADR-0003](docs/adr/0003-gleitkomma.md)).
- **Ein Mathe-Kern für CPU und GPU:** Flächenoperationen sind header-only und `__host__ __device__` ([ADR-0005](docs/adr/0005-gemeinsamer-mathe-kern.md)).
- **Geisterplatten mit L1-Strafe:** virtuelle dünne Gläser, deren Benutzung teuer ist und die nur aktiv werden, wenn das reale System die Aufgabe nicht schafft ([docs/optimierung.md](docs/optimierung.md)).
- **Homotopie:** NA, Feld, Spektrum und Strafgewichte werden schrittweise hochgefahren.

## Bauen

Voraussetzungen: CMake ≥ 3.25, ein C++20-Compiler (GCC ≥ 11, Clang ≥ 14 oder MSVC 2022), optional CUDA Toolkit **12.x** (nicht 13, siehe [ADR-0002](docs/adr/0002-cuda-12.md)).

```bash
# nur CPU
cmake --preset release-cpu
cmake --build --preset release-cpu
ctest --preset release-cpu

# mit CUDA 12 oder 13 (RTX 3090 = sm_86, RTX 6000 Ada = sm_89; ADR-0015)
cmake --preset release-cuda
cmake --build --preset release-cuda
ctest --preset release-cuda
./build/release-cuda/mikroptim --devices

# Beispiel: Paraxialdaten und RMS-Wellenfront einer Testlinse
./build/release-cpu/mikroptim trace data/systems/einzellinse_bk7.json
```

Mit Visual Studio liegen die Programme unter `build/<preset>/Release/` bzw. `Debug/`. Sind unter Windows mehrere CUDA-Toolkits installiert, wählt `-T "cuda=<Pfad zum CUDA-Toolkit>"` beim Konfigurieren das richtige (siehe [docs/hardware.md](docs/hardware.md)).

Ein System auswerten (Paraxialdaten, RMS-Wellenfront pro Feld und Wellenlänge, versagte Strahlen; mit `--json` maschinenlesbar, siehe [docs/cli.md](docs/cli.md)):

```bash
./build/release-cpu/mikroptim trace data/systems/einzellinse_bk7.json
./build/release-cpu/mikroptim trace data/systems/einzellinse_bk7.json --json
```

Die Variablen eines Systems optimieren (Levenberg-Marquardt, Iterationsprotokoll, Ergebnis als neue Systemdatei; mit `--gpu` auf dem Gerät, mit `--runs K` als Multistart, siehe [docs/cli.md](docs/cli.md#mikroptim-optimize)):

```bash
./build/release-cpu/mikroptim optimize data/systems/einzellinse_bk7_biegung.json -o biegung_opt.json
./build/release-cpu/mikroptim trace biegung_opt.json
```

Mit Visual Studio liegt das Programm unter `build/<preset>/Release/mikroptim.exe` bzw. `Debug/`.

Aus Python (Modul `mikroptim`, Ergebnisse als NumPy-Arrays; Installation und API: [python/README.md](python/README.md)):

```bash
python -m pip wheel . -w dist --no-deps && python -m pip install --no-index --find-links dist mikroptim
python -c "import mikroptim as m; print(m.trace(m.load_system('data/systems/einzellinse_bk7.json')).rms_waves)"
```

## Verzeichnisstruktur

```
include/mikroptim/   öffentliche Header (Mathe-Kern host+device, Datenmodell, APIs)
src/cli/             Kommandozeilenprogramm
src/cuda/            CUDA-Backend (Kernel, Batch-Trace, Reduktionen)
tests/               GoogleTest-Tests (analytische Orakel, CPU/GPU-Abgleich)
bench/               Benchmarks
python/              Python-Bindings (nanobind, Modul mikroptim, M9-T1)
tools/               Hilfsskripte (z. B. Generator der Quadraturtabelle)
data/systems/        Testsysteme und Startsysteme (JSON)
data/glass/          Glasdaten (JSON)
docs/                Spezifikation, Architektur, Numerik, Optimierung, ADRs, Plan
```

## Dokumentation

Einstieg: [docs/README.md](docs/README.md).

## Lizenz

MIT, siehe [LICENSE](LICENSE).
