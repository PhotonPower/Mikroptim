# Python-Bindings (`mikroptim`)

Python-Modul für den C++-Kern (M9-T1, [nanobind](https://github.com/wjakob/nanobind) 3.1.0): Systeme laden, Paraxialdaten, Wellenfront, Auswertung mit PSF/MTF, Optimierung, Batches vieler Varianten und die TraceEngine. Ergebnisse sind NumPy-Arrays (`float64`, Ganzzahlen `int64`, Status `uint8`). Gerechnet wird mit denselben Funktionen wie in der CLI ([docs/cli.md](../docs/cli.md)); die Tests prüfen, dass die Ergebnisse bitgleich zur CLI sind.

## Installation

Voraussetzungen: Python ≥ 3.10, NumPy, ein C++20-Compiler und CMake ≥ 3.25 (wie für den C++-Teil, siehe [README.md](../README.md)). Gebaut wird über [scikit-build-core](https://scikit-build-core.readthedocs.io) (`pyproject.toml` im Wurzelverzeichnis); nanobind lädt CMake per FetchContent in fester Version (Git, mit Untermodulen).

```bash
# Wheel bauen (nur CPU) und installieren
python -m pip wheel . -w dist --no-deps
python -m pip install --no-index --find-links dist mikroptim
python -m pip install numpy pytest          # falls noch nicht vorhanden

# Tests (aus dem Repository-Wurzelverzeichnis)
python -m pytest python/tests
```

Die Version des Pakets stammt aus `include/mikroptim/version.hpp` (gleich `project(VERSION …)` in CMake). Das Wheel enthält nur das Erweiterungsmodul (keine Tests, kein Programm); `MIKROPTIM_BUILD_TESTS` und `MIKROPTIM_BUILD_BENCH` sind dabei aus.

**Linux und Windows.** Lokal ist das Wheel unter Windows (MSVC, Python 3.11) gebaut und getestet; Linux-Wheels (GCC) baut die CI im Job „Python-Wheel“ (`.github/workflows/ci.yml`, Ubuntu 24.04 und Windows, Python 3.12) und führt dort die pytest-Suite aus. Das Linux-Wheel trägt das Plattform-Tag `linux_x86_64` (kein manylinux, also für den Rechner, auf dem es gebaut wurde).

### Mit CUDA

Standard ist ein reines CPU-Wheel. Mit CUDA (Toolkit 12.x wie für den C++-Teil, [docs/hardware.md](../docs/hardware.md)):

```bash
MIKROPTIM_ENABLE_CUDA=ON python -m pip wheel . -w dist --no-deps          # Linux
```

```powershell
# Windows: Toolkit 12.9 ausdrücklich wählen, wenn ein neueres (13.x) im PATH liegt
$env:MIKROPTIM_ENABLE_CUDA = "ON"
python -m pip wheel . -w dist --no-deps -C cmake.args="-T cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
```

Das CUDA-Wheel braucht zur Laufzeit die CUDA-Laufzeitbibliotheken (cudart, cuFFT) des Toolkits. Unter Windows trägt `mikroptim/__init__.py` das `bin`-Verzeichnis des Toolkits, mit dem gebaut wurde, per `os.add_dll_directory` ein (Angabe in `mikroptim/_build_info.py`); ein anderes Verzeichnis lässt sich mit der Umgebungsvariable `MIKROPTIM_CUDA_BIN_DIR` angeben. Ohne CUDA-Build oder ohne Gerät werfen GPU-Funktionen `mikroptim.GpuUnavailableError`.

### Im CMake-Build (Entwicklung, ctest)

```bash
cmake --preset release-cpu -DMIKROPTIM_BUILD_PYTHON=ON
cmake --build --preset release-cpu
ctest --preset release-cpu -R PythonBindings
```

Mit `MIKROPTIM_BUILD_PYTHON=ON` (Standard `OFF`) baut CMake das Modul `mikroptim._mikroptim` nach `build/<preset>/python/mikroptim/` (zusammen mit den Python-Dateien) und registriert den ctest `PythonBindings.pytest`. Er startet pytest mit `PYTHONPATH=build/<preset>/python` und vergleicht mit dem Programm `mikroptim` desselben Builds (Umgebungsvariable `MIKROPTIM_CLI`); fehlen pytest oder NumPy, wird er übersprungen. Alle statischen Bibliotheken werden dann mit `-fPIC` gebaut.

## Beispiel

```python
import numpy as np
import mikroptim as mko

system = mko.load_system("data/systems/einzellinse_bk7.json")
p = mko.paraxial(system)                     # bei λ_ref, wie `mikroptim trace`
print(p.focal_length, p.to_dict())           # to_dict(): Aufbau wie das JSON der CLI

wf = mko.trace(system, rings=10, spokes=8)   # gpu=True, precision="df64" mit CUDA
print(wf.rms_waves)                          # (Felder, Wellenlängen), Wellen bei λ_ref
opd = wf.samples(1, 0)["opd_mm"]             # OPD der Stützstellen eines Tupels

ev = mko.evaluate(system, grid=64)           # Modus EVAL
karte = ev.opd_map(1, 1, remove="tilt")      # (N, N), NaN außerhalb der Pupille
psf = ev.psf(1, 1)                           # skalare PSF, MTF, Strehl
print(psf.strehl, psf.mtf_sagittal[:5])

start = mko.load_system("data/systems/einzellinse_bk7_biegung.json")
res = mko.optimize(start, max_iterations=100)          # wie `mikroptim optimize`
print(res.status, res.merit, res.history)              # Verlauf als NumPy
res.system.save("biegung_opt.json")

batch = mko.SystemBatch(start, 1000)         # viele Varianten
batch.curvatures[:, 1] *= 1.0 + 1e-3 * np.random.standard_normal(1000)
merit = mko.merit_cpu(batch)["merit"]        # oder mko.TraceEngine(0).merit(batch)
```

Die CLI ist auch aus Python erreichbar: `python -m mikroptim trace system.json --json` bzw. `mko.run_cli([...])` (Rückgabe `(exit_code, stdout, stderr)`; Ausgabe am Ende, kein fortlaufendes Protokoll).

## API

| Name | Inhalt |
|---|---|
| `load_system(path)` | Systemdatei samt Glaskatalogen laden → `System` |
| `System` | unveränderlich; Eigenschaften `name`, `file`, `num_surfaces`, `surface_types`, `curvatures`, `thicknesses`, `semi_diameters`, `materials`, `indices` (S, L), `wavelengths_nm`, `reference_nm`, `source_type`, `field_angles_deg`, `target`, `warnings`, …; `variables()`; `to_json()`, `System.from_json(text, base_dir)`, `save(path)` (Glaskataloge relativ zur neuen Datei) |
| `system_to_dict`, `system_from_dict` | JSON-Export/Import als `dict` (zum Ändern von Systemen) |
| `paraxial(system, wavelength_nm=None)` | `Paraxial` (Brennweite, Schnittweiten, Hauptebenen, Petzval, Austrittspupille, `abcd` (2, 2)); `to_dict()` wie das Feld `paraxial` von `mikroptim trace --json` |
| `trace(system, rings=10, spokes=8, gpu=False, device=0, precision="fp64")` | `Wavefront`: `rms_waves`, `rms_mm`, `piston_mm`, `rays`, `failed` (F, L), `image_points` (F, 3), `pupil_modes`, `samples(f, l)` (rho, phi, weight, opd_mm, opd_waves, status) |
| `evaluate(system, grid=64, half=False, reference="exit-pupil", gpu=False, …)` | `Evaluation`: `stats` (dict von (F, L)-Arrays), `opd_map(f, l, remove)`, `spots(f, l)`, `pupil_points()`, `psf(f, l, padding=4, numerical_aperture=0, gpu=False)` → `Psf` (`psf` (M, M), `mtf_sagittal`, `mtf_tangential`, `frequencies()`, `strehl`, `dx`, `airy_radius`, …) |
| `optimize(system, max_iterations=100, rings=10, spokes=8, runs=1, perturbation=0.05, seed=0, fail_weight=None, hard_failure=False, gpu=False, device=0, jacobian="fd", callback=None)` | `OptimizeResult` (`jacobian` `"fd"` oder `"adjoint"` wie `--jacobian`): `system` (Ergebnis), `status`, `converged`, `merit`, `start_merit`, `history`, `x`, `x0`, `variable_names`, `bounds`, `runs`, `log` (Iterationsprotokoll als dict von Arrays), `warnings`; `callback(info)` nach jedem Zyklus |
| `SystemBatch(system, variants=1)` | V Varianten; Sichten `curvatures`, `thicknesses` (V, S), `indices` (V, S, L), `pupil_radius`, `field_angles`, `field_targets`, `ghost_coeffs`, …; `get_variables()`/`set_variables(x)` (V, P), `copy()`, `select(first, count)`, `validate()` |
| `residuals_cpu(batch, …)`, `merit_cpu(batch, …)` | Residuen (V, stride), OPD, Status, RMS (V, F, L) bzw. Merit pro Variante als dict |
| `TraceEngine(device=0)` | GPU: `residuals(batch, …)`, `merit(batch, …, precision)`, `release()` |
| `has_cuda`, `device_count()`, `devices()`, `gpu_available()` | GPU-Abfrage |
| `run_cli(args)` | Unterbefehl der CLI im Prozess |

Einheiten wie im C++-Teil: Längen in mm, Wellenlängen in nm, Winkel im Batch in rad (`System.field_angles_deg` in Grad), Merit in Wellen² bei λ_ref ([docs/optimierung.md](../docs/optimierung.md)). Flächenindizes von `System` sind Dateiflächen, die eines `SystemBatch` Tabellenflächen (Geisterplatten als eingefügte Flächen, [docs/dateiformat.md](../docs/dateiformat.md)).

### Kopien und Sichten

| Wo | Art |
|---|---|
| Eigenschaften von `System`, Methoden-Rückgaben (`samples`, `opd_map`, `stats`, `runs`, `log`, `history`, `residuals_cpu`, …) | **Kopie**: eigenes NumPy-Array, Ändern wirkt nicht zurück |
| Array-Eigenschaften von `Wavefront` und `Psf` | **schreibgeschützte Sicht** auf das Ergebnisobjekt (kein Kopieren); das Array hält das Objekt am Leben |
| Array-Eigenschaften von `SystemBatch` | **beschreibbare Sicht** auf den Speicher des Batches: Änderungen wirken auf die nächste Rechnung mit diesem Batch; das Array hält den Batch am Leben (seine Größe ist fest, die Sicht bleibt gültig). `copy()` und `select()` liefern unabhängige Batches |

Eingaben (`set_variables`) verlangen C-zusammenhängende `float64`-Arrays; andere werden von nanobind umgewandelt (Kopie).

### Fehler

| C++ | Python |
|---|---|
| `FormatError` (Systemdatei, mit JSON-Pfad) | `mikroptim.FormatError` (Unterklasse von `ValueError`) mit `json_path`, `message`, `file` |
| `GlassError` | `mikroptim.GlassError` (`ValueError`) |
| `std::invalid_argument` | `ValueError` |
| `std::out_of_range` | `IndexError` |
| `CudaError` | `mikroptim.CudaError` (`RuntimeError`) |
| GPU nicht verfügbar (ohne CUDA gebaut, kein Gerät) | `mikroptim.GpuUnavailableError` (`RuntimeError`) |
| Datei fehlt (`load_system`) | `FileNotFoundError` |

Lange Rechnungen (Laden, Trace, Auswertung, Optimierung) geben die GIL frei; der Rückruf von `optimize` läuft im aufrufenden Thread mit GIL, eine Ausnahme darin bricht die Optimierung ab und wird weitergereicht.

### Grenzen (M9-T1)

- Nicht gebunden: Systemsynthese (`mikroptim synth`), Einfüge-Operatoren, Glas-Einrasten, vektorielle PSF, Mehr-GPU-Engine; über `run_cli` sind die Unterbefehle aber erreichbar.
- `OptimizeResult` enthält keine Berichte der Nebenbedingungen und keine Geisterdiagnose (die stehen in `mikroptim optimize --json`).
