# Benchmarks

Ergebnisse werden in [docs/benchmarks.md](../docs/benchmarks.md) eingetragen (Messumgebung mit GPU, Treiber, CUDA-Version, Build-Typ, Datum und Commit).

## Bauen

Die Benchmarks entstehen nur in Builds mit `MIKROPTIM_ENABLE_CUDA=ON` und der Option `MIKROPTIM_BUILD_BENCH` (Standard `ON`). Die CI kompiliert sie im CUDA-Job, führt sie aber nicht aus (keine GPU).

```bash
cmake --preset release-cuda && cmake --build --preset release-cuda --target trace_throughput
./build/release-cuda/bench/trace_throughput            # Linux (Makefile/Ninja)
.\build\release-cuda\bench\Release\trace_throughput.exe  # Windows (Visual Studio)
```

Messungen nur mit einem Release-Build; ein Debug-Build warnt auf stderr.

## trace_throughput (M2-T6)

Misst den Durchsatz der Trace-Kernel über die öffentliche `mko::cuda::TraceEngine` für synthetische Systeme mit S Flächen und V Varianten.

**Systeme.** Frei erfundene Folgen schwacher Menisken aus einem Kron- und einem Flintglas (Buchdahl-Modelle), bei ungeradem S − 1 mit einem verkitteten Glied am Ende; Blende 2 mm Radius, Summe der dünnen Brechkräfte 1/40 mm⁻¹, Bildebene im paraxialen Fokus bei 587,56 nm, Felder 0 … 3°, Wellenlängen gleichmäßig 486,13 … 656,27 nm. Variante 0 ist ungestört, alle anderen sind mit festem Seed leicht gestört (Krümmungen und Dicken ±2 %, Arbeitsabstand ±1 %, Brechzahlen ±0,002, Pupillenradius ±5 %), sodass praktisch keine Strahlen versagen. Der Anteil versagter Strahlen wird pro Fall ausgegeben.

**Messgrößen** pro Fall (Zahlentyp, Modus, S, V), jeweils Median und Minimum aus `--repeat` Läufen nach `--warmup` Aufwärmläufen, gemessen mit der Host-Uhr (`std::chrono::steady_clock`):

- **Engine:** `TraceEngine::run` – Prüfung der Werte auf dem Host, Upload des Batches, Kernel, Download (RESIDUALS mit Status pro Strahl, ohne OPD; MERIT mit Merit und versagten Strahlen pro Variante). Das ist die Zeit, die ein Optimierer pro Aufruf sieht.
- **Kernel:** `DeviceResiduals::run` bzw. `DeviceMerit::run` auf einem schon hochgeladenen `DeviceSystemBatch` (Feldablage hochladen, Kernel bzw. beide Stufen von MERIT, Synchronisation), ohne Upload und Download. Die Engine stellt keine CUDA-Events bereit; die Messung um den synchronen Aufruf enthält deshalb die Start- und Synchronisationslatenz (ca. 50 µs), die nur für kleine V ins Gewicht fällt. Passt ein Batch nicht in einem Stück auf das Gerät, entfällt die Kernelmessung („-“); die Engine-Messung teilt dann in Tranchen (Spalte `Tr`).
- Strahlen/s = V · Strahlen pro Variante / Zeit, Strahl-Flächen/s = Strahlen/s · S.
- **Modi `adjoint-*` (M5-T1):** Grundgerüst des Modus ADJOINT (Vorwärtsdurchlauf mit Aufzeichnung pro Strahl im Local Memory und Lesen der Aufzeichnung in absteigender Reihenfolge wie im späteren Rückwärtsdurchlauf; `none` ohne Aufzeichnung, `full` alle Zustände, `checkpoint` Zustände alle 8 Flächen mit Neu-Vorwärtsrechnen der Abschnitte). Noch nicht in der Engine: „Engine“ ist hier Upload, `DeviceAdjointForward::run` und Download (Residuen, Status, OPL, Prüfsummen), „Kernel“ der Lauf auf dem hochgeladenen Batch (eine Tranche, Spalte `Tr` = 1).

**Optionen** (Listen durch Komma getrennt):

| Option | Bedeutung | Standard |
|---|---|---|
| `--device N` | CUDA-Gerät | 0 |
| `--surfaces LISTE` | Flächenzahlen S (4 … 64) | 10,20,30,40 |
| `--variants LISTE` | Variantenzahlen V ≥ 1 | 1,10,100,1000,10000 |
| `--mode LISTE` | `residuals`, `merit`, `residuals+extrema`, `merit+extrema`, `all` (alle vier); Grundgerüst ADJOINT (M5-T1): `adjoint-none`, `adjoint-full`, `adjoint-checkpoint`, `adjoint` (alle drei) | residuals,merit |
| `--type LISTE` | Zahlentyp der Kernel: `fp64`, `df64` (M7-T1, `TraceSettings::precision`), `fp32prefilter` (FP32-Vorfilterung, M7-T2; ein Fall pro S und V, unabhängig von `--mode`) | fp64 |
| `--fields F` | Felder (Feldwinkel 0 … 3°) | 3 |
| `--wavelengths L` | Wellenlängen | 3 |
| `--rings N`, `--spokes M` | Pupillenquadratur N × M | 10 × 8 |
| `--repeat R` | gemessene Läufe pro Fall | 10 |
| `--warmup W` | Aufwärmläufe pro Fall | 2 |
| `--no-kernel` | nur über die Engine messen | – |
| `--seed N` | Seed der Störungen (pro S um S verschoben) | 20261018 |
| `--csv DATEI` | Ergebnisse als CSV (`-` = Standardausgabe) | – |
| `--json DATEI` | Ergebnisse als JSON (`-` = Standardausgabe) | – |
| `--quiet` | keine Tabelle | – |

Exit-Codes: 0 Erfolg, 1 Fehler, 2 Aufruffehler, 3 kein CUDA-Gerät (wie `mikroptim trace --gpu`).

**Ausgabe.** Die Tabelle beginnt mit GPU (Name, Architektur, SMs, Speicher, FP32:FP64), der vom Treiber unterstützten und der verwendeten CUDA-Version und dem Build-Typ; danach eine Zeile pro Fall und am Ende die Summe der versagten Strahlen. CSV: eine Zeile pro Fall mit den Spalten `gpu, type, mode, surfaces, variants, rays_per_variant, rays, engine_median_s, engine_min_s, engine_rays_per_s, engine_ray_surfaces_per_s, kernel_median_s, kernel_min_s, kernel_rays_per_s, kernel_ray_surfaces_per_s, failed_rays, tranches` (Kernelspalten leer, wenn nicht gemessen). JSON:

```json
{
  "format": "mikroptim.bench.trace_throughput/1",
  "mikroptim_version": "0.1.0", "build_type": "Release",
  "device": { "index": 0, "name": "…", "compute_capability": "8.6", "multiprocessors": 82,
              "fp32_fp64_ratio": 64, "memory_gib": 24.0,
              "driver_cuda_version": "13.3", "runtime_cuda_version": "12.9" },
  "settings": { "fields": 3, "wavelengths": 3, "rings": 10, "spokes": 8, "repeat": 10,
                "warmup": 2, "seed": 20261018, "block_threads": 128 },
  "results": [ { "type": "fp64", "mode": "merit", "surfaces": 30, "variants": 10000,
                 "rays_per_variant": 510, "rays": 5100000, "failed_rays": 0, "tranches": 1,
                 "engine": { "median_s": …, "min_s": …, "rays_per_s": …, "ray_surfaces_per_s": … },
                 "kernel": { … } oder null } ]
}
```

**Smoke-Test.** `ctest` führt im CUDA-Build `bench_trace_throughput_smoke` aus (S = 10 und 40, V = 10, alle Modi, ein Lauf) und verlangt, dass kein Strahl versagt; ohne GPU wird der Test übersprungen (Exit-Code 3).

**Zahlentyp (M7-T1).** `--type fp64,df64` misst beide Rechentypen der Kernel (`configure_number_type` setzt `TraceSettings::precision`, die Kernelzeit nutzt dieselbe Einstellung in `ResidualRunOptions`/`MeritRunOptions`); Tabelle, CSV und JSON führen den Typ als Spalte. Ergebnisse: [docs/benchmarks.md, „df64“](../docs/benchmarks.md#df64-m7-t1).

**FP32-Vorfilterung (M7-T2).** `--type fp32prefilter` misst `TraceEngine::prefilter_variants` (Engine: Prüfung, Upload, Kernel, Download der Rohdaten, Entscheidung auf dem Host mit Standardkriterien) und `DevicePrefilter::run` (Kernel) einmal pro S und V; die Modus-Spalte heißt `prefilter`, versagte Strahlen sind sicher und unsicher versagte. Das ist kein Merit-Pfad: Die Vorfilterung liefert nur Schranken für die Auswahl im Multistart. Ergebnisse: [docs/benchmarks.md, „FP32-Vorfilterung“](../docs/benchmarks.md#fp32-vorfilterung-m7-t2).
