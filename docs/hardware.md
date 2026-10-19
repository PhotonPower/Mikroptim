# Hardware und CUDA

## Zielhardware

| | RTX 3090 | Tesla P100 |
|---|---|---|
| Architektur | Ampere, sm_86 | Pascal, sm_60 |
| Speicher | 24 GB GDDR6X | 16 GB HBM2 |
| FP32 : FP64 | 64 : 1 (ca. 0,55 TFLOPS FP64) | 2 : 1 (ca. 4,7–5,3 TFLOPS FP64) |
| Rolle | Entwicklung, Debugging, FP32-Vorfilterung, Produktionsläufe mit df64 (M7-T1) | FP64-Referenz und Produktionsläufe |

**df64 auf der RTX 3090 (M7-T1).** Mit `TraceSettings::precision = Precision::Df64` rechnen die Kernel mit Paaren aus zwei float ([numerik.md, „Zahlentyp df64“](numerik.md#zahlentyp-df64-m7-t1)): ca. 48 Bit Genauigkeit bei FP32-Rate, OPD und Residuen weichen von FP64 typisch um einige 10⁻¹¹ mm ab (höchstens 10⁻⁹ mm, Abnahme M7-T1). Der Kernel ist damit auf der RTX 3090 etwa 4-mal so schnell wie in FP64 (S = 30: 3,5·10⁸ statt 8,3·10⁷ Strahlen/s, [benchmarks.md](benchmarks.md#df64-m7-t1); seit den Tupelpaketen von M7-T4 4,9·10⁸ statt 1,1·10⁸, [benchmarks.md](benchmarks.md#profiling-und-tuning-m7-t4)). Die Tesla P100 rechnet FP64 mit halber FP32-Rate; dort bringt df64 (ca. 10–20 FP32-Befehle pro Operation) nichts, sie bleibt FP64-Referenz. Ergebnisse aus df64 und FP64 sind nicht bitgleich; Läufe, die bitgleich reproduzierbar sein müssen (ADR-0007, Multi-GPU M7-T3), verwenden auf allen Geräten denselben Zahlentyp.

## CUDA-Version

CUDA **12.x**, empfohlen 12.9. CUDA 13 hat die Unterstützung für Maxwell, Pascal und Volta entfernt (ADR-0002). Nach dem Kenntnisstand bei Projektstart ist die Treiberserie R580 die letzte mit Pascal-Unterstützung; vor Treiber-Updates auf dem P100-Rechner prüfen.

Architekturen: `CMAKE_CUDA_ARCHITECTURES = 60-real;86-real;86-virtual`.

**Mehrere Toolkits unter Windows:** Liegt ein CUDA 13 im `PATH`, bricht CMake (richtigerweise) ab. Mit dem Visual-Studio-Generator wählt die Toolset-Option das Toolkit, z. B.

```powershell
cmake --preset release-cuda -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
```

nvcc 12.9 warnt bei sm_60, dass die Offline-Kompilierung für Architekturen unter sm_75 künftig entfällt; das ist erwartet (ADR-0002) und kein Fehler.

## Regeln für Kernel-Code

- Warp-Primitive nur als `_sync`-Varianten.
- Shared Memory pro Block ≤ 48 KB.
- Keine Tensor Cores, keine Features oberhalb sm_60 ohne Fallback.
- Batchgrößen aus `cudaMemGetInfo` ableiten: `mko::cuda::device_memory_info()` und `max_variants_on_device()` mit Sicherheitsreserve (Formel in [architektur.md, „Speicherbedarf“](architektur.md#speicherbedarf)); auf der P100 (16 GB) passen deutlich weniger Varianten als auf der RTX 3090 (24 GB).

## Belegung und Blockabbildung (M7-T4)

Die Trace-Kernel `RESIDUALS` und `MERIT` laufen mit 128 Threads pro Block; seit M7-T4 rechnet ein Block ein Paket aufeinanderfolgender Tupel (f, l) einer Variante mit bis zu 512 Strahlen ([architektur.md, „Kernel-Abbildung“](architektur.md#kernel-abbildung)), Kolben und Summen in einer Reihenfolge, die von Blockgröße und Paketen nicht abhängt. Die Belegung begrenzen die Register (nicht Shared Memory, nicht Blockzahl):

| Register pro Thread | sm_86 (64 K Register, max. 48 Warps) | sm_60 (64 K Register, max. 64 Warps) |
|---|---|---|
| ≤ 96 | 5 Blöcke = 20 Warps (42 %) | 5 Blöcke = 20 Warps (31 %) |
| 97 … 128 | 4 Blöcke = 16 Warps (33 %) | 4 Blöcke = 16 Warps (25 %) |

Auf der RTX 3090 begrenzt der FP64-Durchsatz (2 Ergebnisse pro Takt und SM), nicht die Belegung: Eine Registergrenze von 96 (`__launch_bounds__(128, 5)`) brachte dort weder in FP64 noch in df64 einen messbaren Gewinn (±2 %, [benchmarks.md](benchmarks.md#profiling-und-tuning-m7-t4)) und ist deshalb nicht voreingestellt. Auf der P100 (32 FP64-Ergebnisse pro Takt und SM) kann die Belegung begrenzen; zum Messen lässt sich die Grenze ohne Codeänderung setzen: `-DCMAKE_CUDA_FLAGS="-DMKO_KERNEL_MIN_BLOCKS=5"` beim Konfigurieren (`src/cuda/wavefront_block.hpp`; ohne den Schalter steht nur die Blockgröße im `__launch_bounds__`, denn schon ein ausdrückliches Minimum von 1 ändert die Registerzuteilung). Die Rechnung im PTX ändert sich dadurch nicht, nur die Registerzuteilung in ptxas; ob die Bits gleich bleiben, zeigt danach `MultiTraceEngine.ResultFingerprint` (Werte wie im Abschnitt „Mehrere GPUs“). Erst nach einer Messung auf der P100 als Standard übernehmen.

## Profiling mit Nsight Compute

Nsight Compute (`ncu`) braucht Zugriff auf die Leistungszähler der GPU. Ohne ihn bricht jede Messung mit `ERR_NVGPUCTRPERM` ab (so am Arbeitsplatz-PC in M2-T6 und M7-T4). Freischalten (einmalig, mit Administratorrechten durch den Maintainer):

- **Windows:** NVIDIA-Systemsteuerung → Menü „Desktop“ → „Entwicklereinstellungen aktivieren“; dann „Entwickler“ → „GPU-Leistungsindikatoren verwalten“ → „Allen Benutzern Zugriff auf die GPU-Leistungsindikatoren erlauben“.
- **Linux (P100-Rechner):** Datei `/etc/modprobe.d/nvidia-profiling.conf` mit `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, danach Treibermodul neu laden bzw. neu starten (`cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` zeigt `0`).

Sinnvolle Messungen danach (Release-Build, ein Lauf genügt):

```bash
# Übersicht: erreichte Belegung, FP64-Pipeline, Speicher, Warp-Zustände der Trace-Kernel
ncu --kernel-name regex:"merit_kernel|residuals_kernel" --launch-skip 3 --launch-count 2 \
    --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section ComputeWorkloadAnalysis --section WarpStateStats --section MemoryWorkloadAnalysis \
    -o m7t4_profil ./build/release-cuda/bench/trace_throughput --surfaces 30 --variants 10000 \
    --mode merit,merit+extrema --type fp64,df64 --repeat 1 --warmup 1 --no-kernel
```

Zu prüfen: Auslastung der FP64-Pipeline (`sm__inst_executed_pipe_fp64`, auf der RTX 3090 erwartet nahe 100 % in FP64), erreichte gegen theoretische Belegung, Stall-Gründe (`smsp__pcsamp_warps_issue_stalled_*`: auf der P100 vermutlich Latenz der FP64-Befehle, dann `MKO_KERNEL_MIN_BLOCKS=5` messen), Local-Memory-Verkehr der Max-Reduktionen (`l1tex__t_bytes_pipe_lsu_mem_local`) und Lane-Ausnutzung (`smsp__thread_inst_executed_per_inst_executed`, nach M7-T4 erwartet nahe 32).

## Speicherreserve und Tranchen

Programme tracen über `mko::cuda::TraceEngine` ([architektur.md, „Host-API TraceEngine“](architektur.md#host-api-traceengine-m2-t4)). Die Engine teilt einen Batch, der nicht in den Gerätespeicher passt, selbst in Tranchen; die Ergebnisse sind davon unabhängig bitgleich.

- **Reserve:** Vom freien Speicher bleibt max(`min_bytes`, `fraction` · frei) unbenutzt, Standard max(256 MiB, 5 %) (`MemoryReserve`, `EngineOptions::reserve`). Sie deckt den CUDA-Kontext, Kernel-Stacks, Local Memory (z. B. `TraceRecord` ab M4/M5) und Fragmentierung. Auf der RTX 3090 (gemessen ca. 22,8 GiB frei am Arbeitsplatz-PC mit Desktop) sind das ca. 1,1 GiB, auf der P100 rechnerisch höchstens ca. 0,8 GiB (5 % von 16 GB; dort nicht gemessen). Die Max-Reduktionen (M2-T3) belegen ca. 2 KB Local Memory pro Thread, also eine Laufzeitreserve von ca. 350 MB (RTX 3090) bzw. 240 MB (P100), die in der Standardreserve enthalten ist. Die Aufzeichnung des Vorwärtsdurchlaufs im Modus ADJOINT (M5-T1) belegt je nach Kapazitätsklasse 0,9–3,3 KB Local Memory pro Thread (vollständig, C = 16 … 64) bzw. 0,8–1,4 KB (Checkpoints), also höchstens ca. 414 MB (RTX 3090) bzw. 373 MB (P100); auch das liegt in der Standardreserve ([architektur.md, „Modus ADJOINT“](architektur.md#modus-adjoint-vorwärtsdurchlauf-mit-aufzeichnung-m5-t1)). Der Rückwärtsdurchlauf (M5-T2) kommt mit seinen Gradienten pro Strahl hinzu; dann ist die Reserve erneut zu prüfen.
- **Teilen einer GPU:** `EngineOptions::memory_limit_bytes` begrenzt den Speicher, den die Engine als verfügbar ansieht (z. B. wenn andere Programme dieselbe GPU nutzen oder zwei Engines auf einer GPU laufen). Die Engine fragt den freien Speicher bei jedem Lauf neu ab; Speicher, den andere Programme zwischen Abfrage und `cudaMalloc` belegen, führt zu einem `CudaError` (kein automatischer Neuversuch).
- **Größenordnung:** `testsystem_mehrlinser.json` (S = 8, 680 Strahlen) passt mit ca. 3,5 Mio. Varianten pro Tranche auf die RTX 3090 (Test `TraceEngine.MaxVariantsFromFreeMemory`); Tranchen sind also erst bei großen Jacobi-Batches oder dichten Quadraturen zu erwarten.

## Mehrere GPUs (M7-T3)

`mko::cuda::MultiTraceEngine` verteilt einen Batch auf mehrere Geräte im selben Rechner (Standard: alle sichtbaren ab sm_60, also RTX 3090 und P100), je eine `TraceEngine` und ein Host-Thread pro Gerät, Anteile nach gemessenem Durchsatz pro Art und Zahlentyp ([architektur.md, „Mehrere GPUs (M7-T3)“](architektur.md#mehrere-gpus-m7-t3)). Hinweise für den Betrieb:

- **Zahlentyp:** Alle Geräte rechnen denselben Zahlentyp (`TraceSettings::precision`). Für gemischte Läufe ist FP64 die natürliche Wahl (P100 schnell, RTX 3090 langsam, die Kalibrierung gibt der P100 den größeren Anteil); df64 lohnt nur auf der RTX 3090. Läufe in df64 und FP64 sind nicht bitgleich.
- **Architekturen:** Bitgleichheit „unabhängig von der Verteilung“ setzt voraus, dass sm_60 und sm_86 dieselben Bits liefern. Nach der Analyse von PTX und SASS (gleiche Gleitkommabefehle im PTX, korrekt gerundete Division und Wurzel, gleiche Zahl kontrahierter FMA auf beiden Architekturen, JIT aus `compute_60`-PTX auf der 3090 bitgleich, [architektur.md](architektur.md#gemischte-architekturen-sm_60-und-sm_86)) ist das zu erwarten, aber erst auf der P100 nachgewiesen, wenn diese Tests dort grün sind:
  ```powershell
  ctest --preset release-cuda -R "MultiTraceEngine\." --output-on-failure
  .\build\release-cuda\tests\Release\mikroptim_tests.exe --gtest_filter=MultiTraceEngine.ResultFingerprint
  ```
  `MultiTraceEngine.RealDevicesBitIdenticalToSingle` vergleicht jedes Gerät mit Gerät 0 und die Verteilung mit dem Einzellauf (wird mit nur einem Gerät übersprungen); `ResultFingerprint` druckt pro Gerät Prüfsummen der Ergebnisbits, die sich auch zwischen getrennten Rechnern vergleichen lassen. Referenz auf der RTX 3090 (nvcc 12.9, Treiber am Arbeitsplatz-PC, neu erhoben mit M5-T1: main ce79590 und M5-T1 liefern dieselben Werte, der Umbau des Trace in M5-T1 ändert RESIDUALS und MERIT also nicht; die Werte von M7-T3 galten auf main schon vorher nicht mehr, vermutlich seit dem Strafresiduen-Block von M4-T2): FP64 RESIDUALS `4c25f7bd60a12b22`, MERIT `8c62727b084a89e7`; DF64 RESIDUALS `b8f3b1c0589fdd36`, MERIT `128ba8f7c507e722`; PREFILTER `198fcadd375e938f`. Die Werte ändern sich mit jeder Änderung an Kerneln oder Testsystem, die die Rechnung berührt; die Tupelpakete von M7-T4 lassen sie unverändert (gleiche Summationsreihenfolge, auf der RTX 3090 nachgeprüft).
- **Eine GPU, mehrere Slots:** Für Tests darf ein Gerät mehrfach in `MultiEngineOptions::devices` stehen; die Slots teilen sich dann Speicher (je 1/k) und Rechenzeit und sind nicht schneller.
- **Speicher:** Jede Engine richtet ihre Tranchen nach dem freien Speicher ihres Geräts (P100: 16 GB, RTX 3090: 24 GB); `slot_engines` setzt Grenzen pro Gerät.

## Durchsatzabschätzung

Annahme: ca. 140 Flop-Äquivalente pro Strahl und Fläche (inklusive zwei Wurzeln und zwei Divisionen in FP64), 30 Flächen, also ca. 4.500 pro Strahl.

| | 3090 FP64 | 3090 df64 | P100 FP64 |
|---|---|---|---|
| Strahlen/s (realistisch) | ca. 5·10⁷ | ca. 2–3·10⁸ (unsicher) | ca. 4–5·10⁸ |
| Jacobi per finite Differenzen (225.000 Strahlen) | ca. 5 ms | ca. 1 ms | ca. 0,5 ms |
| LM-Iteration für 1.000 Läufe (2·10⁸ Strahlen) | ca. 4 s | ca. 1 s | ca. 0,5 s |

Diese Werte sind Schätzungen und werden in M2 durch Messungen in [benchmarks.md](benchmarks.md) ersetzt. Weicht eine Messung um mehr als den Faktor 3 ab, wird die Ursache im PR erklärt.

**Messung (M2-T6, `bench/trace_throughput`, [benchmarks.md](benchmarks.md)):** Auf der RTX 3090 in FP64 liefert der Kernel bei S = 30 und 10.000 Varianten 8,4·10⁷ Strahlen/s (über die TraceEngine mit Upload und Download 7,0–7,7·10⁷), die Jacobi mit 225.000 Strahlen dauert 3,2 ms (Engine 3,5–3,8 ms), 2·10⁸ Strahlen 2,4 s (Engine 2,6–2,7 s). Die Schätzung ist also um den Faktor 1,3–1,7 pessimistisch; der Kernel ist auf der RTX 3090 durch den FP64-Durchsatz begrenzt (ca. 2,5·10⁹ Strahl-Flächen/s unabhängig von S). Die P100-Messung steht noch aus.

**Messung df64 (M7-T1, `trace_throughput --type fp64,df64`):** RTX 3090, S = 30, 10.000 Varianten: Kernel 3,5·10⁸ Strahlen/s (≈ 1,04·10¹⁰ Strahl-Flächen/s), über die Engine 2,5·10⁸ (MERIT) bzw. 1,9·10⁸ (RESIDUALS); Jacobi mit 225.000 Strahlen 0,76 ms (Engine 1,6–1,8 ms), 2·10⁸ Strahlen 0,58 s (Engine 0,94–1,35 s). Das trifft die Schätzung (2–3·10⁸ Strahlen/s, ca. 1 ms, ca. 1 s) im Kernel um den Faktor 1,2–1,7 besser; über die Engine ist der Upload des Batches im Verhältnis teurer als in FP64 (Messreihe in [benchmarks.md](benchmarks.md#df64-m7-t1)).
