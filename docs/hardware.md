# Hardware und CUDA

## Zielhardware

| | RTX 3090 | Tesla P100 |
|---|---|---|
| Architektur | Ampere, sm_86 | Pascal, sm_60 |
| Speicher | 24 GB GDDR6X | 16 GB HBM2 |
| FP32 : FP64 | 64 : 1 (ca. 0,55 TFLOPS FP64) | 2 : 1 (ca. 4,7–5,3 TFLOPS FP64) |
| Rolle | Entwicklung, Debugging, FP32-Vorfilterung, Produktionsläufe mit df64 (M7-T1) | FP64-Referenz und Produktionsläufe |

**df64 auf der RTX 3090 (M7-T1).** Mit `TraceSettings::precision = Precision::Df64` rechnen die Kernel mit Paaren aus zwei float ([numerik.md, „Zahlentyp df64“](numerik.md#zahlentyp-df64-m7-t1)): ca. 48 Bit Genauigkeit bei FP32-Rate, OPD und Residuen weichen von FP64 typisch um einige 10⁻¹¹ mm ab (höchstens 10⁻⁹ mm, Abnahme M7-T1). Der Kernel ist damit auf der RTX 3090 etwa 4-mal so schnell wie in FP64 (S = 30: 3,5·10⁸ statt 8,3·10⁷ Strahlen/s, [benchmarks.md](benchmarks.md#df64-m7-t1)). Die Tesla P100 rechnet FP64 mit halber FP32-Rate; dort bringt df64 (ca. 10–20 FP32-Befehle pro Operation) nichts, sie bleibt FP64-Referenz. Ergebnisse aus df64 und FP64 sind nicht bitgleich; Läufe, die bitgleich reproduzierbar sein müssen (ADR-0007, Multi-GPU M7-T3), verwenden auf allen Geräten denselben Zahlentyp.

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

## Speicherreserve und Tranchen

Programme tracen über `mko::cuda::TraceEngine` ([architektur.md, „Host-API TraceEngine“](architektur.md#host-api-traceengine-m2-t4)). Die Engine teilt einen Batch, der nicht in den Gerätespeicher passt, selbst in Tranchen; die Ergebnisse sind davon unabhängig bitgleich.

- **Reserve:** Vom freien Speicher bleibt max(`min_bytes`, `fraction` · frei) unbenutzt, Standard max(256 MiB, 5 %) (`MemoryReserve`, `EngineOptions::reserve`). Sie deckt den CUDA-Kontext, Kernel-Stacks, Local Memory (z. B. `TraceRecord` ab M4/M5) und Fragmentierung. Auf der RTX 3090 (gemessen ca. 22,8 GiB frei am Arbeitsplatz-PC mit Desktop) sind das ca. 1,1 GiB, auf der P100 rechnerisch höchstens ca. 0,8 GiB (5 % von 16 GB; dort nicht gemessen). Die Max-Reduktionen (M2-T3) belegen ca. 2 KB Local Memory pro Thread, also eine Laufzeitreserve von ca. 350 MB (RTX 3090) bzw. 240 MB (P100), die in der Standardreserve enthalten ist. Wird in M4/M5 mehr Local Memory pro Strahl gebraucht, ist die Reserve zu prüfen.
- **Teilen einer GPU:** `EngineOptions::memory_limit_bytes` begrenzt den Speicher, den die Engine als verfügbar ansieht (z. B. wenn andere Programme dieselbe GPU nutzen oder zwei Engines auf einer GPU laufen). Die Engine fragt den freien Speicher bei jedem Lauf neu ab; Speicher, den andere Programme zwischen Abfrage und `cudaMalloc` belegen, führt zu einem `CudaError` (kein automatischer Neuversuch).
- **Größenordnung:** `testsystem_mehrlinser.json` (S = 8, 680 Strahlen) passt mit ca. 3,5 Mio. Varianten pro Tranche auf die RTX 3090 (Test `TraceEngine.MaxVariantsFromFreeMemory`); Tranchen sind also erst bei großen Jacobi-Batches oder dichten Quadraturen zu erwarten.

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
