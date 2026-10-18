# Hardware und CUDA

## Zielhardware

| | RTX 3090 | Tesla P100 |
|---|---|---|
| Architektur | Ampere, sm_86 | Pascal, sm_60 |
| Speicher | 24 GB GDDR6X | 16 GB HBM2 |
| FP32 : FP64 | 64 : 1 (ca. 0,55 TFLOPS FP64) | 2 : 1 (ca. 4,7–5,3 TFLOPS FP64) |
| Rolle | Entwicklung, Debugging, FP32-Vorfilterung, später df64 | FP64-Referenz und Produktionsläufe |

## CUDA-Version

CUDA **12.x**, empfohlen 12.9. CUDA 13 hat die Unterstützung für Maxwell, Pascal und Volta entfernt (ADR-0002). Nach dem Kenntnisstand bei Projektstart ist die Treiberserie R580 die letzte mit Pascal-Unterstützung; vor Treiber-Updates auf dem P100-Rechner prüfen.

Architekturen: `CMAKE_CUDA_ARCHITECTURES = 60-real;86-real;86-virtual`.

## Regeln für Kernel-Code

- Warp-Primitive nur als `_sync`-Varianten.
- Shared Memory pro Block ≤ 48 KB.
- Keine Tensor Cores, keine Features oberhalb sm_60 ohne Fallback.
- Batchgrößen aus `cudaMemGetInfo` ableiten: `mko::cuda::device_memory_info()` und `max_variants_on_device()` mit Sicherheitsreserve (Formel in [architektur.md, „Speicherbedarf“](architektur.md#speicherbedarf)); auf der P100 (16 GB) passen deutlich weniger Varianten als auf der RTX 3090 (24 GB).

## Durchsatzabschätzung

Annahme: ca. 140 Flop-Äquivalente pro Strahl und Fläche (inklusive zwei Wurzeln und zwei Divisionen in FP64), 30 Flächen, also ca. 4.500 pro Strahl.

| | 3090 FP64 | 3090 df64 | P100 FP64 |
|---|---|---|---|
| Strahlen/s (realistisch) | ca. 5·10⁷ | ca. 2–3·10⁸ (unsicher) | ca. 4–5·10⁸ |
| Jacobi per finite Differenzen (225.000 Strahlen) | ca. 5 ms | ca. 1 ms | ca. 0,5 ms |
| LM-Iteration für 1.000 Läufe (2·10⁸ Strahlen) | ca. 4 s | ca. 1 s | ca. 0,5 s |

Diese Werte sind Schätzungen und werden in M2 durch Messungen in [benchmarks.md](benchmarks.md) ersetzt. Weicht eine Messung um mehr als den Faktor 3 ab, wird die Ursache im PR erklärt.
