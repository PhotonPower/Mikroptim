# ADR-0010: Normalgleichungen mit eigenem Kernel, Cholesky auf dem Host

- **Status:** vorgeschlagen
- **Datum:** 2026-10-19

## Kontext

Der Plan (M3-T4, optimierung.md §4) sieht für den gebündelten Levenberg-Marquardt JᵀJ und Jᵀr per `cublasDgemmStridedBatched` und die Lösung per `cusolverDnDpotrfBatched`/`potrsBatched` vor. Abnahme 3 und ADR-0007 verlangen, dass ein Lauf in einem Bündel von 1.000 Läufen bitgleich dasselbe Ergebnis liefert wie allein. Die Zahl der aktiven Läufe ändert sich während der Optimierung (fertige Läufe scheiden aus, am Ende oft nur noch einer).

cuBLAS garantiert Bitgleichheit nur für identische Aufrufe (gleiche Größen, gleiche Architektur und SM-Zahl), nicht für dieselbe Matrix in Aufrufen mit verschiedenem `batchCount`. Messung auf der RTX 3090 (CUDA 12.9, `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)`, Pointer-Mode Host, alle Matrizen im Batch gleich): Anteil der Elemente von JᵀJ, die sich zwischen `batchCount` = 1 und `batchCount` ≥ 2 unterscheiden:

| R × P | abweichend |
|---|---|
| 11 × 4 | 5 von 16 |
| 683 × 14 | 0 von 196 |
| 5.000 × 14 | 194 von 196 |
| 50.000 × 30 | 890 von 900 |
| 200.000 × 14 | 196 von 196 |
| 20.000 × 101 | 10.069 von 10.201 |

Zwischen `batchCount` = 2, 7, 100 und 1.000 und zwischen den Positionen im Batch gab es keine Unterschiede; das ist aber nicht dokumentiert und kann sich mit Problemgröße, Toolkit oder GPU (P100) ändern. Für `potrfBatched` gilt dasselbe Argument. Zusätzlich braucht die aktive Menge pro Lauf ein anderes reduziertes System.

## Entscheidung

- **JᵀJ und Jᵀr** rechnet ein eigener Kernel (`src/cuda/normal_equations.cu`): Gram-Matrix G = [J r₀]ᵀ[J r₀] pro Lauf, ein Block mit 16 × 16 Threads pro Lauf und Kachelpaar, jedes Element eine FMA-Kette über die Zeilen in aufsteigender Reihenfolge. Das Ergebnis eines Elements hängt nur von zwei Spalten eines Laufs ab, nicht von Zahl oder Lage der Läufe, und ist bitgleich zur Host-Funktion `normal_equations_host` (`std::fma`, gleiche Reihenfolge).
- **Cholesky und aktive Menge** laufen auf dem Host, pro Lauf unabhängig auf Host-Threads, mit demselben Code wie der CPU-Pfad. Heruntergeladen werden nur A und g ((P + 1)² Werte pro Lauf).
- cuBLAS/cuSOLVER werden nicht gelinkt.

## Konsequenzen

- Bitgleichheit Gruppe gegen Einzellauf ist konstruktiv gesichert (Test: 1.000 Läufe, auch mit erzwungenen Teilstücken und Tranchen).
- CPU- und GPU-Pfad teilen die gesamte Steuerlogik einschließlich der Lösung; Unterschiede entstehen nur aus J und F.
- Kosten: Der Kernel nutzt FP64-FMA ohne Tensor Cores (ohnehin ausgeschlossen); für P = 100, R = 5·10⁴ und 1.000 Läufe etwa 2,5·10¹¹ FMA pro Jacobi-Rechnung (≈ 1 s auf der RTX 3090), klein gegen die FD-Jacobi derselben Größe. Die Cholesky-Zerlegung kostet ≈ P³/6 FMA pro Lauf (P = 100, 1.000 Läufe: ≈ 1,7·10⁸, einige 10 ms auf mehreren Host-Threads). Falls das bei M5 (adjungierte Jacobi, deutlich billiger) relevant wird, kann ein deterministischer Cholesky-Kernel (ein Block pro Lauf, P ≤ ~128) folgen.
- Frage an den Maintainer: Annahme dieser Abweichung vom Plan, oder cuBLAS mit festem `batchCount` (z. B. immer ≥ 2 durch Auffüllen) trotz fehlender Garantie?

## Verworfene Alternativen

- `cublasDgemmStridedBatched` mit festen Optionen: siehe Messung, keine Garantie über `batchCount`.
- Immer mit konstantem `batchCount` aufrufen (Auffüllen mit Kopien): verlässt sich auf undokumentiertes Verhalten und verschwendet Rechenzeit.
- JᵀJ per QR von J statt Normalgleichungen (bessere Kondition): größerer Umbau, kein Teil des Plans; wäre bei sehr unterschiedlich gewichteten Residuen eine Option (optimierung.md §4, „Kondition und Gewichte“).
