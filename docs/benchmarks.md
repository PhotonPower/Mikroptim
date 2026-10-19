# Benchmarks

Gemessener Durchsatz der Trace-Kernel. Messprogramm: `bench/trace_throughput` (M2-T6, Optionen und Ausgabeformate in [bench/README.md](../bench/README.md)). Die Schätzungen, mit denen die Messungen verglichen werden, stehen in [hardware.md, „Durchsatzabschätzung“](hardware.md#durchsatzabschätzung).

## Übersicht

Eine Zeile pro Messreihe: S = 30 Flächen, V = 10.000 Varianten, Standardquadratur 10 × 8, 3 Felder × 3 Wellenlängen (510 Strahlen pro Variante), Modus MERIT über die TraceEngine (inklusive Upload und Download); in Klammern der Kernel allein.

| Datum | Commit | GPU | Treiber / CUDA | Typ | Flächen | Varianten × Strahlen | Strahlen/s |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | 33356d9 + M2-T6 | RTX 3090 | 610.88 / 12.9 | FP64 | 30 | 10.000 × 510 | 7,7·10⁷ (Kernel 8,4·10⁷) |
| 2026-10-18 | a210b89 + M7-T1 | RTX 3090 | 610.88 / 12.9 | df64 | 30 | 10.000 × 510 | 2,5·10⁸ (Kernel 3,5·10⁸) |
| 2026-10-18 | 2297b58 + M7-T2 | RTX 3090 | 610.88 / 12.9 | FP32-Vorfilterung (kein Merit-Pfad) | 30 | 10.000 × 510 | 4,7·10⁸ (Kernel 1,7·10⁹) |
| 2026-10-19 | 59c2408 + M7-T4 | RTX 3090 | 610.88 / 12.9 | FP64 | 30 | 10.000 × 510 | 1,0·10⁸ (Kernel 1,1·10⁸), Minimum statt Median |
| 2026-10-19 | 59c2408 + M7-T4 | RTX 3090 | 610.88 / 12.9 | df64 | 30 | 10.000 × 510 | 3,8·10⁸ (Kernel 4,9·10⁸), Minimum statt Median |
| ausstehend | | Tesla P100 | | FP64 | 30 | 10.000 × 510 | ausstehend (Befehl unten) |

## Messprogramm in Kürze

- **Systeme:** frei erfundene Folgen schwacher Menisken (Biegung ±1,5) aus einem Kron- und einem Flintglas (Buchdahl-Modelle), bei ungeradem S − 1 mit einem verkitteten Glied am Ende; Blende 2 mm Radius, Summe der dünnen Brechkräfte 1/40 mm⁻¹, Bildebene im paraxialen Fokus bei 587,56 nm. Felder 0 … 3°, Wellenlängen 486,13 … 656,27 nm. Kein reales Objektiv.
- **Varianten:** Variante 0 ungestört, alle anderen mit festem Seed leicht gestört (Krümmungen und Dicken ±2 %, Arbeitsabstand ±1 %, Brechzahlen ±0,002, Pupillenradius ±5 %). In allen Messungen ist **kein einziger Strahl versagt** (90,7 Mio. Strahlen der Hauptreihe, 400 Mio. der Zusatzreihe), die Zeiten enthalten also keine maskierten Strahlen.
- **Engine:** Zeit für `TraceEngine::run` (Prüfung der Werte auf dem Host, Upload des Batches, Kernel, Download der Ergebnisse; RESIDUALS mit Status pro Strahl, ohne OPD), synchron, mit der Host-Uhr.
- **Kernel:** Zeit für `DeviceResiduals::run` bzw. `DeviceMerit::run` auf einem schon hochgeladenen `DeviceSystemBatch` (Feldablage hochladen, Kernel bzw. beide Stufen von MERIT, `cudaDeviceSynchronize`), ohne Upload des Batches und ohne Download. Die Engine bietet keine CUDA-Events an; gemessen wird mit der Host-Uhr um den synchronen Aufruf, für kleine V dominiert daher die Start- und Synchronisationslatenz (ca. 50 µs).
- Median aus 20 Läufen nach 3 Aufwärmläufen. Strahlen/s = V · 510 / Zeit, Strahl-Flächen/s = Strahlen/s · S.

## Messumgebung RTX 3090

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 24 GiB), Arbeitsplatz-PC mit Desktop |
| Takt unter Last | ca. 1.940 MHz (nvidia-smi während eines Laufs, P2, ca. 200 W) |
| Treiber | 610.88 (unterstützt CUDA 13.3) |
| CUDA | Toolkit und Laufzeit 12.9, `CMAKE_CUDA_ARCHITECTURES = 60-real;86-real;86-virtual` |
| Build | Preset `release-cuda`, MSVC 14.44 (Visual Studio 2022 Build Tools), Release, Windows 11 Pro |
| Datum, Stand | 2026-10-18, main 33356d9 (Bibliotheken unverändert durch M2-T6) |
| Befehl | `trace_throughput --mode all --repeat 20 --warmup 3 --csv … --json …` |

## Ergebnisse RTX 3090 (FP64)

Alle 4 × 5 Fälle der Standardreihe, Modi RESIDUALS und MERIT. Der Kernel RESIDUALS ist so schnell wie MERIT (Abweichung ≤ 2 %, siehe CSV/JSON), deshalb hier nur eine Kernelspalte.

| S | V | Strahlen | RESIDUALS Engine [ms] | RESIDUALS Engine [Str/s] | MERIT Engine [ms] | MERIT Engine [Str/s] | Kernel MERIT [ms] | Kernel MERIT [Str/s] | Kernel MERIT [Strahl-Flächen/s] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 510 | 0,212 | 2,40·10⁶ | 0,229 | 2,23·10⁶ | 0,060 | 8,55·10⁶ | 8,55·10⁷ |
| 10 | 10 | 5.100 | 0,582 | 8,77·10⁶ | 0,248 | 2,06·10⁷ | 0,085 | 5,97·10⁷ | 5,97·10⁸ |
| 10 | 100 | 51.000 | 0,593 | 8,60·10⁷ | 0,532 | 9,59·10⁷ | 0,336 | 1,52·10⁸ | 1,52·10⁹ |
| 10 | 1.000 | 510.000 | 3,467 | 1,47·10⁸ | 2,644 | 1,93·10⁸ | 2,223 | 2,29·10⁸ | 2,29·10⁹ |
| 10 | 10.000 | 5.100.000 | 31,43 | 1,62·10⁸ | 24,53 | 2,08·10⁸ | 22,56 | 2,26·10⁸ | 2,26·10⁹ |
| 20 | 1 | 510 | 0,213 | 2,40·10⁶ | 0,215 | 2,38·10⁶ | 0,075 | 6,77·10⁶ | 1,35·10⁸ |
| 20 | 10 | 5.100 | 0,265 | 1,92·10⁷ | 0,264 | 1,93·10⁷ | 0,125 | 4,08·10⁷ | 8,15·10⁸ |
| 20 | 100 | 51.000 | 0,772 | 6,61·10⁷ | 0,762 | 6,69·10⁷ | 0,572 | 8,92·10⁷ | 1,78·10⁹ |
| 20 | 1.000 | 510.000 | 5,354 | 9,53·10⁷ | 4,748 | 1,07·10⁸ | 4,223 | 1,21·10⁸ | 2,42·10⁹ |
| 20 | 10.000 | 5.100.000 | 52,67 | 9,68·10⁷ | 46,01 | 1,11·10⁸ | 41,65 | 1,22·10⁸ | 2,45·10⁹ |
| 30 | 1 | 510 | 0,303 | 1,68·10⁶ | 0,243 | 2,10·10⁶ | 0,105 | 4,87·10⁶ | 1,46·10⁸ |
| 30 | 10 | 5.100 | 0,360 | 1,42·10⁷ | 0,312 | 1,63·10⁷ | 0,212 | 2,41·10⁷ | 7,23·10⁸ |
| 30 | 100 | 51.000 | 1,042 | 4,89·10⁷ | 0,980 | 5,20·10⁷ | 0,765 | 6,66·10⁷ | 2,00·10⁹ |
| 30 | 1.000 | 510.000 | 7,422 | 6,87·10⁷ | 6,812 | 7,49·10⁷ | 6,061 | 8,41·10⁷ | 2,52·10⁹ |
| 30 | 10.000 | 5.100.000 | 72,44 | 7,04·10⁷ | 66,09 | 7,72·10⁷ | 60,50 | 8,43·10⁷ | 2,53·10⁹ |
| 40 | 1 | 510 | 0,588 | 8,68·10⁵ | 0,316 | 1,62·10⁶ | 0,123 | 4,15·10⁶ | 1,66·10⁸ |
| 40 | 10 | 5.100 | 0,408 | 1,25·10⁷ | 0,396 | 1,29·10⁷ | 0,207 | 2,47·10⁷ | 9,86·10⁸ |
| 40 | 100 | 51.000 | 1,299 | 3,93·10⁷ | 1,236 | 4,12·10⁷ | 1,003 | 5,08·10⁷ | 2,03·10⁹ |
| 40 | 1.000 | 510.000 | 9,496 | 5,37·10⁷ | 9,202 | 5,54·10⁷ | 8,051 | 6,33·10⁷ | 2,53·10⁹ |
| 40 | 10.000 | 5.100.000 | 94,66 | 5,39·10⁷ | 85,98 | 5,93·10⁷ | 79,66 | 6,40·10⁷ | 2,56·10⁹ |

Für V ≤ 10 schwanken die Engine-Zeiten von Lauf zu Lauf um bis zu den Faktor 2 (Latenz von `cudaMemGetInfo`, Kopien und Synchronisation, ca. 0,2–0,6 ms); aussagekräftig ist dort nur die Größenordnung.

**Max-Reduktionen** (V = 10.000):

| S | Modus | Engine [ms] | Engine [Str/s] | Kernel [ms] | Kernel [Str/s] |
|---:|---|---:|---:|---:|---:|
| 10 | RESIDUALS | 31,43 | 1,62·10⁸ | 21,81 | 2,34·10⁸ |
| 10 | MERIT | 24,53 | 2,08·10⁸ | 22,56 | 2,26·10⁸ |
| 10 | RESIDUALS + Extrema | 38,57 | 1,32·10⁸ | 29,18 | 1,75·10⁸ |
| 10 | MERIT + Extrema | 32,20 | 1,58·10⁸ | 29,92 | 1,70·10⁸ |
| 20 | RESIDUALS | 52,67 | 9,68·10⁷ | 41,34 | 1,23·10⁸ |
| 20 | MERIT | 46,01 | 1,11·10⁸ | 41,65 | 1,22·10⁸ |
| 20 | RESIDUALS + Extrema | 65,35 | 7,80·10⁷ | 54,77 | 9,31·10⁷ |
| 20 | MERIT + Extrema | 59,97 | 8,50·10⁷ | 55,64 | 9,17·10⁷ |
| 30 | RESIDUALS | 72,44 | 7,04·10⁷ | 60,06 | 8,49·10⁷ |
| 30 | MERIT | 66,09 | 7,72·10⁷ | 60,50 | 8,43·10⁷ |
| 30 | RESIDUALS + Extrema | 104,9 | 4,86·10⁷ | 82,13 | 6,21·10⁷ |
| 30 | MERIT + Extrema | 88,38 | 5,77·10⁷ | 80,96 | 6,30·10⁷ |
| 40 | RESIDUALS | 94,66 | 5,39·10⁷ | 78,87 | 6,47·10⁷ |
| 40 | MERIT | 85,98 | 5,93·10⁷ | 79,66 | 6,40·10⁷ |
| 40 | RESIDUALS + Extrema | 122,6 | 4,16·10⁷ | 107,2 | 4,76·10⁷ |
| 40 | MERIT + Extrema | 116,0 | 4,40·10⁷ | 106,8 | 4,77·10⁷ |

Die Max-Reduktionen kosten im Kernel ca. 33 % (`RaySurfaceRecord` und `SurfaceExtrema` im Local Memory, [architektur.md](architektur.md#kernel-merit-m2-t3)).

**Zusatzreihe zu den Jacobi- und LM-Schätzungen** (S = 30, `--surfaces 30 --variants 441,392157 --mode residuals,merit --repeat 5`):

| Fall | Strahlen | RESIDUALS Engine | MERIT Engine | Kernel MERIT | Kernel [Str/s] |
|---|---:|---:|---:|---:|---:|
| V = 441 (Jacobi, ≈ 225.000 Strahlen) | 224.910 | 3,79 ms | 3,46 ms | 3,16 ms | 7,1·10⁷ |
| V = 392.157 (LM, ≈ 2·10⁸ Strahlen) | 200.000.070 | 2,74 s | 2,60 s | 2,36 s | 8,5·10⁷ |

## Auswertung

**Skalierung mit der Flächenzahl.** Für V ≥ 1.000 ist der Durchsatz in Strahl-Flächen/s nahezu unabhängig von S (2,3–2,6·10⁹). Eine Ausgleichsgerade durch die Kernelzeiten bei V = 10.000 ergibt 1,90 ms pro Fläche und ca. 3 ms fest (RESIDUALS 2,8 ms, MERIT 3,5 ms einschließlich der zweiten Stufe), d. h. die Arbeit pro Strahl außerhalb der Flächenschleife (Startstrahl, Referenzsphäre, Kolben, Residuum, Reduktionen) entspricht etwa 1,5–2 Flächen. Die Strahlen/s fallen also nahezu wie 1/(S + 2).

**Skalierung mit der Variantenzahl.** Jede Variante belegt F · L = 9 Blöcke. Bei 4 residenten Blöcken pro SM (siehe Profilnotiz) passen 328 Blöcke gleichzeitig auf die 82 SMs, also gut 36 Varianten pro Welle. Unter V ≈ 100 ist die GPU nicht ausgelastet (Latenz dominiert), ab V ≈ 1.000 (27 Wellen) ist der Durchsatz gesättigt; V = 10.000 bringt nur noch 0–2 %.

**Engine gegen Kernel.** Bei V = 10.000 kostet die Engine gegenüber dem Kernel allein im Modus MERIT 2–6 ms (Upload des Batches: (S·(2 + L) + 1 + 2F) · 8 Byte pro Variante, bei S = 40 also 16,6 MB, dazu die Prüfung auf dem Host), im Modus RESIDUALS 10–16 ms (zusätzlich Download von Residuen und Status, 4,6 kB pro Variante, 46 MB). Das entspricht ca. 3–6 GB/s aus bzw. in nicht gepinnten Host-Speicher. Gepinnte Puffer und Überlappung über Streams sind Sache von M7.

## Vergleich mit der Schätzung (hardware.md)

| Größe (RTX 3090, FP64) | Schätzung | Messung | Verhältnis |
|---|---|---|---|
| Strahlen/s, S = 30 | ca. 5·10⁷ | 7,0–7,7·10⁷ (Engine), 8,4·10⁷ (Kernel), V = 10.000 | 1,4–1,7 × schneller |
| Jacobi per finite Differenzen (225.000 Strahlen) | ca. 5 ms | 3,5–3,8 ms (Engine), 3,2 ms (Kernel) | 1,3–1,6 × schneller |
| LM-Iteration für 1.000 Läufe (2·10⁸ Strahlen) | ca. 4 s | 2,6–2,7 s (Engine), 2,4 s (Kernel) | 1,5–1,7 × schneller |

**Alle Abweichungen liegen innerhalb des Faktors 3**; die Schätzung ist leicht pessimistisch. Gründe:

- **FP64-Durchsatz:** Die Schätzung rechnet mit 0,55 TFLOPS (Referenz-Boost 1,70 GHz); unter Last lief die Karte mit ca. 1,94 GHz, also 82 SMs · 2 FP64-Ergebnisse pro Takt · 2 Flop · 1,94 GHz ≈ 0,64 TFLOPS (+14 %).
- **Ausnutzung:** 5·10⁷ Strahlen/s · 4.500 Flop entsprechen 0,23 TFLOPS, also ca. 40 % des Spitzenwerts. Gemessen sind 2,5·10⁹ Strahl-Flächen/s; mit den angenommenen 140 Flop pro Strahl und Fläche wären das 0,35 TFLOPS ≈ 55 % der 0,64 TFLOPS. Andersherum: Bei voller Auslastung der FP64-Einheiten stünden ca. 250 Flop-Äquivalente pro Strahl und Fläche zur Verfügung. Der Kernel ist auf der RTX 3090 also klar durch den FP64-Durchsatz begrenzt (Compute Capability 8.6: 2 FP64-Ergebnisse pro Takt und SM, ein FP64-Warp-Befehl braucht also 16 Takte Durchsatz), nicht durch Speicher oder Belegung.
- **Leerlauf-Threads:** Ein Block hat 128 Threads, die Standardquadratur 80 Stützstellen außerhalb der Achse (Warps 0–1 voll, Warp 2 halb, Warp 3 ohne Arbeit) und 10 auf der Achse (ein Warp mit 10 aktiven Lanes). Pro (Variante, Wellenlänge) arbeiten so 170 Strahlen in 7 aktiven Warps (224 Lanes): ca. 76 % Lane-Ausnutzung. Weil ein FP64-Warp-Befehl unabhängig von der aktiven Maske die Einheit belegt, gehen bis zu ca. 24 % des FP64-Durchsatzes verloren; eine dichtere Packung (z. B. 96 Threads pro Block oder mehrere Tupel pro Block) ist ein Kandidat für M7-T4. Die Schätzung berücksichtigt das nicht; es wird durch den höheren Takt und die bessere Ausnutzung mehr als ausgeglichen.
- **Upload/Download:** 7–10 % der Engine-Zeit im Modus MERIT und 17–31 % im Modus RESIDUALS bei V = 10.000 (siehe oben); in der Schätzung nicht enthalten.

## Profilnotiz

Register und Stack aus `cuobjdump --dump-resource-usage build/release-cuda/Release/mikroptim_cuda.lib` (unverändert gegenüber der Tabelle in [architektur.md](architektur.md#kernel-merit-m2-t3)); Belegung daraus berechnet (Registerdatei 64 K pro SM, Zuteilung in Einheiten von 256 Registern pro Warp), weil die Kernel-Symbole außerhalb der Bibliothek nicht für `cudaOccupancyMaxActiveBlocksPerMultiprocessor` erreichbar sind:

| Kernel | sm_86 (RTX 3090) | sm_60 (P100) |
|---|---|---|
| `residuals_kernel<double, false>` | 110 Register → 4 Blöcke/SM = 16 von 48 Warps (33 %) | 118 Register → 4 Blöcke/SM = 16 von 64 Warps (25 %) |
| `merit_kernel<double, false>` | 110 Register → 4 Blöcke/SM (33 %) | 110 Register → 4 Blöcke/SM (25 %) |
| `merit_kernel<double, true>` | 112 Register, Stack 2.120 B → 4 Blöcke/SM (33 %) | 105 Register, Stack 2.120 B → 4 Blöcke/SM (25 %) |

Shared Memory (2–6 KB statisch) begrenzt nicht. Auf der RTX 3090 reichen 16 Warps pro SM, um die FP64-Einheiten zu sättigen. Auf der P100 (32 FP64-Ergebnisse pro Takt und SM) wird die Belegung von 25 % dagegen voraussichtlich zum begrenzenden Faktor (Latenz der FP64-Befehle und der Lade-Operationen schlecht versteckt); das prüft die P100-Messung. Hardwarezähler (Nsight Compute: erreichte Belegung, Auslastung der FP64-Pipeline) waren hier nicht verfügbar (`ERR_NVGPUCTRPERM`, Zähler nur mit Administratorrechten); das gehört zu M7-T4. (Stand M2-T6; aktuelle Register und die Messungen von M7-T4: [„Profiling und Tuning (M7-T4)“](#profiling-und-tuning-m7-t4).)

## df64 (M7-T1)

Rechentyp df64 (Paar aus zwei float, [numerik.md, „Zahlentyp df64“](numerik.md#zahlentyp-df64-m7-t1)) gegen FP64 auf derselben Karte, Messprogramm und Systeme wie oben. Messumgebung wie in „Messumgebung RTX 3090“, Stand main a210b89 mit M7-T1 (Branch `m7/t1-df64`), Datum 2026-10-18. Befehle:

```text
trace_throughput --type fp64,df64 --surfaces 10,30,40 --variants 10000 --mode residuals,merit,merit+extrema --repeat 20 --warmup 3 --csv … --json …
trace_throughput --type fp64,df64 --surfaces 30 --variants 441,392157 --mode residuals,merit --repeat 5
```

Kein Strahl versagt (9,2·10⁷ bzw. 8,0·10⁸ Strahlen). V = 10.000, 510 Strahlen pro Variante, Median aus 20 Läufen:

| S | Modus | Engine FP64 [ms] | Engine df64 [ms] | Faktor | Kernel FP64 [ms] | Kernel df64 [ms] | Faktor | Kernel df64 [Str/s] | Kernel df64 [Strahl-Flächen/s] |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | RESIDUALS | 37,85 | 16,14 | 2,3 | 22,46 | 5,71 | 3,9 | 8,93·10⁸ | 8,9·10⁹ |
| 10 | MERIT | 25,30 | 8,06 | 3,1 | 22,73 | 6,16 | 3,7 | 8,29·10⁸ | 8,3·10⁹ |
| 10 | MERIT + Extrema | 32,36 | 12,38 | 2,6 | 29,93 | 9,92 | 3,0 | 5,14·10⁸ | 5,1·10⁹ |
| 30 | RESIDUALS | 74,01 | 27,16 | 2,7 | 61,25 | 14,51 | 4,2 | 3,52·10⁸ | 1,06·10¹⁰ |
| 30 | MERIT | 68,34 | 20,56 | 3,3 | 61,39 | 14,66 | 4,2 | 3,48·10⁸ | 1,04·10¹⁰ |
| 30 | MERIT + Extrema | 90,10 | 35,09 | 2,6 | 83,08 | 28,81 | 2,9 | 1,77·10⁸ | 5,3·10⁹ |
| 40 | RESIDUALS | 96,02 | 33,66 | 2,9 | 80,74 | 19,29 | 4,2 | 2,64·10⁸ | 1,06·10¹⁰ |
| 40 | MERIT | 88,98 | 26,84 | 3,3 | 80,92 | 19,51 | 4,1 | 2,61·10⁸ | 1,05·10¹⁰ |
| 40 | MERIT + Extrema | 119,35 | 47,13 | 2,5 | 109,23 | 38,11 | 2,9 | 1,34·10⁸ | 5,4·10⁹ |

Jacobi und LM-Iteration (S = 30, Median aus 5 Läufen):

| Fall | Strahlen | FP64 MERIT Engine / Kernel | df64 MERIT Engine / Kernel | df64 RESIDUALS Engine / Kernel |
|---|---:|---:|---:|---:|
| V = 441 (Jacobi) | 224.910 | 3,59 / 3,19 ms | 1,56 / 0,76 ms | 1,84 / 0,77 ms |
| V = 392.157 (LM) | 200.000.070 | 2,70 / 2,40 s | 0,94 / 0,59 s | 1,35 / 0,58 s |

**Auswertung.**

- **Kernel:** df64 ist für S ≥ 30 4,2-mal so schnell wie FP64, für S = 10 3,7–3,9-mal (der Anteil außerhalb der Flächenschleife – Startstrahl mit zwei df64-sincos, Referenzsphäre, Kolben, Umwandlungen – wiegt dort mehr). Wie in FP64 ist der Durchsatz in Strahl-Flächen/s nahezu unabhängig von S (1,05·10¹⁰ gegen 2,5·10⁹).
- **Engine:** Nur 2,3–3,3-mal so schnell, weil Upload des Batches, Download der Ergebnisse und Prüfung auf dem Host nicht kürzer werden (gleiche Bytes wie FP64) und jetzt 29 % (MERIT) bzw. 47 % (RESIDUALS) der Engine-Zeit bei S = 30 ausmachen. Gepinnte Puffer und Überlappung über Streams (M7) bringen für df64 entsprechend mehr.
- **Max-Reduktionen** verdoppeln die df64-Kernelzeit (in FP64 +35 %): `RaySurfaceRecord` und `SurfaceExtrema` im Local Memory werden nicht schneller. Kleinere Felder für kleine S sind ein Kandidat für M7-T4.
- **Gegen die Schätzung** in [hardware.md](hardware.md#durchsatzabschätzung) (df64 ca. 2–3·10⁸ Strahlen/s bei S = 30, Jacobi ca. 1 ms, LM ca. 1 s): im Kernel 3,5·10⁸ Strahlen/s, 0,76 ms, 0,59 s, also 1,2–1,7-mal besser; über die Engine 2,5·10⁸ Strahlen/s, 1,6 ms und 0,94 s (MERIT), innerhalb der Schätzung.
- **Warum nicht mehr als Faktor 4:** Eine df64-Operation kostet 9–20 FP32-Befehle (Addition 20, Multiplikation 9, Division und Wurzel je eine IEEE-Division bzw. -Wurzel als Unterprogramm plus ca. 15), und die Abhängigkeitsketten sind lang. Mit 96–104 Registern laufen 16–20 Warps pro SM (architektur.md, „Zahlentypen der Kernel“), zu wenig, um diese Latenzen ganz zu verstecken; dazu kommen die leeren Lanes der Blöcke (siehe oben, 76 % Lane-Ausnutzung). Hardwarezähler waren nicht verfügbar; Profil und Tuning (Blockgröße, `__launch_bounds__`, Registerdruck) sind M7-T4.

Register und Stack der df64-Kernel: [architektur.md, „Zahlentypen der Kernel“](architektur.md#zahlentypen-der-kernel-m7-t1) (sm_86: `residuals_kernel<df64, false>` 96 Register, Stack 48 B → 5 Blöcke/SM = 20 von 48 Warps; `merit_kernel<df64, false>` 104 Register, Stack 32 B → 4 Blöcke/SM; mit Max-Reduktionen Stack 2.080–2.096 B). Genauigkeit gegen FP64 (Abnahme M7-T1): [numerik.md](numerik.md#zahlentyp-df64-m7-t1). Ergänzend misst der abgeschaltete GPU-Test `CudaDf64.DISABLED_Throughput` (`--gtest_also_run_disabled_tests`) gestörte Varianten der Testsysteme mit versagten Strahlen: `einzellinse_bk7.json` und `testsystem_mehrlinser.json`, je 10.000 Varianten, MERIT über die Engine 2,8–3,2-mal so schnell wie FP64.

## FP32-Vorfilterung (M7-T2)

Kernel der FP32-Vorfilterung (`TraceEngine::prefilter_variants`, [numerik.md, „FP32-Vorfilterung“](numerik.md#fp32-vorfilterung-m7-t2)) gegen MERIT in FP64 und df64, Messprogramm und Systeme wie oben (`--type fp32prefilter`: ein Fall pro S und V, Modus-Spalte `prefilter`; Engine = `prefilter_variants` mit Prüfung, Upload, Kernel, Download der Rohdaten und Entscheidung auf dem Host, Kernel = `DevicePrefilter::run`). Messumgebung wie in „Messumgebung RTX 3090“, Stand main 2297b58 mit M7-T2 (Branch `m7/t2-fp32-vorfilter`), Datum 2026-10-18. Befehle:

```text
trace_throughput --type fp64,df64,fp32prefilter --surfaces 10,30,40 --variants 10000 --mode merit --repeat 20 --warmup 3
trace_throughput --type fp64,df64,fp32prefilter --surfaces 30 --variants 100000,392157 --mode merit --repeat 5 --warmup 1
```

Kein Strahl versagt. V = 10.000, 510 Strahlen pro Variante, Median aus 20 Läufen:

| S | Engine FP64 [ms] | Engine df64 [ms] | Engine FP32-Vorfilter [ms] | Kernel FP64 [ms] | Kernel df64 [ms] | Kernel FP32-Vorfilter [ms] | Faktor gegen FP64 / df64 (Kernel) | Kernel FP32 [Strahl-Flächen/s] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28,62 | 8,09 | 4,53 | 25,74 | 5,68 | 1,36 | 19 / 4,2 | 3,8·10¹⁰ |
| 30 | 74,93 | 20,71 | 10,88 | 68,88 | 15,29 | 3,01 | 23 / 5,1 | 5,1·10¹⁰ |
| 40 | 103,52 | 36,71 | 14,24 | 94,64 | 19,23 | 4,03 | 23 / 4,8 | 5,1·10¹⁰ |

(S = 40, FP64: zweite Messung mit 20 Läufen; die erste ergab für den Kernel 130,6 ms, offenbar gestört.) Große Batches (S = 30, Median aus 5 Läufen):

| V | Strahlen | FP64 Engine / Kernel | df64 Engine / Kernel | FP32-Vorfilter Engine / Kernel |
|---:|---:|---:|---:|---:|
| 100.000 | 5,1·10⁷ | 773 / 694 ms | 247 / 169 ms | 120 / 36,8 ms |
| 392.157 | 2,0·10⁸ | 3,04 / 2,76 s | 0,98 / 0,68 s | 0,45 / 0,15 s |

**Auswertung.**

- **Kernel:** 1,3–1,7·10⁹ Strahlen/s bei S = 30 (5,1·10¹⁰ Strahl-Flächen/s), 18–23-mal so schnell wie FP64 und 4,5–5-mal so schnell wie df64. Das Verhältnis 64:1 der Spitzenleistung wird nicht erreicht: Der Kernel ist nicht mehr durch die Rechenwerke begrenzt, sondern durch Latenzen, Shared-Memory-Zugriffe, die drei zusätzlichen Grenzabstände pro Fläche und die leeren Lanes der Blöcke (76 % Lane-Ausnutzung bei 80 Stützstellen, siehe oben). 72 Register, also 7 Blöcke pro SM (architektur.md, „FP32-Vorfilterung“).
- **Engine:** nur 6,7–6,9-mal so schnell wie FP64 und 1,9–2,2-mal so schnell wie df64, weil Prüfung der Werte (`validate_values`) und Upload des FP64-Batches auf dem Host gleich bleiben und jetzt rund zwei Drittel der Zeit ausmachen (V = 392.157: 0,30 von 0,45 s). Ein Multistart, der den Batch ohnehin auf dem Host erzeugt, profitiert von gepinnten Puffern und überlappten Uploads (M7-T4) entsprechend mehr.
- **Einsatz:** Bei 10⁵–4·10⁵ Startvarianten sortiert die Vorfilterung in 0,12–0,45 s aus; die FP64-Bewertung nur der übrigen spart bei 50–98 % Ablehnungen (je nach Kriterium, vgl. die Abnahme in numerik.md) 0,4–3 s pro Durchgang.

Durchsatz mit versagten Strahlen (Multistart-Varianten des Mehrlinsers, 10.000 Varianten, 8,5 % versagte Strahlen, GPU-Test `CudaPrefilter.Throughput`, über die Engine): Vorfilterung 4,9–5,2 ms, MERIT FP64 26–29 ms, df64 8,2–8,9 ms.

## ADJOINT-Grundgerüst (M5-T1)

Vorwärtsdurchlauf mit Aufzeichnung pro Strahl (Modi `adjoint-none`, `adjoint-full`, `adjoint-checkpoint`, [architektur.md, „Modus ADJOINT“](architektur.md#modus-adjoint-vorwärtsdurchlauf-mit-aufzeichnung-m5-t1)) gegen RESIDUALS. Messumgebung wie oben (Treiber unterstützt CUDA 13.3, Toolkit 12.9), Branch `m5/t1-trace-record`, Datum 2026-10-19. **Einschränkung:** Während der Messungen liefen auf derselben GPU zeitweise Tests anderer Arbeitsbäume; die Absolutwerte streuen zwischen drei Läufen um bis zu 30 % (FP64 RESIDUALS bei S = 16: 30,9 / 59,2 / 100,1 ms), die Verhältnisse innerhalb eines Laufs sind stabiler. Die Tabelle zeigt den ruhigsten Lauf; eine Wiederholung auf einer freien GPU (und auf der P100) steht aus.

```text
trace_throughput --type fp64,df64 --surfaces 16,30,60 --variants 10000 --mode residuals,adjoint --repeat 20 --warmup 5
```

Kernelzeit [ms], V = 10.000, 510 Strahlen pro Variante, kein Strahl versagt:

| S | Typ | RESIDUALS | adjoint-none | adjoint-full | adjoint-checkpoint |
|---:|---|---:|---:|---:|---:|
| 16 | FP64 | 59,24 | 56,59 | 55,65 | 105,82 |
| 30 | FP64 | 77,27 | 94,18 | 100,57 | 158,91 |
| 60 | FP64 | 158,34 | 158,68 | 159,57 | 304,46 |
| 16 | df64 | 12,94 | 13,16 | 22,48 | 36,42 |
| 30 | df64 | 21,30 | 17,66 | 49,28 | 81,23 |
| 60 | df64 | 34,19 | 34,23 | 57,00 | 90,09 |

Auswertung: In FP64 kostet die vollständige Aufzeichnung auf der RTX 3090 0–7 % (der Kernel ist durch FP64 begrenzt), Checkpoints 70–90 % (Neu-Vorwärtsrechnen). In df64 ist die vollständige Aufzeichnung durch den Speicherverkehr begrenzt (+67 … +180 %), Checkpoints sind noch langsamer. Gewählt ist daher die vollständige Aufzeichnung mit Kapazitätsklassen; Register und Local Memory pro Variante in architektur.md. Die „Engine“-Spalten der Modi `adjoint-*` (Upload, Lauf, Download ohne die Aufzeichnung) sind wegen der geteilten GPU nicht aussagekräftig und hier weggelassen.

## Profiling und Tuning (M7-T4)

**Ergebnis in Kürze.** Die Kernel RESIDUALS und MERIT rechnen seit M7-T4 einen Block pro *Tupelpaket* (aufeinanderfolgende Tupel (f, l) einer Variante, bis 512 Strahlen) statt pro Tupel; Kolben und Summen bleiben in der Reihenfolge von M2, die Ergebnisse sind **bitgleich** zu vorher. Dazu ein Doppelpuffer-Akkumulator für die Max-Reduktionen und eine schnellere Prüfung der Werte auf dem Host. Auf der RTX 3090 ist der Kernel in FP64 1,24–1,64-mal und in df64 1,47–2,16-mal so schnell, über die Engine im Modus MERIT 1,28–2,0-mal (RESIDUALS in df64 wegen des unveränderten Downloads nur 1,0–1,4-mal; Tabellen unten). Für die Synthese-Konfiguration (5 Felder × 7 Wellenlängen) 1,25–1,48-mal. Gemessen und verworfen: Registergrenze (`__launch_bounds__` mit 5 Blöcken), gepinnte Zwischenpuffer für den Upload.

### Messumgebung und Methode

RTX 3090 wie in „Messumgebung RTX 3090“ (Treiber 610.88, CUDA 12.9, Release), main 59c2408 („vorher“, unverändert gebaut) gegen Branch `m7/t4-tuning` („nachher“), Datum 2026-10-19.

- **Keine Hardwarezähler:** Nsight Compute 2026.3 bricht mit `ERR_NVGPUCTRPERM` ab (Leistungszähler nur für Administratoren). Freischalten und sinnvolle `ncu`-Aufrufe: [hardware.md, „Profiling mit Nsight Compute“](hardware.md#profiling-mit-nsight-compute). Ersatzweise: Register, Stack und Shared Memory aus `cuobjdump --dump-resource-usage`, Belegung daraus berechnet, gezielte Experimente mit der Quadratur (Lane-Ausnutzung), Varianten derselben Kernel (Paketgröße, Registergrenze) und eine Aufschlüsselung der Engine-Zeit (Hilfsprogramm, nicht im Repository).
- **Geteilter Rechner:** Während der Messungen liefen zeitweise Tests und ein `mikroptim synth` anderer Arbeitsbäume auf derselben GPU und ein fremdes Programm auf ca. 20 der 24 CPU-Kerne. Medianwerte schwankten dadurch um bis zu den Faktor 2. Die Tabellen zeigen deshalb das **Minimum** aus 3 Runden × 20 Läufen (nach 3 Aufwärmläufen); vorher und nachher liefen pro Fall direkt hintereinander und erst, wenn mindestens 9 s kein fremder Mikroptim-Prozess lief (in der Abschlussmessung musste nie gewartet werden). Die Minima von „vorher“ stimmen mit den Medianen der früheren Abschnitte überein (FP64 MERIT S = 30: 65,1 gegen 60,5 ms in M2-T6; df64: 16,3 gegen 14,7 ms in M7-T1). Hostseitige Zeiten (Engine, Prüfung, Upload) sind unsicherer als die Kernelzeiten.
- **Befehl** pro Fall (Skript im Arbeitsverzeichnis, nicht im Repository): `trace_throughput --surfaces S --variants 441,10000 --mode residuals,merit,merit+extrema --type T --repeat 20 --warmup 3 --csv -`, für die Synthese-Konfiguration zusätzlich `--fields 5 --wavelengths 7 --variants 441,2000`, für die Paketgröße `--pack-rays P`. In keinem Fall unterscheiden sich die versagten Strahlen zwischen vorher und nachher.

### Diagnose vor M7-T4

**1. Leere Lanes kosten FP64-Durchsatz.** Bis M7-T3 rechnete ein Block à 128 Threads ein Tupel; bei der Standardquadratur hat ein Tupel 80 Strahlen (Warps 0, 1 voll, Warp 2 halb) bzw. auf der Achse 10 (ein Warp mit 10 Lanes). Ein Experiment mit der Quadratur (main, MERIT, S = 30, V = 10.000, 3 Felder × 3 Wellenlängen, Kernel-Median) zeigt, dass die Zeit mit der Zahl **aktiver Warps** wächst, nicht mit der Zahl der Strahlen:

| N × M | Strahlen pro (v, l) | aktive Warps pro (v, l) | FP64 [ms] | ms pro Warp | df64 [ms] |
|---|---:|---:|---:|---:|---:|
| 4 × 8 | 4 + 32 + 32 = 68 | 1 + 1 + 1 = 3 | 31,8 | 10,6 | 12,7 |
| 10 × 6 | 10 + 60 + 60 = 130 | 1 + 2 + 2 = 5 | 49,4 | 9,9 | 13,6 |
| 10 × 8 (Standard) | 10 + 80 + 80 = 170 | 1 + 3 + 3 = 7 | 67,0 | 9,6 | 17,0 |
| 12 × 8 | 12 + 96 + 96 = 204 | 1 + 3 + 3 = 7 | 67,6 | 9,7 | 17,5 |
| 16 × 8 | 16 + 128 + 128 = 272 | 1 + 4 + 4 = 9 | 85,1 | 9,5 | 20,1 |

12 × 8 rechnet 20 % mehr Strahlen in derselben Zeit wie 10 × 8: Ein FP64-Warp-Befehl belegt die FP64-Einheiten unabhängig von der Maske. Bei der Standardquadratur sind 170 von 224 Lanes aktiv (76 %); dicht gepackt bräuchten die 510 Strahlen einer Variante 16 statt 21 Warps (Erwartung: −24 % Kernelzeit in FP64). Leere Warps (Warp 3 des Blocks) kosten dagegen kaum etwas: Bei 16 Warps pro SM sind die FP64-Einheiten der RTX 3090 auch mit den aktiven allein ausgelastet.

**2. Register und Belegung** (`cuobjdump`, main 59c2408; Blöcke à 128 Threads pro SM auf sm_86):

| Kernel | sm_60 | sm_86 | Blöcke/SM sm_86 |
|---|---|---|---:|
| `residuals_kernel<double, false/true>` | 126 / 128 Register | 96 / 96 Register | 5 / 5 |
| `merit_kernel<double, false/true>` | 116 / 116 | 118 / 120 | 4 / 4 |
| `residuals_kernel<df64, false/true>` | 121 / 122 | 96 / 96 | 5 / 5 |
| `merit_kernel<df64, false/true>` | 113 / 119 | 96 / 96 | 5 / 5 |

(Stack 48–112 B, mit Max-Reduktionen 2.096–2.184 B; seit M7-T1/M4-T2/M5-T1 also andere Zahlen als in der Profilnotiz oben.) Shared Memory (2–6 KB statisch) begrenzt nicht.

**3. Engine.** Aufschlüsselung eines Engine-Aufrufs (Minimum aus 25 Läufen, V = 10.000, S = 30, 2 Felder × 2 Wellenlängen, 12,4 MB Batch): `validate_values` 4,8–6,8 ms, Upload 1,2–1,6 ms, Kernel MERIT 25 ms (FP64) bzw. 6,6 ms (df64), Engine MERIT 33 bzw. 14 ms. In df64 war also die **Prüfung der Werte auf dem Host** der größte Einzelposten neben dem Kernel: MSVC übersetzt `std::isfinite` in einen Aufruf von `_dtest` pro Wert (ca. 1,5 Mio. Werte).

### Maßnahmen

| Maßnahme | Ergebnis | Status |
|---|---|---|
| **Tupelpakete** (architektur.md, „Kernel-Abbildung“): Block pro Paket aufeinanderfolgender Tupel einer Variante, Strahlen dicht auf den Threads, Kolben und Quadratsumme mit 128 virtuellen Lanes pro Tupel (Reihenfolge von M2) | Kernel 1,24–2,16-mal so schnell (Tabellen unten); bitgleich | umgesetzt |
| **Max-Reduktionen mit Doppelpuffer** (`PendingExtrema`: Beobachter schreibt max/min sofort in den zweiten Puffer, Umschalten nach gültigem Strahl) statt Aufzeichnung plus `add_ray` | halber Local-Memory-Verkehr; Mehrkosten der Extrema bei S = 30, V = 10.000 in df64 von +83 % auf +44 %, in FP64 von +32 % auf +15 % (zusammen mit den Paketen); bitgleich | umgesetzt |
| **`validate_values`**: Endlichkeit über die Exponentenbits | 4,8–6,8 → 1,7–3,3 ms (Minimum; Median unter Last 11–13 → 3–10 ms) | umgesetzt |
| **Registergrenze** `__launch_bounds__(128, 5)` (96 Register, 5 statt 4 Blöcke pro SM für FP64-MERIT und df64) | Kernel ±2 % (FP64 MERIT S = 30: 47,5 → 48,0 ms, df64 11,1 → 11,0 ms), kleine Spills (Stack +32 B) | verworfen auf der RTX 3090; Messschalter `MKO_KERNEL_MIN_BLOCKS` für die P100 ([hardware.md](hardware.md#belegung-und-blockabbildung-m7-t4)) |
| **Gepinnte Zwischenpuffer** für den Upload (2 × 4 MiB, Abschnitte überlappt per `cudaMemcpyAsync`) und den Download | Upload 12,4 MB: Minimum 1,2–2,0 ms gegen 1,2–1,6 ms pageable; Median unter Last uneinheitlich (1,3–7,9 gegen 2,3–6,9 ms). Der Host kopiert ohnehin einmal (std::vector ist pageable), der Treiber tut dasselbe | verworfen |
| **Streams und Überlappung** Upload i + 1 / Kernel i | nicht umgesetzt: bei S = 30 passen ca. 10⁶ Varianten in eine Tranche, Mehrtranchen-Läufe sind selten; der Upload ist nach der Prüfung ≤ 2 ms | offen (bei Bedarf) |
| **Local Memory nur für die tatsächliche Flächenzahl** (Template-Klassen) | kein Gewinn erwartet: Der Verkehr ist schon proportional zu S, nur die Laufzeitreserve hinge von S ab | nicht umgesetzt |

**Wahl der Paketgröße** (`pack_rays`, MERIT, S = 30, 3 Felder × 3 Wellenlängen, Kernel-Minimum [ms]; die Ergebnisse sind für jede Größe bitgleich):

| `pack_rays` | Blöcke pro Variante | FP64, V = 441 | FP64, V = 10.000 | df64, V = 441 | df64, V = 10.000 |
|---:|---:|---:|---:|---:|---:|
| 1 (Abbildung von M2 im neuen Code) | 9 | 3,082 | 62,481 | 0,821 | 17,714 |
| 128 | 6 | 2,769 | 56,473 | 0,650 | 14,140 |
| 256 | 3 | 2,817 | 50,784 | 0,639 | 12,437 |
| 512 (Standard) | 1 | 2,710 | 47,516 | 0,568 | 10,745 |
| 1024 | 1 | 2,713 | 47,725 | 0,564 | 10,270 |

Mit `pack_rays` = 1 (ein Tupel pro Block wie in main) ist der neue Code in FP64 etwas schneller als main (62,5 gegen 65,1 ms) und in df64 9 % langsamer (17,7 gegen 16,3 ms): Die zusätzliche Arbeit (Tupelsuche pro Strahl, Kolben aus dem Speicher statt aus Registern, mehr Register) ist klein gegen den Gewinn der Pakete. Der Gewinn kommt aus den gefüllten Warps (FP64: −24 % wie aus der Diagnose erwartet) und, besonders in df64, aus der Verteilung der festen Arbeit pro Block (Laden der Tabelle, Bildpunkt, Synchronisation, Reduktionen) auf mehr Strahlen. 512 ist die kleinste Größe, die bei der Standardkonfiguration eine Variante in einen Block packt; 1024 bringt nur in df64 bei großen V noch 4 %, kostet aber bei kleinen V Parallelität (weniger, größere Blöcke) und Shared Memory im Modus MERIT. Die Blockgröße bleibt 128 Threads (bei 512 Strahlen 4 Durchgänge pro Thread; die Belegung begrenzen die Register, siehe oben).

### Ergebnisse vorher/nachher

Standardreihe (3 Felder × 3 Wellenlängen, 510 Strahlen pro Variante; Minimum, siehe Methode):

| Typ | S | V | Modus | Kernel vorher [ms] | Kernel nachher [ms] | Faktor | Engine vorher [ms] | Engine nachher [ms] | Faktor | Kernel nachher [Str/s] |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| FP64 | 10 | 441 | RESIDUALS | 1,343 | 0,883 | 1,52 | 2,113 | 1,639 | 1,29 | 2,55·10⁸ |
| FP64 | 10 | 441 | MERIT | 1,234 | 0,908 | 1,36 | 1,625 | 1,098 | 1,48 | 2,48·10⁸ |
| FP64 | 10 | 441 | MERIT + Extrema | 1,529 | 1,029 | 1,49 | 1,804 | 1,245 | 1,45 | 2,19·10⁸ |
| FP64 | 30 | 441 | RESIDUALS | 2,990 | 2,361 | 1,27 | 4,310 | 3,173 | 1,36 | 9,53·10⁷ |
| FP64 | 30 | 441 | MERIT | 2,963 | 2,391 | 1,24 | 3,444 | 2,693 | 1,28 | 9,41·10⁷ |
| FP64 | 30 | 441 | MERIT + Extrema | 3,943 | 2,745 | 1,44 | 4,461 | 3,088 | 1,44 | 8,19·10⁷ |
| FP64 | 40 | 441 | RESIDUALS | 3,860 | 3,081 | 1,25 | 5,467 | 3,908 | 1,40 | 7,30·10⁷ |
| FP64 | 40 | 441 | MERIT | 3,884 | 3,105 | 1,25 | 4,419 | 3,422 | 1,29 | 7,24·10⁷ |
| FP64 | 40 | 441 | MERIT + Extrema | 5,171 | 3,592 | 1,44 | 5,782 | 3,952 | 1,46 | 6,26·10⁷ |
| FP64 | 10 | 10.000 | RESIDUALS | 25,467 | 17,343 | 1,47 | 39,739 | 30,186 | 1,32 | 2,94·10⁸ |
| FP64 | 10 | 10.000 | MERIT | 25,816 | 17,777 | 1,45 | 28,370 | 19,204 | 1,48 | 2,87·10⁸ |
| FP64 | 10 | 10.000 | MERIT + Extrema | 32,925 | 20,132 | 1,64 | 35,734 | 21,786 | 1,64 | 2,53·10⁸ |
| FP64 | 30 | 10.000 | RESIDUALS | 65,198 | 47,137 | 1,38 | 83,163 | 61,903 | 1,34 | 1,08·10⁸ |
| FP64 | 30 | 10.000 | MERIT | 65,131 | 47,559 | 1,37 | 73,223 | 50,601 | 1,45 | 1,07·10⁸ |
| FP64 | 30 | 10.000 | MERIT + Extrema | 86,090 | 54,628 | 1,58 | 94,639 | 58,619 | 1,61 | 9,34·10⁷ |
| FP64 | 40 | 10.000 | RESIDUALS | 83,917 | 62,089 | 1,35 | 105,038 | 78,425 | 1,34 | 8,21·10⁷ |
| FP64 | 40 | 10.000 | MERIT | 84,608 | 62,534 | 1,35 | 93,473 | 67,129 | 1,39 | 8,16·10⁷ |
| FP64 | 40 | 10.000 | MERIT + Extrema | 112,626 | 72,241 | 1,56 | 122,416 | 77,969 | 1,57 | 7,06·10⁷ |
| df64 | 10 | 441 | RESIDUALS | 0,415 | 0,233 | 1,78 | 1,267 | 0,955 | 1,33 | 9,64·10⁸ |
| df64 | 10 | 441 | MERIT | 0,434 | 0,246 | 1,76 | 0,727 | 0,444 | 1,64 | 9,13·10⁸ |
| df64 | 10 | 441 | MERIT + Extrema | 0,579 | 0,324 | 1,79 | 0,890 | 0,560 | 1,59 | 6,95·10⁸ |
| df64 | 30 | 441 | RESIDUALS | 0,825 | 0,542 | 1,52 | 1,802 | 1,319 | 1,37 | 4,15·10⁸ |
| df64 | 30 | 441 | MERIT | 0,830 | 0,560 | 1,48 | 1,253 | 0,829 | 1,51 | 4,02·10⁸ |
| df64 | 30 | 441 | MERIT + Extrema | 1,389 | 0,800 | 1,74 | 1,852 | 1,122 | 1,65 | 2,81·10⁸ |
| df64 | 40 | 441 | RESIDUALS | 1,061 | 0,714 | 1,49 | 2,113 | 1,581 | 1,34 | 3,15·10⁸ |
| df64 | 40 | 441 | MERIT | 1,064 | 0,724 | 1,47 | 1,612 | 1,052 | 1,53 | 3,11·10⁸ |
| df64 | 40 | 441 | MERIT + Extrema | 1,766 | 1,059 | 1,67 | 2,348 | 1,452 | 1,62 | 2,12·10⁸ |
| df64 | 10 | 10.000 | RESIDUALS | 7,156 | 3,972 | 1,80 | 21,322 | 16,764 | 1,27 | 1,28·10⁹ |
| df64 | 10 | 10.000 | MERIT | 7,527 | 4,148 | 1,81 | 9,903 | 5,632 | 1,76 | 1,23·10⁹ |
| df64 | 10 | 10.000 | MERIT + Extrema | 11,454 | 5,291 | 2,16 | 13,697 | 6,953 | 1,97 | 9,64·10⁸ |
| df64 | 30 | 10.000 | RESIDUALS | 16,317 | 10,802 | 1,51 | 33,283 | 24,628 | 1,35 | 4,72·10⁸ |
| df64 | 30 | 10.000 | MERIT | 16,265 | 10,359 | 1,57 | 22,127 | 13,570 | 1,63 | 4,92·10⁸ |
| df64 | 30 | 10.000 | MERIT + Extrema | 29,691 | 14,880 | 2,00 | 36,132 | 17,846 | 2,02 | 3,43·10⁸ |
| df64 | 40 | 10.000 | RESIDUALS | 21,054 | 14,145 | 1,49 | 39,910 | 39,540 | 1,01 | 3,61·10⁸ |
| df64 | 40 | 10.000 | MERIT | 21,220 | 14,250 | 1,49 | 28,711 | 17,553 | 1,64 | 3,58·10⁸ |
| df64 | 40 | 10.000 | MERIT + Extrema | 38,883 | 19,700 | 1,97 | 47,530 | 23,671 | 2,01 | 2,59·10⁸ |

Synthese-Konfiguration (5 Felder × 7 Wellenlängen, Feld 0 auf der Achse: 2.310 Strahlen pro Variante; V = 441 entspricht einer Jacobi-Matrix mit 440 Spalten, V = 2.000 einem kleinen Multistart):

| Typ | S | V | Modus | Kernel vorher [ms] | Kernel nachher [ms] | Faktor | Engine vorher [ms] | Engine nachher [ms] | Faktor | Kernel nachher [Str/s] |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| FP64 | 30 | 441 | RESIDUALS | 12,449 | 9,825 | 1,27 | 15,413 | 12,575 | 1,23 | 1,04·10⁸ |
| FP64 | 30 | 441 | MERIT | 12,493 | 9,951 | 1,26 | 13,101 | 10,287 | 1,27 | 1,02·10⁸ |
| FP64 | 40 | 441 | RESIDUALS | 16,112 | 12,831 | 1,26 | 19,186 | 15,649 | 1,23 | 7,94·10⁷ |
| FP64 | 40 | 441 | MERIT | 16,196 | 12,952 | 1,25 | 16,904 | 13,492 | 1,25 | 7,87·10⁷ |
| FP64 | 30 | 2.000 | RESIDUALS | 55,974 | 43,322 | 1,29 | 68,533 | 54,919 | 1,25 | 1,07·10⁸ |
| FP64 | 30 | 2.000 | MERIT | 56,484 | 43,704 | 1,29 | 58,826 | 44,887 | 1,31 | 1,06·10⁸ |
| FP64 | 40 | 2.000 | RESIDUALS | 73,076 | 57,036 | 1,28 | 86,072 | 69,075 | 1,25 | 8,10·10⁷ |
| FP64 | 40 | 2.000 | MERIT | 73,398 | 57,365 | 1,28 | 76,451 | 58,690 | 1,30 | 8,05·10⁷ |
| df64 | 30 | 441 | RESIDUALS | 2,857 | 2,126 | 1,34 | 6,801 | 4,800 | 1,42 | 4,79·10⁸ |
| df64 | 30 | 441 | MERIT | 2,857 | 2,286 | 1,25 | 3,555 | 2,656 | 1,34 | 4,46·10⁸ |
| df64 | 40 | 441 | RESIDUALS | 4,074 | 2,759 | 1,48 | 7,130 | 5,555 | 1,28 | 3,69·10⁸ |
| df64 | 40 | 441 | MERIT | 3,776 | 2,837 | 1,33 | 4,530 | 3,438 | 1,32 | 3,59·10⁸ |
| df64 | 30 | 2.000 | RESIDUALS | 12,667 | 9,639 | 1,31 | 25,122 | 20,798 | 1,21 | 4,79·10⁸ |
| df64 | 30 | 2.000 | MERIT | 13,473 | 9,974 | 1,35 | 15,312 | 10,771 | 1,42 | 4,63·10⁸ |
| df64 | 40 | 2.000 | RESIDUALS | 16,894 | 13,160 | 1,28 | 29,523 | 23,920 | 1,23 | 3,51·10⁸ |
| df64 | 40 | 2.000 | MERIT | 17,252 | 12,843 | 1,34 | 19,782 | 14,468 | 1,37 | 3,60·10⁸ |

**Auswertung.**

- **FP64:** 1,24–1,38-mal (ohne Extrema, S ≥ 30), wie von der Lane-Ausnutzung erwartet (21 → 16 Warps: 1,31); bei S = 10 mehr (1,36–1,52), weil dort die feste Arbeit pro Block mehr wiegt. In der Synthese-Konfiguration 1,25–1,29: Mit 5 Feldern war die Lane-Ausnutzung vorher schon 79 % (2.310 Strahlen in 91 Warps), jetzt 99 % (5 Pakete mit 470, 480, 480, 480 und 400 Strahlen, zusammen 73 Warps).
- **df64:** 1,47–1,81-mal ohne Extrema. df64 ist latenzbegrenzt; dort wirkt neben den gefüllten Warps die Verteilung der festen Arbeit pro Block (Laden der Tabelle, df64-Umwandlungen, zwei `__syncthreads` und die Reduktionen) auf viermal so viele Strahlen.
- **Max-Reduktionen:** zusätzlich zu den Paketen der Doppelpuffer, insgesamt 1,44–1,64 (FP64) bzw. 1,67–2,16 (df64).
- **Engine:** zusätzlich der schnellere `validate_values`; MERIT 1,28–1,48 (FP64) bzw. 1,51–1,76 (df64), mit Extrema bis 2,0. RESIDUALS gewinnt in df64 weniger (1,01–1,37), weil dort der Download von Residuen, Strafresiduen und Status (ca. 87 MB bei V = 10.000) unverändert bleibt.
- **Bitgleichheit:** Alle Ausgaben (Residuen, OPD, Status, Merit, Zähler, Max-Reduktionen) sind bitgleich zu main: 60 Fälle (3 Testsysteme mit Versagen und Geisterplatten, Quadraturen 10 × 8 weich, 6 × 5 hart, 16 × 16 weich, 3 × 3 hart, 12 × 40 hart mit 480 Stützstellen pro Feld, FP64 und df64, je 300 gestörte Varianten, 214 MB Ausgaben) mit gleichem SHA-256 aus dem Build von main und dem Branch, ebenso für Paketgrößen 1, 128, 1.024 und 4.000. Im Repository prüfen `CudaTuplePacks.ResultsAreBitIdenticalForAllPackSizes` (Paketgrößen 1, 128, 512, 4.096) und unverändert `MultiTraceEngine.ResultFingerprint` (dieselben Prüfsummen wie in [hardware.md](hardware.md#mehrere-gpus-m7-t3)); `CudaMerit.MatchesSumOfResiduals` findet jetzt 3.999 von 4.000 Varianten bitgleich zur Nachrechnung der Kernel-Reihenfolge. GPU gegen CPU ändert sich damit nicht (alle Toleranztests unverändert grün).

**Register nach M7-T4** (sm_60 / sm_86, Blöcke pro SM auf sm_86): `residuals_kernel<double>` 122 / 96 (5), mit Extrema 116 / 96 (5); `merit_kernel<double>` 107 / 104 (4), mit Extrema 110 / 104 (4); `residuals_kernel<df64>` 108 / 106 (4), mit Extrema 108 / 124 (4); `merit_kernel<df64>` 96 / 112 (4), mit Extrema 96 / 96 (5). Statisches Shared Memory 3,6 KB, mit Extrema 7,6 KB (vorher 2,0 bzw. 6,1 KB). df64 belegt auf sm_86 teils einen Block weniger als vorher und ist trotzdem schneller; eine Grenze von 96 Registern brachte nichts (siehe Maßnahmen).

### Offene Punkte

- **P100:** Messung der Pakete und der Registergrenze (Befehle unten). Auf der P100 (FP64 mit halber FP32-Rate) dürfte die Belegung (4–5 von 16 Blöcken, 25–31 %) eher begrenzen als auf der RTX 3090; ein Build mit `-DMKO_KERNEL_MIN_BLOCKS=5` zeigt, ob 96 Register helfen.
- **Nsight Compute:** nach dem Freischalten der Zähler die Annahmen dieses Abschnitts prüfen (FP64-Pipeline nahe 100 %, Lane-Ausnutzung nahe 32, Local-Memory-Verkehr der Extrema), [hardware.md](hardware.md#profiling-mit-nsight-compute).
- **FP32-Vorfilterung und EVAL** rechnen weiter einen Block pro Tupel. Die Vorfilterung ist über die Engine durch Prüfung und Upload begrenzt (Kernel 3 ms von 11 ms bei S = 30, V = 10.000), die Pakete brächten dort ca. 7 % der Engine-Zeit; EVAL hat dichte Raster (viele Strahlen pro Tupel) und eher das umgekehrte Problem (zu wenige Blöcke bei wenigen Varianten, architektur.md).
- **ADJOINT** (M5-T1/M5-T2) nutzt noch die alte Abbildung; die Pakete lassen sich übernehmen, wenn der Rückwärtsdurchlauf steht (gleiche Reduktionsreihenfolge, also bitgleiche Residuen wie RESIDUALS).
- **Download** der Residuen im Modus RESIDUALS über die Engine (bei S = 40, V = 10.000 in df64 ca. 25 ms außerhalb des Kernels, fast alles Download) bleibt der größte Posten außerhalb des Kernels; der Optimierer nutzt für die Jacobi-Matrix `run_on_device` ohne Download.

## P100-Messung nachholen

Auf dem P100-Rechner (Linux, CUDA 12.x):

```bash
cmake --preset release-cuda && cmake --build --preset release-cuda --target trace_throughput
./build/release-cuda/mikroptim --devices          # Index der P100 ablesen
./build/release-cuda/bench/trace_throughput --device <P100-Index> --mode all --repeat 20 --warmup 3 \
    --csv p100.csv --json p100.json
./build/release-cuda/bench/trace_throughput --device <P100-Index> --surfaces 30 \
    --variants 441,392157 --mode residuals,merit --repeat 5
```

Zu M7-T4 (Tupelpakete und Registergrenze) zusätzlich:

```bash
# Paketgröße: 1 entspricht der Abbildung vor M7-T4, 512 ist der Standard
for p in 1 128 256 512 1024; do
  ./build/release-cuda/bench/trace_throughput --device <P100-Index> --surfaces 30 --variants 441,10000 \
      --mode merit,merit+extrema --type fp64,df64 --pack-rays $p --repeat 20 --warmup 3 --csv p100_pack_$p.csv
done
# Registergrenze 96 (5 Blöcke pro SM) in einem zweiten Build
cmake -S . -B build/minblocks5 -DCMAKE_BUILD_TYPE=Release -DMIKROPTIM_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_FLAGS="-DMKO_KERNEL_MIN_BLOCKS=5" && cmake --build build/minblocks5 --target trace_throughput
./build/minblocks5/bench/trace_throughput --device <P100-Index> --surfaces 10,30,40 --variants 441,10000 \
      --mode residuals,merit,merit+extrema --type fp64 --repeat 20 --warmup 3 --csv p100_minblocks5.csv
```

Lohnt die Registergrenze auf der P100 (mehr als einige Prozent), sollte sie pro Architektur voreingestellt werden (Frage an den Maintainer; die Bits sind mit `MultiTraceEngine.ResultFingerprint` zu prüfen).

Unter Windows mit dem Visual-Studio-Generator liegt das Programm unter `build/release-cuda/bench/Release/trace_throughput.exe`. Dann die Zeile „Tesla P100“ in der Übersicht, eine Tabelle wie oben und den Vergleich mit der Schätzung (4–5·10⁸ Strahlen/s bei S = 30) ergänzen. Erwartung aus dem Spitzenwert (56 SMs · 32 FP64-Ergebnisse pro Takt · 2 Flop · 1,48 GHz ≈ 5,3 TFLOPS, ca. 8 × RTX 3090): bei gleicher Ausnutzung bis zu ca. 7·10⁸ Strahlen/s, wegen der geringen Belegung (siehe Profilnotiz) eher weniger.
