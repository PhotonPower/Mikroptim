# Benchmarks

Gemessener Durchsatz der Trace-Kernel. Messprogramm: `bench/trace_throughput` (M2-T6, Optionen und Ausgabeformate in [bench/README.md](../bench/README.md)). Die Schätzungen, mit denen die Messungen verglichen werden, stehen in [hardware.md, „Durchsatzabschätzung“](hardware.md#durchsatzabschätzung).

## Übersicht

Eine Zeile pro Messreihe: S = 30 Flächen, V = 10.000 Varianten, Standardquadratur 10 × 8, 3 Felder × 3 Wellenlängen (510 Strahlen pro Variante), Modus MERIT über die TraceEngine (inklusive Upload und Download); in Klammern der Kernel allein.

| Datum | Commit | GPU | Treiber / CUDA | Typ | Flächen | Varianten × Strahlen | Strahlen/s |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | 33356d9 + M2-T6 | RTX 3090 | 610.88 / 12.9 | FP64 | 30 | 10.000 × 510 | 7,7·10⁷ (Kernel 8,4·10⁷) |
| 2026-10-18 | a210b89 + M7-T1 | RTX 3090 | 610.88 / 12.9 | df64 | 30 | 10.000 × 510 | 2,5·10⁸ (Kernel 3,5·10⁸) |
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

Shared Memory (2–6 KB statisch) begrenzt nicht. Auf der RTX 3090 reichen 16 Warps pro SM, um die FP64-Einheiten zu sättigen. Auf der P100 (32 FP64-Ergebnisse pro Takt und SM) wird die Belegung von 25 % dagegen voraussichtlich zum begrenzenden Faktor (Latenz der FP64-Befehle und der Lade-Operationen schlecht versteckt); das prüft die P100-Messung. Hardwarezähler (Nsight Compute: erreichte Belegung, Auslastung der FP64-Pipeline) waren hier nicht verfügbar (`ERR_NVGPUCTRPERM`, Zähler nur mit Administratorrechten); das gehört zu M7-T4.

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

Unter Windows mit dem Visual-Studio-Generator liegt das Programm unter `build/release-cuda/bench/Release/trace_throughput.exe`. Dann die Zeile „Tesla P100“ in der Übersicht, eine Tabelle wie oben und den Vergleich mit der Schätzung (4–5·10⁸ Strahlen/s bei S = 30) ergänzen. Erwartung aus dem Spitzenwert (56 SMs · 32 FP64-Ergebnisse pro Takt · 2 Flop · 1,48 GHz ≈ 5,3 TFLOPS, ca. 8 × RTX 3090): bei gleicher Ausnutzung bis zu ca. 7·10⁸ Strahlen/s, wegen der geringen Belegung (siehe Profilnotiz) eher weniger.
