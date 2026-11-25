# M6-T14: Verzeichnung als eigene Nebenbedingung, Querfarbe in der Farbregel

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t14--verzeichnung-als-eigene-nebenbedingung-querfarbe-in-der-farbregel), freigegeben vom Maintainer am 2026-11-24; Entscheidung: [ADR-0019](../adr/0019-verzeichnung-hauptstrahl.md). Ausgangslage: Befund F20 aus [M6-T13](m6-t13-farbe-frueh.md) (am Feldrand begrenzt die Kippung der Wellenfront). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090), FP64, Release mit CUDA 12.9, Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json`; keine Linsendaten realer Objektive.

**Kurzfassung.**

- **ADR-0019:** Die Lage des Bildes verlässt die Wellenfront-Merit, die Querfarbe bleibt darin. Optionales Feld `reference_point` (`ideal` Standard, `chief_ray`, `centroid`): Mittelpunkt der Referenzsphäre im realen Hauptstrahl bzw. Strahlschwerpunkt **bei λ_ref**, gemeinsam für alle Wellenlängen eines Feldes; die Verzeichnung ist eine eigene Nebenbedingung `distortion` (|D_f| ≤ D_max aus dem Hauptstrahl). Ohne die neuen Felder bitgleich.
- **Befund zur Diagnose von M6-T13:** Die Kippung am Rand von F3Z ist nicht die Verzeichnung des Hauptstrahls. Der Hauptstrahl liegt bei hoher NA bis zu 0,18 µm (ein halbes Airy-Scheibchen) neben dem Schwerpunkt; mit dem Hauptstrahl als Mittelpunkt **steigt** der RMS bei 546 nm/70 % von 0,035 auf 0,115 Wellen, mit dem Schwerpunkt sinkt er auf 0,027. Deshalb gibt es neben `chief_ray` die Art `centroid`, die für die Synthese empfohlen ist und in allen Messläufen gilt.
- **Bestes System (beugungsbegrenzt nach Strehl):** **Z1S** ab dem 7-Linser Z1f: NA 0,50, **57 % Feld** (vorher 10 %, H1G8-Feldstufe 19 %), 18 El., 436–656 nm, RMS um den Schwerpunkt 0,029/0,061/0,078 Wellen, **Strehl ≥ 0,81** (FFT, lageunabhängig), Verzeichnung −0,70 %, Querfarbe 0,41 µm. Der Kontrolllauf mit idealem Bildpunkt und denselben Werkzeugen (Z1I) erreicht 55 % Feld erst nach zwei Zyklen und nur Strehl ≥ 0,60.
- **Höchste NA polychromatisch:** F3ZS ab F3Z: NA **0,65** bei **52 % Feld** (vorher 0,640 bei 41 %), 22 El., Strehl ≥ 0,71 (vorher 0,66), 405–707 nm. Alle Stufen enden am Elementbudget (20).
- **Etappenziel (NA 0,7, volles Feld, beugungsbegrenzt 436–656 nm) nicht erreicht.** Grenze jetzt: **Querfarbe** (0,38–0,57 µm am Rand, 0,6–0,85 Airy-Radien) und das Elementbudget; die Verzeichnung bleibt unter 1 % (Z1SF erreicht die Grenze bei 66 % Feld).
- **Farbregel:** `--color-module-lateral` griff in M6-T13 nie, weil das Seidel-Kriterium am Punkt bei hoher NA die (reale, kleine) Längsfarbe überschätzt. Neu `--color-lateral-rays`: Querfarbe aus realen Hauptstrahlen gegen den Achs-RMS; in den Messläufen erkannte die Farbregel in 17 von 59 Entscheidungen Querfarbe und setzte das Modul an die Stelle mit größtem |y·ȳ|.
- **Dichtes Flint als Variante** (SYN-28 als Zusatzplatte, Entscheidung Maintainer: nicht ins Wörterbuch): mehr Feld (66 % statt 57 %), schlechtere Güte (Strehl 0,63), die Platte wurde nie zu einem Element.
- **Messzeit** ≈ 6,5 h GPU (höchstens zwei Läufe gleichzeitig), Läufe nicht abgebrochen bis auf F3ZS c3 (Abschnitt 3).

## 1. Umsetzung

| Punkt der Aufgabe | Umsetzung |
|---|---|
| 1. ADR | [ADR-0019](../adr/0019-verzeichnung-hauptstrahl.md): Bildpunkt aus realen Strahlen (`chief_ray`/`centroid`), Querfarbe bleibt in der Wellenfront (Kippung relativ zu λ_ref), Verzeichnung als Nebenbedingung, Verhältnis zu ADR-0017 (Punkt 3 optional ersetzt, κ eingefroren mit ȳ_f) und zur Toleranz-Kippung (unabhängig) |
| 2. Merit-/Residuenpfad | `BatchTopology::reference_point`, `reference_height` [V][F] (Host und Gerät, nur bei den neuen Arten im Speicher), `batch_image_point` liest es → RESIDUALS, MERIT, df64, FP32-Vorfilter, EVAL ohne weitere Änderung; `update_reference_height` in `make_system_batch`, `VariableSet::apply`, `ToleranceModel::apply`, Proben der Geisterdiagnose → FD-Jacobi mit ∂ȳ/∂x; adjungierte Jacobi lehnt ab (Folgeaufgabe). `ConstraintType::Distortion` im Block der paraxialen Bedingungen. CLI `--reference-point` (`trace`, `optimize`, `synth`, `eval`), `--distortion D[,W]` (`optimize`, `synth`) |
| 3. Farbregel/Diagnose | `OpticalHomotopyOptions::color_lateral_rays` (`synth --color-lateral-rays`); Spanne der Hauptstrahlhöhen über die Wellenlängen (`batch_chief_ray_height(…, l)`) im Text der Empfehlung |
| 4. Messung | Abschnitt 3; Vergleichstabelle mit `tools/m6t14_tabelle.py` |
| 5. Doku | numerik.md §7 („Bildpunkt aus realen Strahlen“), optimierung.md §2.8 und §7 („Querfarbe aus realen Hauptstrahlen“), dateiformat.md, cli.md, CHANGELOG, Plan |

Skript `tools/m6t6_zielspezifikation.py`: `--bildpunkt ideal|hauptstrahl|schwerpunkt`, `--verzeichnung D[,W]`, `--farb-querfarbe-strahlen`.

## 2. Hauptstrahl oder Schwerpunkt

Ausgangssystem F3Z c1 (M6-T13, NA 0,640, 41 % Feld), `mikroptim trace --rings 16`, RMS bei 546 nm:

| Bildpunkt | Achse | 70 % | 100 % |
|---|---|---|---|
| ideal (h_f = f tan θ) | 0,021 | 0,035 | 0,105 |
| Hauptstrahl bei λ_ref | 0,021 | 0,115 | 0,166 |
| Schwerpunkt bei λ_ref | 0,021 | 0,027 | 0,043 |

Die Verzeichnung des Hauptstrahls ist −0,21 % bzw. −0,42 % (gleich der Bewertung von M6-T13), der Schwerpunkt liegt aber über dem idealen Punkt (Kippung der Bewertung +0,025 µm bei 70 %), 0,18 µm neben dem Hauptstrahl. Mit dem Hauptstrahl als Mittelpunkt bestraft die Merit diesen Abstand (Kippanteil der Koma) mit vollem Gewicht. Der Schwerpunkt entfernt die Kippung bei λ_ref wie die Bewertung „ohne Kippung“ (0,0265/0,0419 Wellen) und lässt die Querfarbe als Kippung der übrigen Wellenlängen stehen (bei 436 nm/100 %: 0,104 statt 0,085; bei 486 nm 0,061 statt 0,121). Deshalb rechnen alle Messläufe mit `centroid`; `chief_ray` bleibt als übliche Konvention wählbar.

## 3. Messung

Zyklus wie M6-T13 (`--zyklus …`, `--zyklus-runden 4`, `--farb-budget 6`, OPT = `--farb-stetig --farb-ziel voll --farb-modul doublet --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85`), dazu die Standards aus M6-T13b (Varianten der Farbregel, Abschluss nur mit Verbesserung, Wachsen bei 0,8 F_max). Neu: `--bildpunkt schwerpunkt --verzeichnung 0.01 --farb-querfarbe-strahlen` (D_max = 1 %, Gewicht 10⁴). Bewertung `tools/m6t14_tabelle.py` (`mikroptim eval --grid 256 --psf`, Referenz Austrittspupille, Felder 0/70/100 % des erreichten Feldes), RMS = größter RMS über 436–656 nm je Feld, einmal um den Schwerpunkt bei λ_ref und einmal um den idealen Punkt (wie M6-T13), Strehl = kleinster FFT-Strehl über 436–656 nm (lageunabhängig, für beide Bildpunkte gleich), Verzeichnung des Hauptstrahls bei 546 nm am Rand, Querfarbe = Spanne der Hauptstrahlhöhen über 436–656 nm am Rand (in Klammern bezogen auf den Airy-Radius 0,61 λ_ref/NA).

| Lauf | Start | Stufe | NA | Feld | Spektrum | El. | RMS 0/70/100 % Schwerpunkt | RMS 0/70/100 % ideal | Strehl min | Verzeichnung | Querfarbe [µm] | Zeit [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F3Z (M6-T13) | F3 | c1_apertur | 0,640 | 41,2 % | 405–707 | 16 | 0,026/0,079/0,108 | 0,026/0,101/0,121 | 0,66 | −0,42 % | 0,35 (0,67) | 7.948 |
| **F3ZS** | F3Z c1 | c0_feld | 0,640 | 50,6 % | 405–707 | 21 | 0,036/0,081/0,096 | 0,036/0,095/0,384 | 0,65 | −0,70 % | 0,41 (0,78) | |
| F3ZS | | c1_apertur | **0,650** | 50,6 % | 405–707 | 22 | 0,034/0,088/0,102 | 0,034/0,103/0,393 | 0,67 | −0,73 % | 0,40 (0,78) | |
| **F3ZS** | | c2_feld | **0,650** | **52,1 %** | 405–707 | 22 | 0,032/0,084/0,099 | 0,032/0,137/0,476 | **0,71** | −0,78 % | 0,38 (0,75) | 12.349 (c0–c2) |
| Z1f (M6-T13, H1G8) | Einzellinse | p1f | 0,50 | 10 % | 436–656 | 7 | 0,048/0,042/0,050 | 0,048/0,041/0,050 | 0,87 | −0,07 % | 0,09 (0,14) | |
| Z1I (Kontrolle, ideal) | Z1f | c0_feld | 0,50 | 25,6 % | 436–656 | 13 | 0,067/0,061/0,087 | 0,067/0,067/0,082 | 0,70 | −0,24 % | 0,26 (0,39) | |
| Z1I (Kontrolle, ideal) | | c1_feld | 0,50 | 55,2 % | 436–656 | 18 | 0,034/0,066/0,095 | 0,034/0,103/0,127 | 0,60 | −0,39 % | 0,37 (0,55) | 2.387 (c0–c1) |
| **Z1S** | Z1f | c0_feld | 0,50 | **56,8 %** | 436–656 | 18 | **0,029/0,061/0,078** | 0,029/0,138/0,436 | **0,81** | −0,70 % | 0,41 (0,62) | 4.689 |
| Z1SF (+ SYN-28) | Z1f | c0_feld | 0,50 | **66,2 %** | 436–656 | 18 | 0,039/0,076/0,097 | 0,039/0,235/0,735 | 0,63 | −1,00 % | 0,57 (0,85) | 4.352 |

Verlauf und Abbrüche:

- **F3ZS:** Feldstufe c0 41 → 50,6 % (M6-T13: Stillstand bei 41–43 %), Aperturstufe c1 0,640 → 0,650, Feldstufe c2 → 52,1 %; alle Runden enden mit `max_elements` (Budget 20, nach Rückkehr zu Geisterplatten 21–22). Die Stufe c3_apertur lief bei Abschluss des Berichts noch und ist nicht bewertet.
- **Z1S, Z1SF:** Feldstufe in vier Runden 10 → 57 % bzw. 66 % (18 El., Budget 18). Die nächste Stufe bricht ab: `mikroptim insert --ghost-return` meldet „return_to_ghost_plates: keine Lage passt“ (kein Luftspalt für die Wörterbücher, Befund außerhalb der Aufgabe, Abschnitt 7).
- **Z1I** (Kontrolle: dieselben Werkzeuge und M6-T13b-Standards, idealer Bildpunkt, ohne Verzeichnungsbedingung und Querfarbe aus Strahlen, nur Feldstufen): 25,6 % nach dem ersten, 55,2 % nach dem zweiten Zyklus, dann „fertig“. Gleiches Feld wie Z1S, aber RMS am Rand 0,095 statt 0,078 und Strehl 0,60 statt 0,81.
- **Ohne ideale Kippung gemessen** (Spalte „ideal“) sehen die neuen Systeme schlecht aus (0,4–0,7 Wellen am Rand): Das ist die zugelassene Verzeichnung (0,7–1 %), keine Bildverschlechterung (Strehl).

Die Läufe F3ZS und Z1S starteten vor dem Merge von M6-T13b mit dem Programm des Stands davor und den Optionen, die seit M6-T13b Standard sind, ausdrücklich (`--farb-varianten --wachsen-marge 0.8 --einfuegen-abschluss`, laut M6-T13b bitgleich zum neuen Standard); Z1I und Z1SF rechneten mit dem Programm nach dem Rebase.

## 4. Diagnose

- **Verzeichnung:** Mit dem Schwerpunkt als Bildpunkt lässt der Optimierer die Verzeichnung wachsen (−0,4 → −0,7 % bei F3ZS, −0,07 → −0,70 % bei Z1S); die Grenze 1 % wird erst bei 66 % Feld erreicht (Z1SF). Bei vollem Feld ist mit Verzeichnung über 1 % zu rechnen (∝ h²); der zulässige Wert ist eine Entscheidung des Maintainers (Spezifikation, offener Punkt 6).
- **Querfarbe** ist jetzt die Grenze am Rand: 0,38–0,57 µm (0,6–0,85 Airy-Radien). Die Farbregel mit `--color-lateral-rays` erkannte sie in 17 von 59 Entscheidungen (σ_lat 0,02–0,16 Wellen gegen Achse 0,07–0,09) und setzte Null-Kittglieder an die Stelle mit größtem |y·ȳ|; jedes Element mehr stößt aber an das Elementbudget, und in den Öffnungsstufen ab F3 wurden die meisten Einfügungen wieder verworfen („Einfügen verworfen“ 35-mal).
- **Seidel- gegen Strahlkriterium** (Orakeltest `ColorStage.LateralColorFromRealChiefRays`, Dreilinser ohne Farbkorrektur): dort entscheiden beide gleich (Achse 16 Wellen Längsfarbe); den Unterschied zeigen erst korrigierte Systeme hoher NA (Abschnitt 2: Seidel-ΔW020 ≈ −0,5 Wellen bei 0,027 Wellen realem Achs-RMS).
- **Dichtes Flint:** SYN-28 als Zusatzplatte an den Lagen der Rückkehr veränderte den Weg (66 % Feld), wurde aber nicht zum Element; das Ergebnis ist schlechter korrigiert. Kein Grund, es ins Wörterbuch zu nehmen.
- **Elementbudget:** In allen Läufen endet die Stufe an `max_elements`; der Rückbau bei 85 % entfernte nichts.

## 5. Tests

| Test | Orakel / Ergebnis |
|---|---|
| `ReferencePoint.ChiefRayHeightThroughPlatesIsAnalytic` | Planplatten: ȳ = Σ t_i tan θ_i, n_i sin θ_i = sin θ, bei λ_ref und jeder Wellenlänge (Dispersion), Schwerpunkt = Hauptstrahl (paralleles Bündel); ≤ 10⁻¹² relativ (`kChiefRayAnalytic`) |
| `ReferencePoint.DistortionOfConcentricSystemIsAnalytic` | zur Blende konzentrisches System: Hauptstrahl unabgelenkt, ȳ = L tan θ, h = f tan θ, D = δ/f für die um δ verschobene Bildebene (0 im Brennpunkt); Residuum √w max(0, \|D\| − D_max), Achse und fremdes Feld 0 |
| `ReferencePoint.MeritEqualsIdealMeritAtChiefRayHeight` | Residuen und Merit mit `chief_ray`/`centroid` **bitgleich** zum idealen Punkt an der Höhe ȳ_f (`image_infinite`, `image_exit_pupil`); Achse unverändert |
| `ReferencePoint.ApplyUpdatesChiefRayHeightLikeRebuild` | Batch nach `apply` bitgleich zum Neuaufbau aus der geänderten Datei |
| `ReferencePoint.FdJacobianContainsChiefRayDerivative` | FD-Jacobi gegen die Kettenregel ∂r/∂x\|_ȳ + (∂r/∂ȳ)(∂ȳ/∂x) aus drei eigenen Differenzen: 8,4·10⁻⁵ bei max \|J\| = 274 (3·10⁻⁷ relativ, Toleranz 10⁻⁵) |
| `ReferencePoint.EvalKeepsDistortionRelativeToIdealPoint` | EVAL: Verzeichnung, Hauptstrahl, Schwerpunkt der Kennzahlen gleich wie mit idealem Punkt |
| `ReferencePoint.FileFormatAndCompatibility` | Lesen/Schreiben, Fehlerpfade (`reference_point`, `max_distortion`, Modus, doppelt), `ideal` = ohne Feld bitgleich |
| `ReferencePoint.AdjointJacobianRejectsChiefRay` | `make_adjoint_batch`, `adjoint_jacobian_cpu` werfen |
| `CudaResiduals.MatchCpuWithChiefRayReference` | RESIDUALS gegen CPU für je 500 gestörte Varianten (Einzellinse, Mehrlinser mit `image_exit_pupil`, Geisterplatte), max \|dr\| 1,6·10⁻¹¹ mm (RTX 3090) |
| `ColorStage.LateralColorFromRealChiefRays` | Empfehlung gegen Orakel aus Hauptstrahlhöhen und Achs-RMS (vier Feldwinkel) |
| `CliOptimize.ReferencePointAndDistortionOptions` | JSON/Text/Ergebnisdatei, adjungiert abgelehnt, Fehlbedienung Exit 2 |
| Python `test_reference_point_and_distortion`, `test_synth_options_lateral_rays` | Skriptschalter |

Lokal: `ctest --preset release-cuda` (RTX 3090) 1154/1154; GCC-Prüfbau (MSYS2, Warnungen als Fehler) 959/959, `test_cuda_residuals.cpp` mit g++ `-fsyntax-only` geprüft; Python-Tests (eigenes Wheel) 317 bestanden, 55 übersprungen.

## 6. Vergleich mit M6-T13

| Größe | M6-T13 | M6-T14 |
|---|---|---|
| höchste NA polychromatisch | F3Z: 0,640 bei 41 %, 16 El., Strehl 0,66 | **F3ZS: 0,650 bei 52 %, 22 El., Strehl 0,71** |
| größtes Feld, Strehl ≥ 0,8, 436–656 nm | Z1f: NA 0,5, 10 %, 7 El. | **Z1S: NA 0,5, 57 %, 18 El., Strehl 0,81** |
| Feldstufe ab dem 7-Linser | 19 % (H1G8) | 57 % (Z1S), 66 % (Z1SF); Kontrolle ideal 55 % nach zwei Zyklen, Strehl 0,60 |
| Grenze am Rand | Kippung (Verzeichnung + Querfarbe) | Querfarbe 0,4–0,6 µm, Elementbudget |
| `--color-module-lateral` | griff nie | Querfarbe aus Strahlen in 17 von 59 Entscheidungen |

## 7. Befunde

| Nr. | Stand nach M6-T14 |
|---|---|
| F20 | gelöst für die Lage des Bildes (Schwerpunkt als Bildpunkt, Verzeichnung als Bedingung); die Querfarbe bleibt Grenze |
| F24 (neu) | Bei hoher NA liegen Hauptstrahl und Schwerpunkt bis zu ein halbes Airy-Scheibchen auseinander; die Diagnose „Kippung = Verzeichnung“ aus M6-T13 war nur teilweise richtig |
| F25 (neu) | `insert --ghost-return` findet bei 18 El. keine Lage („keine Lage passt“) und beendet den Zyklus nach der Feldstufe (Z1S, Z1SF); außerhalb dieser Aufgabe |
| F26 (neu) | Alle Öffnungsstufen enden am Elementbudget (20 bzw. 18); der Rückbau bei 85 % entfernt nichts |

## 8. Fragen an den Maintainer

1. **Zulässige Verzeichnung** (Spezifikation, offener Punkt 6): gemessen mit D_max = 1 % (Gewicht 10⁴). Welcher Wert soll gelten? Bei vollem Feld ist mehr als 1 % zu erwarten.
2. **Standard des Skripts:** Soll `--bildpunkt schwerpunkt --verzeichnung <D_max> --farb-querfarbe-strahlen` Standard von `tools/m6t6_zielspezifikation.py` werden (Bibliothek und CLI bleiben bei `ideal`, ADR-0019 Punkt 7)?
3. `centroid` statt des im Auftrag genannten Hauptstrahls (Abschnitt 2) – einverstanden? `chief_ray` ist umgesetzt und getestet, aber für die Synthese ungünstig.
4. Folgeaufgaben: adjungierte Jacobi mit ∂ȳ/∂x (Rang-1-Korrektur), F25 (Rückkehr zu Geisterplatten bei vielen Elementen), Elementbudget für die Öffnungsstufen.

## 9. Dateien

```
OPT="--farb-stetig --farb-ziel voll --farb-modul doublet --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85"
T14="--bildpunkt schwerpunkt --verzeichnung 0.01 --farb-querfarbe-strahlen"
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t13/F3Z/c1_apertur_z2r_s2.json --elementbudget 20 --farb-budget 6 $OPT $T14 --out docs/ergebnisse/m6-t14/F3ZS
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t13/H1G8/z1f_s2.json --elementbudget 18 --farb-budget 6 $OPT $T14 --out docs/ergebnisse/m6-t14/Z1S
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld --zyklus-runden 4 --start docs/ergebnisse/m6-t13/H1G8/z1f_s2.json --elementbudget 18 --farb-budget 6 $OPT --out docs/ergebnisse/m6-t14/Z1I
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld --zyklus-runden 4 --start docs/ergebnisse/m6-t13/H1G8/z1f_s2.json --elementbudget 18 --farb-budget 6 $OPT $T14 --woerterbuch-glas SYN-28 --out docs/ergebnisse/m6-t14/Z1SF
python tools/m6t14_tabelle.py NAME=SYSTEM.json ... --out DIR --gpu --json docs/ergebnisse/m6-t14/m6t14_tabelle.json
```

Die Verzeichnisse sind mit `--archiv` verkleinert; `m6t14_tabelle.json`/`.md` enthalten die Tabelle aus Abschnitt 3.
