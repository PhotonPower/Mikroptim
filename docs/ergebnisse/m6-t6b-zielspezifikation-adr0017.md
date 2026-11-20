# M6-T6b: Lauf auf der Zielspezifikation mit der Referenzsphäre um den Bildpunkt (ADR-0017)

Wiederholung von M6-T6 ([Bericht](m6-t6-zielspezifikation.md), [Meilensteinplan](../plan/meilensteine.md#m6-t6b--lauf-auf-der-zielspezifikation-mit-adr-0017)) mit der neuen Konvention der Referenzsphäre `image_infinite` ([ADR-0017](../adr/0017-referenzsphaere-bildpunkt.md), [Bericht M6-T7](m6-t7-referenzsphaere.md)). Ziel: Plan-Apochromat 50x/0,95, volles Feld (±4,345°), 436–656 nm beugungsbegrenzt ([spezifikation.md](../spezifikation.md)). Alle Zahlen stammen aus Läufen auf dem Rechner des Maintainers (RTX 3090, geteilt mit anderen Agenten und einem Sprachmodell, GPU-Last fast immer 99 %), Release-Build mit CUDA 12.9 vom Stand `04e1ea6`. Annahmen (Arbeitsabstand, Durchmesser, Abgleichlänge, Gewichte, Gläser) unverändert wie in M6-T6, Abschnitt 1; Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json` – die Ergebnisse sind nicht herstellbar.

**Kurzfassung.** Das Ziel wird weiterhin **nicht** erreicht. Mit der neuen Referenzsphäre sind Merit und Bewertung jetzt dieselbe Größe (Abweichung ≤ 1 %), die Falle F12 tritt in keinem Lauf auf, und die Ergebnisse sind ehrlich:

- **Bestes beugungsbegrenztes System** (Kandidat **V6-Z2**, monochromatisch): 9 Elemente, NA 0,50, volles Feld, 546 nm, RMS **0,015/0,024/0,033 Wellen** bei 0/70/100 % Feld, Strehl (FFT) 0,99/0,98/0,96 – derselbe Aufbau wie Z2 aus M6-T6 (0,034/0,043/0,057), unter der neuen Merit nachoptimiert.
- **Höchste beugungsbegrenzte NA** (Kandidat **V2-M1**, monochromatisch): **NA 0,855 bei 10 % Feld mit 5 Elementen**, RMS 0,048/0,047/0,072 Wellen bei 0/7/10 % Feld, Strehl 0,91/0,92/0,83 („Apertur zuerst“; M6-T6: Falle F12, M6-T7: NA 0,4725).
- **Bestes polychromatisches System** (Kandidat **V3**, „Farbe früh“, Farbregel): 12 Elemente, NA 0,535, **27 % Feld**, 405–707 nm: auf der Achse 0,016–0,029 Wellen (436–656 nm), größter Wert 0,108 Wellen (436 nm am Rand des Teilfelds), Strehl (FFT) ≥ 0,63, für 486–656 nm ≥ 0,89. M6-T6: kein polychromatisches System unter 0,43 Wellen.
- **Bestes polychromatisches System bei vollem Feld** (Kandidat **V7**, Start aus dem M6-T6-Zwischenstand bei NA 0,5, „Farbe früh“): 11 Elemente, NA 0,50, 405–707 nm, 0,03–0,26 Wellen (436–656 nm), Strehl (FFT) 0,28–0,97. M6-T6-Endsystem: 1,4–59 Wellen.
- **Neuer Engpass (F14):** Ab der Einzellinse bleibt die Feldstufe in allen Läufen bei **27 % Feld** stehen (F_B knapp über der Schwelle, keine Geisterplatte aktiv), unabhängig von NA (0,4 oder 0,5), Elementzahl (3–10) und erzwungenem Einfügen. Derselbe 9-Linser aus M6-T6 ist unter der neuen Merit bei vollem Feld sogar besser als damals: Der Engpass ist der Weg des Controllers, nicht die Merit.
- **Laufzeit:** neun Varianten mit zusammen 10,7 h Programmlaufzeit (Summe der Programmaufrufe, je 0,4–2,6 h, bis zu sechs Läufe gleichzeitig auf der geteilten GPU), Wanduhr 3,1 h (22:39–01:43).
- **M11-T7 nachgemessen:** `--global cma` ist an einem neuen Zwischenstand bei **allen vier** Startwerten besser als `starts` mit gleichem Budget (Abschnitt 6).

## 1. Änderungen am Skript

[`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py):

| Änderung | Zweck |
|---|---|
| `--referenzsphaere K`, **Standard `image_infinite`**; Hauptlauf, Vergleichszweig und OSC-Versuch setzen das Feld `reference_sphere` der Startdatei ausdrücklich (auch `last_surface`, bitgleich zu einer Datei ohne Feld, ohne die Warnung aus #102) | neue Läufe mit ADR-0017; die Läufe von M6-T6 und M11-T4 rechnet `--referenzsphaere last_surface` nach (Befehle in den Berichten ergänzt, `aktualisieren_und_testen.sh --m6t6-dokumentiert` übergibt es) |
| F12-Schutz (`paraxial_focus` im OSC-Versuch) nur noch bei `last_surface` standardmäßig an; `--mit-f12-schutz` erzwingt ihn (Lauf E von M6-T7) | M6-T7: mit der neuen Konvention entbehrlich |
| **Fehler behoben:** `eval --psf`/`--vector` brach bei vollem Feld eines bei kleinem Feld entworfenen Systems mit „NA vorgeben“ ab und beendete das Skript (M6-T7, Vergleichszweig). Jetzt wird mit `--na` = Pupillenradius/Brennweite wiederholt und eine Notiz `<name>_<key>_na.txt` geschrieben; ein fehlgeschlagenes Ergebnis bleibt als `.fehler` liegen, damit eine Fortsetzung es nicht als vorhanden überspringt | Test `python/tests/test_tool_m6t6.py`; im Lauf an V4 (`p8_vollfeld_psf`, `p8_vollfeld_vektor`) aufgetreten und abgefangen |
| Bewertung mit Referenz `infinite` (wie die Merit) statt `vertex`, wenn die Datei `image_infinite` hat (`<name>_unendlich.json`) | die Referenz durch den Scheitel ist für die neue Merit bedeutungslos |
| `--sinusbedingung W` (reicht `synth --sine-condition W` durch) | Variante V4 |
| `--erzwungen-min-spalt G`, `--erzwungen-blende`, `--erzwungen-mehr N` (Standard wie M6-T6: 0,7 mm, ohne Blendenspalt, 0) | Variante V5 gegen den Stillstand ohne Spalt |
| Zwischenkandidat Z2 am **erreichten** Feldanteil der Feldstufe (bisher immer mit den Feldwinkeln 100 %; in M6-T6 gleichbedeutend, weil die Feldstufe 100 % erreichte) | Z2 ist sonst bei 27 % Feld nicht sinnvoll |
| Zusammenfassung `m6t6_lauf.json` mit Referenzsphäre, synth-Optionen, NA der Feldstufe und Einstellungen des erzwungenen Einfügens; `--archiv` behält `lauf.log`, die NA-Notizen und die monochromatischen Kandidaten `m*` | Nachvollziehbarkeit |

## 2. Varianten und Begründung

Ausgang ist der M6-T6-Ablauf (Stufen p1 Apertur bei 10 % Feld → p2 Feld 10 → 100 % → p3 Apertur → 0,95 → p4/p5 Farbe → p6 Feinoptimierung ohne Geisterplatten, 405/700 nm mit Gewicht 0,1 → p7 Glas-Einrasten → p8 Bewertung, dazu Z2 = Ende von p2 monochromatisch nachoptimiert). Alle Läufe mit `--gpu`, `image_infinite`, 16 Ringen × 8 Speichen, FD-Jacobi, FP64.

| Variante | Optionen | Begründung |
|---|---|---|
| V1 Basis | – (`synth --stall 0`) | direkter Vergleich mit M6-T6 („Feld vor Apertur“ bei NA 0,5) |
| V2 Apertur zuerst | `--na-feld 0,95` | M6-T7: Apertur zuerst erreicht NA 0,95 ohne F12; Feld danach bei hoher NA |
| V3 Farbe früh | `--farbregel --farbe-frueh --synth-arg "--stall 0"` | F13: Farbe einführen, solange das System Abstand zur Schwelle hat (Farbregel M11-T5c, Operator B) |
| V4 Apertur zuerst + Sinusbedingung | `--na-feld 0,95 --sinusbedingung 327,5` | F4: Koma bei 10 % Feld unsichtbar; Gewicht = Koma-Äquivalent bei 50 % Feld (`osc_weight_for_field(0,5)`) |
| V5 Basis + mehr erzwungenes Einfügen | `--erzwungen-min-spalt 0,5 --erzwungen-blende --erzwungen-mehr 3` | V1 stand in p2 „ohne Spalt für ein erzwungenes Einfügen“ |
| V6 Start aus M6-T6-p2 | `--start docs/ergebnisse/m6-t6/p2_final.json` | trennt Weg und Merit: der 9-Linser von M6-T6 bei NA 0,5 und vollem Feld unter der neuen Merit |
| V7 wie V6 + Farbe früh | V6 + V3-Optionen | Empfehlung 5 des M6-T6-Berichts: aus Z2 polychromatisch weiter |
| V8 wie V7, Farbbudget 3 | V7 + `--farb-budget 3` | mehr Elemente in den Farbstufen |
| V9 Feld bei NA 0,4 | `--na-feld 0,4` | F5: Reserve für die Feldstufe |

**Nicht eingesetzt** (begründet): Stagnationsabbruch (`--stall`; cli.md: auf der Zielspezifikation schädlich, Aperturstufe bei NA 0,37 bzw. Einfüge-Serien), Augmented Lagrangian (ADR-0018: allein ohne Gewinn, nur mit `--stall` messbar besser; mit der Strahlsuche nicht kombinierbar), toleranzbewusste Merit (für späte Stufen; die Synthese erreicht die Spezifikation noch nicht), stetige Spektrum-Homotopie (M6-T6: ohne Wirkung belegt, solange das monochromatische System an der Schwelle liegt).

## 3. Ablauf der Varianten

„erreicht“ = letzter Punkt der Stufe (NA, Feldanteil); Elemente = Glaselemente mit Geisterplatten-Hilfsebenen; Zeit = Summe der Programmaufrufe der Variante (geteilte GPU).

| Variante | p1 (Apertur, 10 % Feld) | p2 (Feld) | p3 (Apertur) | p4/p5 (Farbe) | Ende (p6/p7) | Elemente am Ende | Zeit [s] |
|---|---|---|---|---|---|---|---|
| V1 | NA 0,50, 5 El., Ziel | **0,272**, 6 El., Stillstand, kein Spalt | 0,505 | Budget, 8 El. | NA 0,505, 27 % | 8 | 2.732 |
| V2 | **NA 0,855**, 5 El., Stillstand | 0,10, 18 El. (Einfüge-Serie nach μ/2) | 0,855, 21 El. | 23 El. | NA 0,855, 10 % | 23 | 9.204 |
| V3 | wie V1 | wie V1 (0,272) | nach den Farbstufen: 0,535, 12 El. | 7 → 8 El., je Start abgelehnt | NA 0,535, 27 %, eingerastet | 12 | 4.895 |
| V4 | NA 0,4575, 6 El. | 0,10, 8 El. | **0,80**, 12 El. | 14 El. | NA 0,80, 10 % | 14 | 6.704 |
| V5 | wie V1 | 0,272, 10 El. (5 erzwungen) | 0,6225, 15 El. (6 erzwungen) | 18 El. | NA 0,6225, 27 % | 18 | 3.728 |
| V6 | (Start 9 El., NA 0,5) | 1,00 (Ziel) | 0,565, 12 El. | 14 El. | NA 0,565, 100 % | 14 | 2.301 |
| V7 | wie V6 | wie V6 | nach den Farbstufen: Start abgelehnt | 10 → 11 El. | NA 0,50, 100 %, eingerastet | 11 | 1.387 |
| V8 | wie V6 | wie V6 | 24 El. (Budget) | 12 → 15 El. | NA 0,50, 100 %, eingerastet | 24 | 3.193 |
| V9 | NA 0,40, 3 El. | **0,272**, 3 El. | 0,6325, 26 El. | 28 El., je Start abgelehnt | NA 0,6325, 27 %, eingerastet | 28 | 4.240 |

Eine Störung trat auf: V5, `p1_r0` endete nach 602 s mit Exit 1 (GPU-Konkurrenz); die Wiederholung lief durch und traf dieselben Entscheidungen wie V1/V3 (deterministisch, gleiche Zahlen bis p1). Alle Stillstände der Synthese enden mit „keine Geisterplatte aktiv (kein Bedarf)“ bei F_B knapp über F_max = 5,1·10⁻³ Wellen² (F3), danach gibt das Skript die Spalte frei, halbiert μ einmal und fügt erzwungen ein. Von 981 Apertur-, Feld- und Spektrumschritten endeten 417 (43 %) mit `max_iterations` (F10).

**Fokuslage (F12):** In allen Stufen aller Varianten liegt der paraxiale Fokus in der Bildebene (hintere Schnittweite gleich dem Abstand letzte Fläche – Objekt auf ≤ 25 µm, `--tabellen`), ohne die Hilfsbedingung `paraxial_focus`.

## 4. Kandidaten

RMS am größten Wert über die Felder, Bewertung `mikroptim eval --grid 256 --reference exit-pupil` (Wellen bei der Wellenlänge des Tupels; die Referenz `infinite` wie die Merit weicht ≤ 1 % ab), Strehl über die FFT (skalar).

| Kandidat | Elemente | NA | Feld | Spektrum | größter RMS 436–656 nm [Wellen] | kleinster Strehl (FFT) 436–656 nm | beugungsbegrenzt (≤ 0,0714) |
|---|---|---|---|---|---|---|---|
| **V6-Z2** (M6-T6-p2, ohne Geisterplatten nachoptimiert) | 9 | 0,50 | 100 % | 546 nm | **0,033** | 0,957 | ja (monochromatisch) |
| V1-Z2 (= V3-Z2) | 6 | 0,50 | 27 % | 546 nm | 0,050 | 0,953 | ja (monochromatisch, Teilfeld) |
| V4-Z2 | 8 | 0,4575 | 10 % | 546 nm | 0,060 | 0,941 | ja (monochromatisch, Teilfeld) |
| **V2-M1** (Ende p1, nachoptimiert) | 5 | **0,855** | 10 % | 546 nm | **0,072** | 0,828 | Grenze (monochromatisch, Teilfeld) |
| V4-M3 (Ende p3, nachoptimiert, mit Sinusbedingung) | 12 | 0,80 | 10 % | 546 nm | 0,091 | 0,853 | nein (knapp) |
| **V3** Endsystem | 12 | 0,535 | 27 % | 405–707 nm | **0,108** (436 nm, Rand) | 0,632 | nein (knapp; 486–656 nm ≤ 0,086) |
| V5 Endsystem | 18 | 0,6225 | 27 % | 405–707 nm | 0,200 | 0,368 | nein |
| **V7** Endsystem | 11 | 0,50 | **100 %** | 405–707 nm | 0,259 | 0,276 | nein |
| V1 Endsystem | 8 | 0,505 | 27 % | 405–707 nm | 0,275 | 0,279 | nein |
| V2 Endsystem | 23 | 0,855 | 10 % | 405–707 nm | 5,6 | 0,004 | nein |
| V6 Endsystem | 14 | 0,565 | 100 % | 405–707 nm | 6,1 | 0,001 | nein |
| V4 Endsystem | 14 | 0,80 | 10 % | 405–707 nm | 6,2 | 0,000 | nein |
| V8 Endsystem | 24 | 0,50 | 100 % | 405–707 nm | 11,6 | 0,000 | nein |
| V9 Endsystem | 28 | 0,6325 | 27 % | 405–707 nm | 1,28 | 0,038 | nein |
| V9-Z2 | 3 | 0,40 | 27 % | 546 nm | 0,040 | 0,979 | ja (monochromatisch, Teilfeld) |

Bei vollem Feld bewertet (die Teilfeld-Kandidaten sind dafür nicht entworfen): V3 0,03/2,1/6,7 Wellen bei 0/70/100 %, V2-M1 0,05/4,2/9,7 Wellen, V4-M3 0,06/15/58 Wellen (4.042 versagte Strahlen am Rand; die PSF lief über `--na`).

### V6-Z2 – bestes beugungsbegrenztes System (monochromatisch, volles Feld)

`docs/ergebnisse/m6-t6b/V6_start_z2alt/z2_s0.json`: 9 Elemente, f′ = 3,29 mm, Arbeitsabstand 0,250 mm, Petzvalradius −130,0 mm; Aufbau wie Z2 im [M6-T6-Bericht](m6-t6-zielspezifikation.md#4-kandidaten-und-endsystem), mit der neuen Merit nachoptimiert (2.000 Zyklen).

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 546,07 | 0,0153 | 0,0153 | 0,991 | 0,991 | 0,991 | – | 0 |
| 3,045 | 546,07 | 0,0241 | 0,0239 | 0,977 | 0,978 | 0,978 | 0,022 | 0 |
| 4,345 | 546,07 | 0,0333 | 0,0332 | 0,957 | 0,957 | 0,957 | -0,033 | 0 |

### V2-M1 – höchste beugungsbegrenzte NA (monochromatisch, 10 % Feld)

`docs/ergebnisse/m6-t6b/V2_apertur_zuerst/m1_s0.json`: Ende der Stufe p1 von V2 (Apertur zuerst bei 10 % Feld) ohne Geisterplatten und Hilfsebenen, 2.000 Zyklen monochromatisch nachoptimiert; 5 Elemente.

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 546,07 | 0,0481 | 0,0481 | 0,913 | 0,913 | 0,912 | – | 0 |
| 0,304 | 546,07 | 0,0466 | 0,0448 | 0,918 | 0,924 | 0,923 | -0,086 | 0 |
| 0,435 | 546,07 | 0,0718 | 0,0699 | 0,816 | 0,828 | 0,827 | -0,091 | 0 |

### V3 – bestes polychromatisches System (27 % Feld)

`docs/ergebnisse/m6-t6b/V3_farbe_frueh/p7.json`: 12 Elemente nach dem Glas-Einrasten (k = 3, 27 Kombinationen; Gläser CAF2, SYN-26, SYN-29, SYN-30, SYN-31, SYN-36), f′ = 3,29 mm, Arbeitsabstand 0,303 mm, Blende bis Objekt 19,25 mm, Petzvalradius −130,0 mm. NA 0,535, Feldwinkel 0/0,828/1,181° (27 % von 0/70/100 %).

| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) | Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |
|---|---|---|---|---|---|---|---|---|
| 0,000 | 404,66 | 0,0536 | 0,0536 | 0,893 | 0,894 | 0,894 | – | 0 |
| 0,000 | 435,83 | 0,0219 | 0,0219 | 0,981 | 0,981 | 0,981 | – | 0 |
| 0,000 | 486,13 | 0,0158 | 0,0158 | 0,990 | 0,990 | 0,990 | – | 0 |
| 0,000 | 546,07 | 0,0159 | 0,0159 | 0,990 | 0,990 | 0,990 | – | 0 |
| 0,000 | 587,56 | 0,0184 | 0,0184 | 0,987 | 0,987 | 0,987 | – | 0 |
| 0,000 | 656,27 | 0,0290 | 0,0290 | 0,967 | 0,967 | 0,967 | – | 0 |
| 0,000 | 706,52 | 0,0414 | 0,0414 | 0,934 | 0,934 | 0,934 | – | 0 |
| 0,828 | 404,66 | 0,1750 | 0,1140 | 0,299 | 0,591 | 0,591 | -0,590 | 0 |
| 0,828 | 435,83 | 0,0833 | 0,0736 | 0,761 | 0,806 | 0,806 | -0,633 | 0 |
| 0,828 | 486,13 | 0,0374 | 0,0373 | 0,946 | 0,947 | 0,947 | -0,501 | 0 |
| 0,828 | 546,07 | 0,0229 | 0,0227 | 0,979 | 0,980 | 0,980 | -0,255 | 0 |
| 0,828 | 587,56 | 0,0285 | 0,0226 | 0,968 | 0,981 | 0,981 | -0,083 | 0 |
| 0,828 | 656,27 | 0,0506 | 0,0286 | 0,904 | 0,969 | 0,969 | 0,179 | 0 |
| 0,828 | 706,52 | 0,0688 | 0,0378 | 0,830 | 0,945 | 0,945 | 0,350 | 0 |
| 1,181 | 404,66 | 0,1990 | 0,1598 | 0,210 | 0,346 | 0,346 | -0,765 | 0 |
| 1,181 | 435,83 | 0,1077 | 0,1071 | 0,633 | 0,632 | 0,631 | -0,809 | 0 |
| 1,181 | 486,13 | 0,0857 | 0,0532 | 0,748 | 0,894 | 0,894 | -0,678 | 0 |
| 1,181 | 546,07 | 0,0603 | 0,0314 | 0,866 | 0,962 | 0,963 | -0,434 | 0 |
| 1,181 | 587,56 | 0,0416 | 0,0310 | 0,934 | 0,965 | 0,964 | -0,262 | 0 |
| 1,181 | 656,27 | 0,0366 | 0,0346 | 0,948 | 0,957 | 0,956 | -0,002 | 0 |
| 1,181 | 706,52 | 0,0548 | 0,0399 | 0,888 | 0,941 | 0,941 | 0,168 | 0 |

Der Unterschied „mit/ohne Kippung“ am Rand ist Farbquerfehler und Verzeichnung (bis −0,8 %), die die Merit mit dem idealen Bildpunkt f·tan θ bestraft.

### V7 – polychromatisch bei vollem Feld

`docs/ergebnisse/m6-t6b/V7_start_z2alt_farbe_frueh/p7.json`: 11 Elemente nach dem Einrasten, f′ = 3,293 mm, Arbeitsabstand 0,300 mm, Blende bis Objekt 12,46 mm, größter Halbdurchmesser 3,5 mm, Petzvalradius −129,9 mm. RMS [Wellen] (Referenz Austrittspupille):

| Feld [°] | 405 | 436 | 486 | 546 | 588 | 656 | 707 |
|---|---|---|---|---|---|---|---|
| 0,000 | 0,262 | 0,028 | 0,131 | 0,088 | 0,033 | 0,058 | 0,114 |
| 3,045 | 0,586 | 0,154 | 0,115 | 0,083 | 0,082 | 0,171 | 0,239 |
| 4,345 | 0,629 | 0,188 | 0,259 | 0,185 | 0,105 | 0,120 | 0,203 |

Strehl (FFT) 0,28–0,97 für 436–656 nm. Ohne Kippung am Rand 0,07–0,18 Wellen: Der größte Anteil ist Farbquerfehler.

Vollständige Flächentabellen: `p7.json` bzw. `z2_s0.json`, `m1_s0.json` der Varianten (Syntheseergebnisse aus frei erfundenen Testgläsern).

## 5. Vergleich mit M6-T6 und M6-T7

| Größe | M6-T6 (`last_surface`) | M6-T7 (#99, `image_infinite`) | M6-T6b (`image_infinite`) |
|---|---|---|---|
| bestes beugungsbegrenztes System | Z2: 9 El., NA 0,5, volles Feld, 546 nm, 0,034/0,043/0,057 Wellen, Strehl ≥ 0,88 | – | **V6-Z2**: derselbe Aufbau, 0,015/0,024/0,033 Wellen, Strehl ≥ 0,957 |
| ab der Einzellinse, Feld vor Apertur | volles Feld bei NA 0,5 mit 9 El. (Merit mit Referenz durch die Hilfsebene) | – | **27 % Feld** (V1, V3, V5, V9; F14) |
| Apertur zuerst, 10 % Feld | NA 0,94 nur über die Falle F12 (real 21 Wellen) | NA 0,4725, 4 El., 0,048/0,061/0,068 Wellen (Strahlsuche, ohne erzwungenes Einfügen) | **NA 0,855, 5 El.**, 0,048/0,047/0,072 Wellen (V2-M1; mit erzwungenem Einfügen) |
| Aperturstufe bei vollem Feld | NA 0,57 (12 El.) | – | NA 0,565 (V6, 12 El.) – praktisch gleich |
| Farbe | Endsystem 22 El., 1,4–59 Wellen | – | V3 (27 % Feld) ≤ 0,108 Wellen, V7 (volles Feld) ≤ 0,26 Wellen; Farbe spät (V1, V6) 0,28 bzw. 6 Wellen |
| Übereinstimmung Merit – Bewertung | Faktor 1,3–2 (F11) | ≤ 1,6 % auf der Achse | ≤ 1 % (Referenz `infinite` gegen `exit-pupil`) |

Folgerung: Mit der neuen Referenzsphäre ist der Engpass nicht mehr die Apertur (F12), sondern das **Feld ab einem bei kleinem Feld entworfenen System** (F14) und weiterhin die Farbe, wenn sie spät kommt (F13). Ein System, das das Feld schon trägt (V6/V7), bleibt in der Aperturstufe bei NA ≈ 0,57 stehen wie in M6-T6.

## 6. M11-T7: `--global cma` gegen `starts` an einem neuen Zwischenstand

Zwischenstand `docs/ergebnisse/m6-t6b/m11t7/zwischenstand.json` (V1, Ende der Feldstufe, ohne Geisterplatten nachoptimiert, aber mit den Feldwinkeln 100 % statt der erreichten 27 % – ein schwerer Punkt: F₀ = 0,104 Wellen², LM steht). Wie in [optimierung.md §15](../optimierung.md#15-memetische-globale-suche-m11-t7): GPU, 16 × 8 Strahlen, σ₀ = 0,02, `cma,generations=8,iterations=50` gegen `starts,budget=5200,generations=100,iterations=50` (gleiches Budget, je 5.200 Zyklen verbraucht). Ergebnis `m11t7/m11t7_vergleich.json`:

| Startwert | 1 | 2 | 3 | 4 | Mittel |
|---|---|---|---|---|---|
| CMA [Wellen²] | **0,0908** | **0,0905** | **0,0952** | **0,0853** | **0,0904** |
| `starts` [Wellen²] | 0,0996 | 0,1009 | 0,1006 | 0,0997 | 0,1002 |

Hier ist CMA bei **jedem** Startwert besser (Vorteil 5–15 %), anders als am M6-T6-Zwischenstand a2 (nur im Mittel). Laufzeit je 4–7 s. Das Abnahmekriterium von M11-T7 („Bericht an M6-T6-Zwischenständen“) ist damit an einem zweiten Zwischenstand belegt.

## 7. Befunde F1–F13 aus M6-T6

| Nr. | Befund (M6-T6) | Stand M6-T6b | Beleg |
|---|---|---|---|
| F1 | Sattelstarts außerhalb der Krümmungsschranken | **erledigt** (seit M6-T6) | kein Auftreten in neun Läufen |
| F2 | feste Luftspalte nach dem Einfügen | **weiterhin offen** im Controller; das Skript gibt die Spalte zwischen den synth-Läufen frei (`synth_loop`) | jede Runde mit Nachoptimierung der Spalte |
| F3 | kein Bedarf trotz F_B > F_max | **weiterhin offen, häufigster Abbruch** | alle Stillstände „keine Geisterplatte aktiv“ knapp über F_max; μ/2 löst in V2 eine Einfüge-Serie aus (5 → 18 El., F bis 3,2 Wellen²) |
| F4 | Feldfehler für die Geisterplatten unsichtbar | **teilweise**: die Sinusbedingung hält die Koma klein (V4-M3: Feldzuwachs ohne Kippung ≤ 0,003 Wellen bis 10 % Feld), das volle Feld bleibt nach Apertur zuerst unerreichbar (V2, V4: Feldstufe bleibt bei 10 %) | Abschnitte 3, 4 |
| F5 | Petzval und Feld, keine Reserve | **weiterhin offen**, verschärft als F14; NA 0,4 als Reserve (V9) ändert nichts | V9 |
| F6 | Pupillenrand bei NA 0,95 | **umgangen** (16 Ringe); bei NA 0,855 Merit und 256er-Raster in Übereinstimmung (V2-M1) | – |
| F7 | Blende weit hinter der Optik | **offen** (Annahme, Maintainer) | – |
| F8 | Pfadabhängigkeit | **weiterhin**: drei Varianten mit identischen Stufen p1/p2 liefern bitgleiche Zwischenstände (deterministisch), kleine Änderungen (μ, erzwungenes Einfügen) führen zu ganz anderen Wegen | V1/V3/V5 |
| F9 | Operator B wird nicht empfohlen | **teilweise erledigt** (Farbregel M11-T5c): in V7 fügt der Controller selbst ein Kittglied ein (Operator B), in V3 das erzwungene Teilen; ohne Farbregel weiter nur Operator A | Protokolle `p4_r*`, `p5_r*` |
| F10 | LM endet mit `max_iterations` | **weiterhin offen**: 43 % der Schritte | Abschnitt 3 |
| F11 | Merit und Bewertung mit verschiedener Referenz | **erledigt** (ADR-0017): `infinite` gegen `exit-pupil` ≤ 1 % | `*_unendlich.json` gegen `*_pupille.json` |
| F12 | Falle der Referenzsphäre | **erledigt** (ADR-0017): Fokus in allen Stufen in der Bildebene, ohne `paraxial_focus` | `--tabellen` |
| F13 | Farbe nach dem monochromatischen Aufbau | **teilweise erledigt**: „Farbe früh“ bei NA 0,5 (V3, V7) liefert fast beugungsbegrenzte polychromatische Systeme (≤ 0,11 bzw. 0,26 Wellen); spät (V1, V5, V6) weiter 0,2–6 Wellen; mehr Budget in den Farbstufen (V8) verschlechtert (24 El., 11,6 Wellen). Ungültige Brechzahl im LM trat nicht auf | Abschnitt 4 |

**Neue Befunde:**

| Nr. | Befund | Beobachtung | Empfehlung |
|---|---|---|---|
| F14 | **Feldstufe ab der Einzellinse endet bei 27 % Feld** | In V1, V3, V5 (NA 0,5, 6–10 El., 0–5 erzwungene Einfügungen) und V9 (NA 0,4, 3 El.) bleibt die Feldstufe beim selben Punkt 0,2719 stehen (Schrittraster 0,1 + …; F_B 5,9·10⁻³ beim nächsten Schritt, steil wachsend: 7,3·10⁻³ bei 0,30, 2,6·10⁻² bei 0,40). Die Wellenfront am Teilfeldrand ist dort 0,087 Wellen, Kippung vernachlässigbar. Der M6-T6-9-Linser trägt das volle Feld unter derselben Merit mit 0,033 Wellen (V6-Z2): Die Lösung existiert, der Controller findet sie ab der Einzellinse nicht mehr (der alte Weg führte über die Merit mit Referenz durch die Hilfsebene 0,15 mm vor dem Objekt) | Feld und Apertur gemeinsam (Diagonale, F5); Einfügen nach Feldbedarf (Plattenlagen mit großer Hauptstrahlhöhe, F4); als Workaround Start aus einem feldtauglichen Zwischenstand (V6/V7) |
| F15 | **Aperturstufe nach den Farbstufen ohne Fortschritt** | V7: nach p4/p5 (11 El., polychromatisch, volles Feld) wird jeder Start der Aperturstufe abgelehnt, und es gibt keinen Spalt ≥ 0,7 mm für ein erzwungenes Einfügen; V3 kommt bei 27 % Feld von 0,50 auf 0,535 | Farbe und Apertur gemeinsam; erzwungenes Teilen statt Einfügen auch in der Aperturstufe |
| | *Stand nach M6-T8* ([Bericht](m6-t8-apertur-feld.md)): **F14** Ursache geklärt (Feldfehler für die Platten unsichtbar, kein Bedarf; 27 % = letzter Rasterpunkt vor F_B = F_max); mit der Feldregel `synth --field-rule` bei NA 0,5 behoben (100 % Feld, 0,039/0,022/0,059 Wellen, aber 25 El.), bei NA 0,855 nur 23 % (`--joint-aperture-field --field-rule --insertion-improve 4`). **F15** nicht bearbeitet. **F16** adressiert durch `--insertion-improve` | | |
| F16 | **Mehr Elemente verschlechtern** | V2 (μ/2 in der Feldstufe: 5 → 18 El.), V5 (18 El.), V8 (24 El.), V9 (28 El.) enden schlechter als ihre schlankeren Geschwister; die Feinoptimierung ohne Geisterplatten konvergiert bei vielen Elementen nicht (`max_iterations`, F10) | Elementbudget klein halten; Einfügen nur mit nachgewiesener Verbesserung des realen Systems (F_B ohne Geisterplatten) |

## 8. Variante V9 (Feld bei NA 0,4)

Die Feldstufe beginnt bei NA 0,4 mit nur 3 Elementen (p1 erreicht NA 0,4 ohne Einfügen) und bleibt am selben Punkt 0,2719 stehen wie bei NA 0,5 – die Reserve von 0,1 in der NA hilft nicht (F5, F14). Die anschließende Aperturstufe kommt bei 27 % Feld bis NA 0,6325, aber über erzwungene Einfügungen und Einfüge-Serien auf 26 Elemente; das Endsystem (28 El.) hat 1,3 Wellen RMS (F16). Der Zwischenkandidat Z2 (3 El., NA 0,4, 27 % Feld, monochromatisch) ist mit 0,040 Wellen beugungsbegrenzt.

## 9. Offene Punkte und Empfehlungen

1. **Feldstufe (F14)** ist der neue Hauptengpass: gemeinsame Homotopie in Apertur und Feld oder ein Bedarfskriterium für Feldfehler (F3/F4) – Controller-Aufgabe, nicht im Skript lösbar. Bis dahin: Startsysteme, die das Feld schon tragen (V6/V7, Patentsysteme des Maintainers, M6-T6-Bericht Punkt 1).
2. **Farbe früh** (V3, V7) ist die wirksamste Kombination; der nächste Versuch sollte „Farbe früh“ mit der Apertur zuerst verbinden (V2-M1 polychromatisch weiter), das ist hier aus Zeitgründen nicht gerechnet.
3. Die Aperturstufe bei vollem Feld endet wie in M6-T6 bei NA ≈ 0,57 (V6) – unabhängig von der Referenzsphäre.
4. Standard der Referenzsphäre für neu erzeugte Startsysteme (Datei ohne Feld = `last_surface`) bleibt eine Entscheidung des Maintainers (M6-T7, Folgerungen); das Skript setzt `image_infinite` ausdrücklich.

## 10. Dateien

Reproduktion (je Variante, Ausgabe in `docs/ergebnisse/m6-t6b/<Variante>`; Optionen aus Abschnitt 2):

```
cmake --preset release-cuda && cmake --build --preset release-cuda
python tools/m6t6_zielspezifikation.py --gpu --out docs/ergebnisse/m6-t6b/V1_basis
python tools/m6t6_zielspezifikation.py --gpu --out docs/ergebnisse/m6-t6b/V3_farbe_frueh --farbregel --farbe-frueh --synth-arg "--stall 0"
python tools/m6t6_zielspezifikation.py --gpu --out docs/ergebnisse/m6-t6b/V7_start_z2alt_farbe_frueh --start docs/ergebnisse/m6-t6/p2_final.json --farbregel --farbe-frueh --synth-arg "--stall 0"
python tools/m6t6_zielspezifikation.py --tabellen docs/ergebnisse/m6-t6b/V1_basis
```

Die monochromatischen Kandidaten M1 (V2) und M3 (V4) entstehen aus den Stufenergebnissen `p1_final.json` bzw. `p3_r15_restart.json` mit den Bausteinen des Skripts (`set_field` auf 10 %, `polish(..., spectrum='mono')`, `evaluate`).

| Datei | Inhalt |
|---|---|
| `m6-t6b/<Variante>/m6t6_lauf.json`, `m6t6_zeiten.json`, `lauf.log` | Zusammenfassung der Stufen, Laufzeiten aller Programmaufrufe, Konsolenprotokoll |
| `m6-t6b/<Variante>/p*_r*.txt` | Textprotokolle der synth-Läufe (die JSON-Protokolle `mikroptim.synth/1` sind wegen der Größe nicht im Repository; `m6t6_lauf.json` fasst Status, Punkt und Elemente jeder Runde zusammen) |
| `m6-t6b/<Variante>/p0_start.json`, Stufenergebnisse, `p6_s*.json`, `p7.json` | Start (mit `reference_sphere`), Ergebnis jeder Stufe, Feinoptimierung, eingerastetes Endsystem |
| `m6-t6b/<Variante>/{p8,z2,m1,m3}_{pupille,unendlich,psf,vektor}.json`, `*_vollfeld_pupille.json` | Bewertungen `mikroptim eval --json` (bei vollem Feld nur die Referenz Austrittspupille) |
| `m6-t6b/m11t7/` | Zwischenstand und Vergleich CMA gegen `starts` |

Die Verzeichnisse sind mit `--archiv` verkleinert (Nachoptimierungen und Kandidaten der erzwungenen Einfügungen entfernt, Pfade relativ), zusätzlich ohne JSON-Protokolle der synth-Läufe und ohne PSF-Dateien bei vollem Feld (6,8 MB); ein erneuter Lauf erzeugt alles neu.
